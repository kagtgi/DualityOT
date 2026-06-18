"""otkit.py -- optimal-transport toolkit for the duality-gap study.

Everything here is import-safe and has no side effects. Covers:
  * Kantorovich gap machinery (rounding for U, c-transform for L)
  * estimators for every family (exact LP, Sinkhorn, low-rank, sliced, minibatch)
  * a GPU-aware Sinkhorn (uses torch+CUDA when available and worthwhile)
  * reusable runners: frontier, composed certificate, dimension sweep, scaling
  * Gaussian / Bures-Wasserstein helpers (closed-form ground truth)
"""
import time
import numpy as np
import ot

# ---- optional GPU backend -------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _HAS_CUDA = False


def gpu_info():
    if _HAS_CUDA:
        return "cuda:" + torch.cuda.get_device_name(0)
    return "cpu" + (" (torch present)" if _HAS_TORCH else " (no torch)")


# Smallest problem for which moving to the GPU pays for the host<->device copy.
GPU_MIN_N = 1500
# All Sinkhorn solves run in the log domain (numerically stable) so that the
# certified bounds stay valid even at tiny eps, where the kernel exp(-M/reg)
# would otherwise underflow to zero with the plain (Knopp) iteration.
SINKHORN_METHOD = "sinkhorn_log"


# ============================ gap machinery ============================
def round_to_marginals(P, a, b):
    """Altschuler, Niles-Weed & Rigollet (2017): project a near-coupling onto
    the transport polytope so <C,P> is a valid UPPER bound on OT*.

    The two min(.,1) scalings guarantee er, ec >= 0 in exact arithmetic; we clamp
    them to kill round-off noise. When the plan is already (essentially) feasible
    the residual mass is ~0, so we return it as-is instead of dividing by a tiny
    floor -- the latter would blow the correction up to ~1e300 for a high-rank or
    converged plan whose marginals already match."""
    P = np.asarray(P, float)
    x = np.minimum(a / np.maximum(P.sum(1), 1e-300), 1.0); P = P * x[:, None]
    y = np.minimum(b / np.maximum(P.sum(0), 1e-300), 1.0); P = P * y[None, :]
    er = np.maximum(a - P.sum(1), 0.0); ec = np.maximum(b - P.sum(0), 0.0)
    s = float(er.sum())
    if s <= 1e-15:
        return P
    return P + np.outer(er, ec) / s


def dual_lower_bound(M, g, a, b):
    """Feasible-dual LOWER bound via the c-transform of potential g:
    f_i = min_j (M_ij - g_j) => f_i + g_j <= M_ij, so a.f + b.g <= OT*."""
    f = (M - g[None, :]).min(axis=1)
    return float(a @ f + b @ g)


def mem_mb(num_floats):
    return num_floats * 8 / 1e6


# ============================ estimators ============================
def exact_ot(M, a, b):
    return float(ot.emd2(a, b, M, numItermax=2_000_000))


def _g_potential_np(log, reg):
    """Dual potential g (length m) from a POT log dict, robust to the solver
    used: 'sinkhorn_log' stores log_v directly, plain 'sinkhorn' stores v."""
    if "log_v" in log:
        return reg * np.asarray(log["log_v"])
    return reg * np.log(np.maximum(np.asarray(log["v"]), 1e-300))


def _g_potential_torch(log, reg):
    if "log_v" in log:
        return reg * log["log_v"]
    return reg * torch.log(torch.clamp(log["v"], min=1e-300))


def sinkhorn_gap(M, a, b, eps_rel, use_gpu=True, numItermax=20000, stopThr=1e-9,
                 return_plan=False):
    """One (log-stabilized) Sinkhorn solve -> a certified two-sided gap.

    Returns (est, U, L, time_s), or (est, U, L, time_s, P) when return_plan.
    U is the cost of the marginal-rounded plan (a rigorous primal UPPER bound);
    L is a.f + b.g for the c-transform pair (a rigorous dual LOWER bound), valid
    for every eps because the solve is done in the log domain. Uses the GPU when
    available and the problem is large enough to amortize the host<->device copy.
    """
    Mmax = float(M.max()); reg = eps_rel * Mmax
    n = M.shape[0]
    if use_gpu and _HAS_CUDA and n >= GPU_MIN_N:
        try:
            Mt = torch.as_tensor(M, dtype=torch.float64, device="cuda")
            at = torch.as_tensor(a, dtype=torch.float64, device="cuda")
            bt = torch.as_tensor(b, dtype=torch.float64, device="cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            Ps, log = ot.sinkhorn(at, bt, Mt, reg, method=SINKHORN_METHOD, log=True,
                                  numItermax=numItermax, stopThr=stopThr)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            g = _g_potential_torch(log, reg)
            f = (Mt - g[None, :]).min(dim=1).values
            L = float((at @ f + bt @ g).item())
            Pr = _round_torch(Ps, at, bt)
            U = float((Pr * Mt).sum().item())
            est = float((Ps * Mt).sum().item())
            P = Ps.detach().cpu().numpy() if return_plan else None
            del Mt, at, bt, Pr, Ps; torch.cuda.empty_cache()
            return (est, U, L, dt, P) if return_plan else (est, U, L, dt)
        except Exception:
            try: torch.cuda.empty_cache()
            except Exception: pass
            # fall through to CPU
    # CPU path
    t0 = time.perf_counter()
    Ps, log = ot.sinkhorn(a, b, M, reg, method=SINKHORN_METHOD, log=True,
                          numItermax=numItermax, stopThr=stopThr)
    dt = time.perf_counter() - t0
    g = _g_potential_np(log, reg)
    U = float(np.sum(round_to_marginals(Ps, a, b) * M))
    L = dual_lower_bound(M, g, a, b)
    est = float(np.sum(Ps * M))
    return (est, U, L, dt, np.asarray(Ps)) if return_plan else (est, U, L, dt)


def _round_torch(P, a, b):
    x = torch.clamp(a / torch.clamp(P.sum(1), min=1e-300), max=1.0); P = P * x[:, None]
    y = torch.clamp(b / torch.clamp(P.sum(0), min=1e-300), max=1.0); P = P * y[None, :]
    er = torch.clamp(a - P.sum(1), min=0.0); ec = torch.clamp(b - P.sum(0), min=0.0)
    s = float(er.sum().item())
    if s <= 1e-15:
        return P
    return P + torch.outer(er, ec) / s


def sliced_lb(X, Y, n_proj=200, seed=1):
    """Sliced W_2^2 -- a feasible LOWER bound. Each projected W_2^2 is <= the
    full W_2^2, so their average SW_2^2 <= W_2^2 for ANY number of projections
    (the bound is exact-sided, not just in expectation)."""
    sw = ot.sliced_wasserstein_distance(X, Y, n_projections=n_proj, p=2, seed=seed)
    return float(sw) ** 2


def lowrank_ub(X, Y, a, b, r, M, numItermax=300):
    """Rank-<=r factored coupling -> feasible UPPER bound."""
    Ga, Gb, _, _ = ot.factored_optimal_transport(X, Y, a, b, r=r, reg=0.0,
                                                  numItermax=numItermax, log=True)
    h = Ga.sum(0)
    P = Ga @ np.diag(1.0 / np.maximum(h, 1e-300)) @ Gb
    P = round_to_marginals(P, a, b)
    return float(np.sum(P * M))


def minibatch_ub(M, n, m=256, seed=7, max_batches=64):
    """Averaged minibatch OT -> (biased) feasible UPPER bound."""
    m = min(m, n)
    K = min(max(1, n // m), max_batches)
    g = np.random.default_rng(seed); vals = []
    for _ in range(K):
        ix = g.choice(n, m, replace=False); iy = g.choice(n, m, replace=False)
        ab = np.ones(m) / m
        vals.append(ot.emd2(ab, ab, M[np.ix_(ix, iy)]))
    return float(np.mean(vals))


def minibatch_ub_xy(X, Y, m=256, seed=7, max_batches=64):
    """Minibatch OT directly from POINT CLOUDS -- builds only m x m cost matrices,
    never the full n x n. This is what lets the upper-bound family scale to
    millions of points (Section: memory stress test): peak memory is O(m^2),
    independent of n."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n = min(X.shape[0], Y.shape[0]); m = min(m, n)
    K = min(max(1, n // m), max_batches)
    g = np.random.default_rng(seed); vals = []
    ab = np.ones(m) / m
    for _ in range(K):
        ix = g.choice(X.shape[0], m, replace=False)
        iy = g.choice(Y.shape[0], m, replace=False)
        Mb = ot.dist(X[ix], Y[iy], metric="sqeuclidean")
        vals.append(ot.emd2(ab, ab, Mb))
    return float(np.mean(vals))


# ============================ alternative fidelity proxies (baselines) ======
def sinkhorn_divergence(X, Y, eps_rel=0.05, numItermax=2000, stopThr=1e-7):
    """Debiased Sinkhorn divergence S_eps = OT_eps(mu,nu) - 1/2 OT_eps(mu,mu)
    - 1/2 OT_eps(nu,nu) (Genevay et al., 2018; Feydy et al., 2019). A popular
    *point* fidelity proxy -- used here only as a baseline to contrast with the
    certified two-sided gap (it is not a bound on OT*)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    nx, ny = X.shape[0], Y.shape[0]
    a = np.ones(nx) / nx; b = np.ones(ny) / ny
    Mxy = ot.dist(X, Y, metric="sqeuclidean")
    reg = eps_rel * float(Mxy.max())
    Mxx = ot.dist(X, X, metric="sqeuclidean")
    Myy = ot.dist(Y, Y, metric="sqeuclidean")
    kw = dict(method=SINKHORN_METHOD, numItermax=numItermax, stopThr=stopThr)
    sxy = float(ot.sinkhorn2(a, b, Mxy, reg, **kw))
    sxx = float(ot.sinkhorn2(a, a, Mxx, reg, **kw))
    syy = float(ot.sinkhorn2(b, b, Myy, reg, **kw))
    return sxy - 0.5 * sxx - 0.5 * syy


def spearman(x, y):
    """Spearman rank correlation (no SciPy dependency): Pearson on ranks."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ============================ memory / OOM stress test ============================
def _gpu_total_gb():
    if _HAS_CUDA:
        try:
            return float(torch.cuda.get_device_properties(0).total_memory) / 1e9
        except Exception:
            return None
    return None


def memory_stress_test(d=5, shift=2.0,
                       dense_Ns=(20000, 30000, 40000, 50000, 55000, 64000,
                                 80000, 100000, 120000),
                       light_Ns=(10000, 50000, 100000, 300000, 1000000),
                       dtypes=("float64", "float32"),
                       eps_rel=0.05, n_proj=100, mb=256, seed=99,
                       use_gpu=True, numItermax=1000):
    """Locate the dense-method OOM wall in BOTH precisions, and show one-sided
    methods scale past it.

    DENSE path (the certifiable two-sided route): for each precision and each n we
    build the n x n squared-Euclidean cost matrix *on the GPU* and run one
    log-domain Sinkhorn solve, recording wall-clock and peak GPU memory -- or, when
    the allocation exceeds VRAM, we catch the out-of-memory error and stop,
    recording the wall at that n. Running float64 (certification-grade) and float32
    shows the wall is fundamental: halving the bytes only buys ~sqrt(2) in n.

    LIGHT path (one-sided surrogates): sliced (point-cloud, no matrix) and
    minibatch (only m x m matrices) are run out to n = 1e6, where the dense path
    is many terabytes. Their peak memory is O(n d) and O(m^2) respectively.

    Returns dict(dense={dtype: [...]}, oom_at={dtype: int|None}, light=[...],
                 gpu_total_gb, device, bytes={dtype:int}). Everything is wrapped so
    a failure never crashes the caller.
    """
    out = {"dense": {}, "oom_at": {}, "light": [],
           "gpu_total_gb": _gpu_total_gb(),
           "device": "cuda" if (use_gpu and _HAS_CUDA) else "cpu",
           "bytes": {"float64": 8, "float32": 4}}
    rng = np.random.default_rng(seed)
    have_cuda = bool(use_gpu and _HAS_CUDA)
    OOM = RuntimeError
    if _HAS_TORCH:
        OOM = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", RuntimeError)
        torch_dtype = {"float64": torch.float64, "float32": torch.float32}

    if not have_cuda:
        # CPU-only host: a single small float64 sweep so the figure still has data,
        # without wedging the machine on a huge dense matrix.
        out["dense"]["float64"] = []; out["oom_at"]["float64"] = None
        for N in [n for n in dense_Ns if n <= 6000]:
            rec = dict(n=int(N), ok=False)
            try:
                pr = sample_gaussian_problem(d=d, n=N, seed=seed, shift=shift)
                M = cost_matrix(pr["X"], pr["Y"]); a = np.ones(N) / N
                t0 = time.perf_counter()
                sinkhorn_gap(M, a, a, eps_rel, use_gpu=False, numItermax=numItermax)
                rec.update(time=time.perf_counter() - t0,
                           mem_gb=mem_mb(2 * N * N) / 1e3, ok=True); del M
            except (RuntimeError, MemoryError) as e:
                rec["error"] = str(e)[:80]
            out["dense"]["float64"].append(rec)
    else:
        for dt_name in dtypes:
            dt = torch_dtype[dt_name]
            out["dense"][dt_name] = []; out["oom_at"][dt_name] = None
            for N in dense_Ns:
                rec = dict(n=int(N), ok=False, time=None, mem_gb=None)
                try:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    Xt = torch.as_tensor(rng.standard_normal((N, d)), dtype=dt, device="cuda")
                    Yt = torch.as_tensor(rng.standard_normal((N, d)) + shift, dtype=dt, device="cuda")
                    at = torch.full((N,), 1.0 / N, dtype=dt, device="cuda"); bt = at.clone()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    Mt = torch.cdist(Xt, Yt) ** 2
                    reg = eps_rel * float(Mt.max())
                    ot.sinkhorn(at, bt, Mt, reg, method=SINKHORN_METHOD, log=True,
                                numItermax=numItermax, stopThr=1e-7)
                    torch.cuda.synchronize(); dt_t = time.perf_counter() - t0
                    rec.update(time=dt_t,
                               mem_gb=float(torch.cuda.max_memory_allocated()) / 1e9,
                               ok=True)
                    del Mt, Xt, Yt, at, bt; torch.cuda.empty_cache()
                except (OOM, RuntimeError, MemoryError) as e:
                    msg = str(e).lower()
                    rec["error"] = "OOM" if ("out of memory" in msg or "alloc" in msg
                                             or isinstance(e, (OOM, MemoryError))) else str(e)[:90]
                    if out["oom_at"][dt_name] is None:
                        out["oom_at"][dt_name] = int(N)
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                out["dense"][dt_name].append(rec)
                if not rec["ok"]:
                    break  # first failure = the wall; larger n only fails harder

    # ---- light sweep (one-sided, point-cloud only) ----
    for N in light_Ns:
        try:
            X = rng.standard_normal((N, d)); Y = rng.standard_normal((N, d)) + shift
            lp = n_proj if N <= 100000 else max(20, n_proj // 2)  # trim projections at 1e6
            t0 = time.perf_counter(); sw = sliced_lb(X, Y, n_proj=lp)
            t_sl = time.perf_counter() - t0
            t0 = time.perf_counter(); U = minibatch_ub_xy(X, Y, m=mb)
            t_mb = time.perf_counter() - t0
            out["light"].append(dict(n=int(N), sliced=sw, minibatch=U,
                                     sliced_time=t_sl, minibatch_time=t_mb,
                                     mem_gb=float((2 * N * d + N * lp) * 8) / 1e9))
            del X, Y
        except (RuntimeError, MemoryError) as e:
            out["light"].append(dict(n=int(N), error=str(e)[:90])); break
    return out


# ============================ runners ============================
def cost_matrix(X, Y):
    return ot.dist(X, Y, metric="sqeuclidean")


def match_samples(X, Y, N=None, seed=0):
    """Subsample X and Y to a common size N (default: the smaller of the two)."""
    g = np.random.default_rng(seed)
    n = min(X.shape[0], Y.shape[0])
    if N is not None:
        n = min(n, int(N))
    ix = g.choice(X.shape[0], n, replace=False)
    iy = g.choice(Y.shape[0], n, replace=False)
    return X[ix], Y[iy]


def run_frontier(X, Y, N=1500, eps_list=(0.2, 0.1, 0.05, 0.02, 0.01),
                 rank_list=(2, 5, 10, 25, 50, 100),
                 proj_list=(5, 20, 50, 200, 1000),
                 mb_list=(16, 32, 64, 128, 256), use_gpu=True, seed=0):
    """Full fidelity-cost frontier vs the EXACT DISCRETE optimum on these samples."""
    X, Y = match_samples(X, Y, N=N, seed=seed)
    n = X.shape[0]
    a = np.ones(n) / n; b = np.ones(n) / n
    M = cost_matrix(X, Y)
    OTstar = exact_ot(M, a, b)
    rows = []

    def add(method, fam, est, U, L, two, t, mem):
        rows.append(dict(method=method, family=fam, est=est, U=U, L=L, two_sided=two,
                         time=t, mem=mem, rel_err=abs(est - OTstar) / max(OTstar, 1e-12)))

    add("Exact (LP)", "exact", OTstar, OTstar, OTstar, True, 0.0, mem_mb(2 * n * n))
    for er in eps_list:
        est, U, L, dt = sinkhorn_gap(M, a, b, er, use_gpu=use_gpu)
        add("Sinkhorn", "entropic", est, U, L, True, dt, mem_mb(2 * n * n))
    for r in rank_list:
        if r >= n:
            continue
        t0 = time.perf_counter(); U = lowrank_ub(X, Y, a, b, r, M); dt = time.perf_counter() - t0
        add("Low-rank", "lowrank", U, U, None, False, dt, mem_mb(2 * n * r))
    for Lp in proj_list:
        t0 = time.perf_counter(); sw2 = sliced_lb(X, Y, n_proj=Lp); dt = time.perf_counter() - t0
        add("Sliced", "sliced", sw2, None, sw2, False, dt, mem_mb(n))
    for mb in mb_list:
        if mb > n:
            continue
        t0 = time.perf_counter(); U = minibatch_ub(M, n, m=mb); dt = time.perf_counter() - t0
        add("Minibatch", "minibatch", U, U, None, False, dt, mem_mb(mb * mb))
    return dict(OTstar=OTstar, n=n, rows=rows)


def composed_certificate(X, Y, N=1500, n_proj=200, mb=256, seed=0):
    """Cheap two-sided certificate G = U_minibatch - L_sliced (both near-linear)."""
    X, Y = match_samples(X, Y, N=N, seed=seed)
    n = X.shape[0]; M = cost_matrix(X, Y)
    a = np.ones(n) / n; b = np.ones(n) / n
    OTstar = exact_ot(M, a, b)
    t0 = time.perf_counter(); L = sliced_lb(X, Y, n_proj=n_proj); t_sl = time.perf_counter() - t0
    t0 = time.perf_counter(); U = minibatch_ub(M, n, m=mb); t_mb = time.perf_counter() - t0
    return dict(OTstar=OTstar, L=L, U=U, G=U - L, relgap=(U - L) / OTstar,
                dual_deficit=(OTstar - L) / OTstar, primal_excess=(U - OTstar) / OTstar,
                time=t_sl + t_mb)


def pca_reduce(Z, d, seed=0):
    """Project to top-d principal components (centered)."""
    from sklearn.decomposition import PCA
    d = int(min(d, Z.shape[1], Z.shape[0]))
    return PCA(n_components=d, random_state=seed).fit_transform(Z)


def dimension_sweep(Xfull, Yfull, dims=(2, 5, 10, 20, 50), n_proj=200, mb=256):
    """Replay the dimensional-deficit story on real data: PCA-reduce a fused
    embedding to each d, then measure the sliced and composed relative gaps."""
    Z = np.vstack([Xfull, Yfull])
    nX = Xfull.shape[0]
    rows = []
    for d in dims:
        Zd = pca_reduce(Z, d)
        Xd, Yd = Zd[:nX], Zd[nX:]
        n = min(Xd.shape[0], Yd.shape[0])
        Xd, Yd = Xd[:n], Yd[:n]
        M = cost_matrix(Xd, Yd); a = np.ones(n) / n; b = np.ones(n) / n
        OTstar = exact_ot(M, a, b)
        L = sliced_lb(Xd, Yd, n_proj=n_proj)
        U = minibatch_ub(M, n, m=mb)
        rows.append(dict(d=int(d), OTstar=OTstar,
                         sliced_rel_gap=(OTstar - L) / OTstar,
                         hybrid_relgap=(U - L) / OTstar,
                         dual_deficit=(OTstar - L) / OTstar,
                         primal_excess=(U - OTstar) / OTstar))
    return dict(rows=rows)


def dimension_sweep_embedded(Xemb, Yemb, dims=(2, 4, 8, 16, 32), N=2000,
                             n_proj=500, mb=256, label="embed", seed=0):
    """Dimensional-deficit sweep on a PRECOMPUTED embedding: for each d we take
    the first d coordinates as-is (no re-projection), so a nonlinear embedding
    (e.g. a diffusion map) is tested in its own geometry. This is the test of
    whether the 1-1/d sliced deficit is a Gaussian/linear artifact or survives in
    a nonlinear latent space. Reports the sliced and composed relative gaps vs the
    exact discrete optimum at each d."""
    g = np.random.default_rng(seed)
    nX, nY = Xemb.shape[0], Yemb.shape[0]
    n = min(nX, nY, int(N))
    ix = g.choice(nX, n, replace=False); iy = g.choice(nY, n, replace=False)
    Xe, Ye = np.asarray(Xemb)[ix], np.asarray(Yemb)[iy]
    a = np.ones(n) / n; b = np.ones(n) / n
    rows = []
    for d in dims:
        d = int(min(d, Xe.shape[1]))
        Xd, Yd = Xe[:, :d], Ye[:, :d]
        M = cost_matrix(Xd, Yd)
        OTstar = exact_ot(M, a, b)
        L = sliced_lb(Xd, Yd, n_proj=n_proj); U = minibatch_ub(M, n, m=mb)
        rows.append(dict(d=d, OTstar=OTstar,
                         sliced_rel_gap=(OTstar - L) / OTstar,
                         hybrid_relgap=(U - L) / OTstar))
        del M
    return dict(rows=rows, label=label, n=n)


def raw_genespace_point(X, Y, N=1500, n_proj=500, mb=256, seed=0):
    """One OT measurement in the FULL native (raw gene) space -- no PCA. Returns
    the exact OT*, the sliced relative deficit, and the composed relative gap, to
    show the framework applies unchanged at very high ambient dimension."""
    g = np.random.default_rng(seed)
    nX, nY = X.shape[0], Y.shape[0]
    n = min(nX, nY, int(N))
    ix = g.choice(nX, n, replace=False); iy = g.choice(nY, n, replace=False)
    Xs, Ys = np.asarray(X)[ix], np.asarray(Y)[iy]
    a = np.ones(n) / n; b = np.ones(n) / n
    M = cost_matrix(Xs, Ys)
    OTstar = exact_ot(M, a, b)
    L = sliced_lb(Xs, Ys, n_proj=n_proj); U = minibatch_ub(M, n, m=mb)
    return dict(d=int(Xs.shape[1]), n=n, OTstar=OTstar,
                sliced_rel_gap=(OTstar - L) / OTstar,
                hybrid_relgap=(U - L) / OTstar)


def scaling(Xfull, Yfull, Ns=(500, 1000, 2000, 4000, 8000, 16000, 32000),
            lp_max_n=4000, use_gpu=True):
    """Cost of certifiability at scale on real data. Exact LP only up to
    lp_max_n (CPU network simplex); Sinkhorn on GPU pushes far past it."""
    rng = np.random.default_rng(0)
    out = {"exact": [], "sinkhorn": [], "sliced": [], "lowrank": [], "minibatch": []}
    nmaxX, nmaxY = Xfull.shape[0], Yfull.shape[0]
    for N in Ns:
        if N > min(nmaxX, nmaxY):
            break
        ix = rng.choice(nmaxX, N, replace=False); iy = rng.choice(nmaxY, N, replace=False)
        X, Y = Xfull[ix], Yfull[iy]
        a = np.ones(N) / N; b = np.ones(N) / N
        if N <= lp_max_n:
            M = cost_matrix(X, Y)
            t0 = time.perf_counter(); exact_ot(M, a, b); out["exact"].append(dict(n=N, time=time.perf_counter() - t0))
        # GPU/CPU Sinkhorn (build cost on the fly to control memory)
        M = cost_matrix(X, Y)
        _, _, _, dt = sinkhorn_gap(M, a, b, 0.05, use_gpu=use_gpu)
        out["sinkhorn"].append(dict(n=N, time=dt))
        t0 = time.perf_counter(); sliced_lb(X, Y, n_proj=200); out["sliced"].append(dict(n=N, time=time.perf_counter() - t0))
        if N <= 8000:
            t0 = time.perf_counter(); lowrank_ub(X, Y, a, b, 20, M); out["lowrank"].append(dict(n=N, time=time.perf_counter() - t0))
        t0 = time.perf_counter(); minibatch_ub(M, N, m=256); out["minibatch"].append(dict(n=N, time=time.perf_counter() - t0))
        del M
    return out


# ============================ Gaussian / Bures (synthetic ground truth) ====
def random_spd(d, seed, cond=5.0):
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((d, d)))
    return (Q * np.linspace(1.0, cond, d)) @ Q.T


def sqrtm_spd(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def bures_w2sq(m0, C0, m1, C1):
    C0h = sqrtm_spd(C0); inner = sqrtm_spd(C0h @ C1 @ C0h)
    return float(np.sum((m0 - m1) ** 2) + max(np.trace(C0 + C1 - 2 * inner), 0.0))


def bw_map_matrix(C0, C1):
    C0h = sqrtm_spd(C0); C0hi = np.linalg.inv(C0h); inner = sqrtm_spd(C0h @ C1 @ C0h)
    return C0hi @ inner @ C0hi


def bw_geodesic_cov(C0, C1, t):
    T = bw_map_matrix(C0, C1); Mt = (1 - t) * np.eye(C0.shape[0]) + t * T
    return Mt @ C0 @ Mt.T


def bures_distance_sq(C0, C1):
    C0h = sqrtm_spd(C0); inner = sqrtm_spd(C0h @ C1 @ C0h)
    return float(max(np.trace(C0 + C1 - 2 * inner), 0.0))


def sample_gaussian_problem(d, n, seed, shift=2.0):
    g = np.random.default_rng(seed)
    m0 = g.standard_normal(d); m1 = g.standard_normal(d) + shift
    C0 = random_spd(d, seed + 1); C1 = random_spd(d, seed + 2)
    X = g.multivariate_normal(m0, C0, size=n); Y = g.multivariate_normal(m1, C1, size=n)
    return dict(m0=m0, C0=C0, m1=m1, C1=C1, X=X, Y=Y)
