"""experiments_synth.py -- reproduce the paper's synthetic results on
closed-form Bures-Wasserstein ground truth. Saves results_synth.json with:
frontier, gap_geometry, dimension, scaling, hybrid_dim, nongauss.
"""
import time
import numpy as np
import ot
import otkit as K


def _frontier_gaussian(d=5, n=1200, seed=7, fast=False):
    if fast: n = 700
    pr = K.sample_gaussian_problem(d=d, n=n, seed=seed)
    X, Y = pr["X"], pr["Y"]
    a = np.ones(n) / n; b = np.ones(n) / n
    M = ot.dist(X, Y, metric="sqeuclidean"); Mmax = M.max()
    W2c = K.bures_w2sq(pr["m0"], pr["C0"], pr["m1"], pr["C1"])
    rows = []
    Gex, lg = ot.emd(a, b, M, log=True, numItermax=500000)
    W2d = float(np.sum(Gex * M))
    rank_list = [2, 25, 100] if fast else [2, 5, 10, 25, 50, 100, 200]
    proj_list = [20, 200] if fast else [5, 20, 50, 200, 1000]
    mb_list = [32, 256] if fast else [16, 32, 64, 128, 256, 512]

    def add(method, fam, est, U, L, two, t, mem):
        rows.append(dict(method=method, family=fam, est=est, U=U, L=L, two_sided=two,
                         time=t, mem=mem, rel_err_disc=abs(est - W2d) / W2d,
                         rel_err_cont=abs(est - W2c) / W2c))
    add("Exact (LP)", "exact", W2d, W2d, float(a @ lg["u"] + b @ lg["v"]), True, 0.0, K.mem_mb(2 * n * n))
    for er in [0.2, 0.1, 0.05, 0.02, 0.01]:
        est, U, L, dt = K.sinkhorn_gap(M, a, b, er, use_gpu=True)
        add("Sinkhorn", "entropic", est, U, L, True, dt, K.mem_mb(2 * n * n))
    for r in rank_list:
        if r >= n: continue
        t0 = time.perf_counter(); U = K.lowrank_ub(X, Y, a, b, r, M); add("Low-rank", "lowrank", U, U, None, False, time.perf_counter() - t0, K.mem_mb(2 * n * r))
    for Lp in proj_list:
        t0 = time.perf_counter(); sw2 = K.sliced_lb(X, Y, n_proj=Lp); add("Sliced", "sliced", sw2, None, sw2, False, time.perf_counter() - t0, K.mem_mb(n))
    for mb in mb_list:
        if mb > n: continue
        t0 = time.perf_counter(); U = K.minibatch_ub(M, n, m=mb); add("Minibatch", "minibatch", U, U, None, False, time.perf_counter() - t0, K.mem_mb(mb * mb))
    return dict(rows=rows, W2_disc=W2d, W2_cont=W2c)


def _gap_geometry(use_gpu=True):
    pr = K.sample_gaussian_problem(d=2, n=1500, seed=20)
    X, Y = pr["X"], pr["Y"]; n = X.shape[0]
    a = np.ones(n) / n; b = np.ones(n) / n
    M = ot.dist(X, Y, metric="sqeuclidean")
    w2d = float(ot.emd2(a, b, M, numItermax=500000))
    Cmid = K.bw_geodesic_cov(pr["C0"], pr["C1"], 0.5)
    rows = []
    # Sweep eps down to 0.002*Mmax; the log-domain solve keeps the certified
    # bounds valid where the plain kernel would underflow.
    for er in [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
        est, U, L, _, Ps = K.sinkhorn_gap(M, a, b, er, use_gpu=use_gpu,
                                          numItermax=100000, stopThr=1e-9,
                                          return_plan=True)
        Yhat = (Ps @ Y) / np.maximum(Ps.sum(1, keepdims=True), 1e-300)
        Z = 0.5 * (X + Yhat); Zc = Z - Z.mean(0)
        geo = K.bures_distance_sq((Zc.T @ Zc) / n, Cmid)
        rows.append(dict(eps_rel=er, gap=U - L, rel_gap=(U - L) / w2d, geo_dist=geo))
    return dict(rows=rows, w2_disc=w2d)


def _dimension(fast=False):
    rows = []
    dims = [1, 2, 4, 8, 16] if fast else [1, 2, 4, 8, 16, 32, 64, 128]
    for d in dims:
        pr = K.sample_gaussian_problem(d=d, n=2000, seed=30 + d)
        X, Y = pr["X"], pr["Y"]; ad = np.ones(2000) / 2000
        Md = ot.dist(X, Y, metric="sqeuclidean")
        w2disc = float(ot.emd2(ad, ad, Md))
        w2cont = K.bures_w2sq(pr["m0"], pr["C0"], pr["m1"], pr["C1"])
        sw2 = K.sliced_lb(X, Y, n_proj=200)
        rows.append(dict(d=d, sliced_rel_gap=(w2disc - sw2) / w2disc,
                         stat_rel_gap=abs(w2disc - w2cont) / w2cont))
    return dict(rows=rows)


def _scaling(use_gpu=True, fast=False):
    out = {"exact": [], "sinkhorn": [], "sliced": [], "lowrank": []}
    Nsync = [200, 500, 1000, 2000] if fast else [200, 500, 1000, 2000, 4000, 8000, 20000, 50000]
    for nn in Nsync:
        pr = K.sample_gaussian_problem(d=5, n=nn, seed=99)
        X, Y = pr["X"], pr["Y"]; a = np.ones(nn) / nn; b = np.ones(nn) / nn
        if nn <= 4000:
            M = ot.dist(X, Y, metric="sqeuclidean")
            t0 = time.perf_counter(); ot.emd2(a, b, M); out["exact"].append(dict(n=nn, time=time.perf_counter() - t0))
        if nn <= 8000:
            M = ot.dist(X, Y, metric="sqeuclidean")
            _, _, _, dt = K.sinkhorn_gap(M, a, b, 0.05, use_gpu=use_gpu); out["sinkhorn"].append(dict(n=nn, time=dt))
            t0 = time.perf_counter(); K.lowrank_ub(X, Y, a, b, 20, M); out["lowrank"].append(dict(n=nn, time=time.perf_counter() - t0))
            del M
        else:
            M = ot.dist(X, Y, metric="sqeuclidean")
            _, _, _, dt = K.sinkhorn_gap(M, a, b, 0.05, use_gpu=use_gpu); out["sinkhorn"].append(dict(n=nn, time=dt)); del M
        t0 = time.perf_counter(); K.sliced_lb(X, Y, n_proj=200); out["sliced"].append(dict(n=nn, time=time.perf_counter() - t0))
    return out


def _hybrid_dim(use_gpu=True, fast=False):
    rows = []
    n = 700 if fast else 1200
    for d in ([1, 2, 8] if fast else [1, 2, 4, 8, 16]):
        pr = K.sample_gaussian_problem(d=d, n=n, seed=11)
        X, Y = pr["X"], pr["Y"]; a = np.ones(n) / n; b = np.ones(n) / n
        M = ot.dist(X, Y, metric="sqeuclidean")
        OTstar = float(ot.emd2(a, b, M, numItermax=500000))
        L = K.sliced_lb(X, Y, n_proj=200); U = K.minibatch_ub(M, n, m=256)
        _, Ue, Le, _ = K.sinkhorn_gap(M, a, b, 0.02, use_gpu=use_gpu)
        rows.append(dict(d=d, OTstar=OTstar, relgap_hyb=(U - L) / OTstar,
                         dual_deficit=(OTstar - L) / OTstar, primal_excess=(U - OTstar) / OTstar,
                         relgap_ent=(Ue - Le) / OTstar))
    return dict(rows=rows)


def _two_moons(n, seed, rot=0.0, shift=(0.0, 0.0), noise=0.08):
    g = np.random.default_rng(seed); n1 = n // 2; n2 = n - n1
    t1 = np.pi * g.random(n1); m1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = np.pi * g.random(n2); m2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    Z = np.vstack([m1, m2]) + noise * g.standard_normal((n, 2))
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return Z @ R.T + np.asarray(shift)


def _nongauss():
    n = 1200
    X = _two_moons(n, 1, rot=0.0); Y = _two_moons(n, 2, rot=np.pi / 2, shift=(1.5, 0.3))
    fr = K.run_frontier(X, Y, N=n, use_gpu=True, seed=0)
    hc = K.composed_certificate(X, Y, N=n)
    return dict(OTstar=fr["OTstar"], rows=fr["rows"], hybrid=hc,
                X=X[:400].tolist(), Y=Y[:400].tolist())


def run_synth(log=print, fast=False, use_gpu=True):
    OUT = {}
    log("[synth A] Gaussian fidelity-cost frontier (d=5)...")
    OUT["frontier"] = _frontier_gaussian(fast=fast)
    log("[synth B] certified gap vs geodesic distortion (d=2)...")
    OUT["gap_geometry"] = _gap_geometry(use_gpu=use_gpu)
    log("[synth C] dimensional deficit (d=1..128)...")
    OUT["dimension"] = _dimension(fast=fast)
    log("[synth D] cost of certifiability (scaling)...")
    OUT["scaling"] = _scaling(use_gpu=use_gpu, fast=fast)
    log("[synth E] composed certificate vs dimension...")
    OUT["hybrid_dim"] = _hybrid_dim(use_gpu=use_gpu, fast=fast)
    log("[synth F] non-Gaussian (two-moons) frontier...")
    OUT["nongauss"] = _nongauss()
    return OUT
