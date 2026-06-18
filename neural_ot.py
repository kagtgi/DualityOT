r"""neural_ot.py -- a minimal, self-contained neural OT (W2) solver used purely
as an *uncertified* reference point on the fidelity-cost frontier.

We implement the two-ICNN maximin estimator of \citet{makkuva2020}. With cost
c(x,y)=||x-y||^2 and the max-correlation identity
    W_2^2 = E||x||^2 + E||y||^2 - 2 * max_{pi} E_pi <x,y>,
the optimal correlation solves the minimax
    max_pi E<x,y> = inf_{f cvx} sup_{g cvx} V(f,g),
    V(f,g) = E_mu[f(x)] + E_nu[ <y, grad g(y)> - f(grad g(y)) ],
so a converged (f,g) yields a neural W_2^2 estimate. The Brenier map T=grad f
pushes mu->nu, giving a map-transport cost as a second (also uncertified)
estimate. Neither f,g is dual-feasible nor is T marginal-feasible, so -- exactly
as the paper argues -- neural OT is "two-sided in form" but its gap is NOT
certified. This module exists to *measure where that uncertified gap sits*.

Everything is best-effort: import is always safe (torch is imported lazily), and
`neural_w2` returns None on any failure so the caller can simply skip the point.
"""
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:
    class ICNN(nn.Module):
        """Input-convex neural network (Amos et al., 2017). Convex in the input:
        hidden-to-hidden weights are clamped non-negative and the activation
        (softplus) is convex and non-decreasing; input skip connections may have
        any sign."""
        def __init__(self, dim, width=64, depth=2):
            super().__init__()
            self.Wx = nn.ModuleList([nn.Linear(dim, width)] +
                                    [nn.Linear(dim, width) for _ in range(depth)])
            self.Wz = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth)])
            self.final = nn.Linear(width, 1, bias=False)
            self.act = nn.Softplus()

        def forward(self, x):
            z = self.act(self.Wx[0](x))
            for i, Wz in enumerate(self.Wz):
                z = self.act(Wz(z) + self.Wx[i + 1](x))
            return self.final(z).squeeze(-1)

        def clamp(self):
            for Wz in self.Wz:
                Wz.weight.data.clamp_(min=0.0)
            self.final.weight.data.clamp_(min=0.0)


def _grad(net, y):
    """grad of a scalar-output ICNN wrt its input (batched), graph kept for the
    outer (f) update. Wrapped in enable_grad so it also works when called from a
    no_grad context (e.g. when evaluating the converged objective)."""
    with torch.enable_grad():
        y = y.requires_grad_(True)
        out = net(y).sum()
        g = torch.autograd.grad(out, y, create_graph=True)[0]
    return g


def _certify_neural(f, g, Xt, Yt, dev, m=400, seed=0):
    """Lemma 1 in practice: from the trained (uncertified) neural potentials,
    produce a VALID certified lower bound on the discrete OT* and the violation
    margin eps_viol. Done on an m x m subsample (numpy) so the cost is bounded.

    For ANY potential vectors u=f(Xs), v=g(Ys): the pair (u-eps_viol, v) with
    eps_viol = max_{i,j}(u_i+v_j-c_ij)_+ is dual-feasible, so
        L_flat = mean(u)+mean(v)-eps_viol  <=  OT*_disc   (Lemma 1).
    The c-transform of v, phi_i = min_j(c_ij - v_j), is the tightest feasible
    completion of v and gives L_ct >= L_flat (also valid). A feasible coupling
    from the Brenier map (nearest target + marginal rounding) gives U >= OT*_disc.
    Returns the certified bracket so neural OT becomes two-sided up to a margin.
    """
    try:
        import numpy as _np
        import otkit as _K
        from scipy.spatial import cKDTree
        import ot as _ot
        n = min(Xt.shape[0], Yt.shape[0]); m = int(min(m, n))
        gsel = _np.random.default_rng(seed)
        ix = gsel.choice(Xt.shape[0], m, replace=False)
        iy = gsel.choice(Yt.shape[0], m, replace=False)
        Xs = Xt[ix]; Ys = Yt[iy]
        with torch.no_grad():
            u = f(Xs).detach().cpu().numpy().astype(float).ravel()
            v = g(Ys).detach().cpu().numpy().astype(float).ravel()
        Xsr = Xs.detach().requires_grad_(True)
        Tx = torch.autograd.grad(f(Xsr).sum(), Xsr)[0].detach().cpu().numpy()
        Xs_np = Xs.detach().cpu().numpy(); Ys_np = Ys.detach().cpu().numpy()
        M = _K.cost_matrix(Xs_np, Ys_np)                       # m x m, ||.||^2
        a = _np.ones(m) / m; b = _np.ones(m) / m
        # violation margin of the raw pair, and the Lemma's flat-corrected bound
        eps_viol = float(_np.maximum(u[:, None] + v[None, :] - M, 0.0).max())
        L_flat = float(a @ u + b @ v - eps_viol)
        # tighter: c-transform of v (feasible by construction)
        phi_ct = (M - v[None, :]).min(axis=1)
        L_ct = float(a @ phi_ct + b @ v)
        # feasible coupling from the Brenier map: nearest target + marginal round
        _, idx = cKDTree(Ys_np).query(Tx, k=1)
        P = _np.zeros((m, m))
        for i, j in enumerate(idx):
            P[i, int(j)] += 1.0 / m
        P = _K.round_to_marginals(P, a, b)
        U = float(_np.sum(P * M))
        otstar = float(_ot.emd2(a, b, M))
        L = max(L_ct, L_flat)                                   # both valid; report tightest
        return dict(eps_viol=eps_viol, neural_L_flat=L_flat, neural_L=L,
                    neural_U=U, neural_G=float(U - L),
                    neural_relgap=float((U - L) / max(U, 1e-12)),
                    neural_otstar_sub=otstar, cert_m=m)
    except Exception as e:
        return dict(neural_cert_error=str(e)[:120])


def neural_w2(X, Y, W2_ref=None, width=128, depth=2, iters=4000, k_inner=10,
              lr=1e-3, use_gpu=True, seed=0):
    """Train the two-ICNN maximin and return an UNCERTIFIED W2^2 estimate.

    The reported `est` is the Brenier-map transport cost E|x - grad f(x)|^2 (a
    primal, always-non-negative quantity that is numerically far more stable than
    the raw maximin value, which is reported separately as `est_dual`). Returns
    dict(est, est_dual, U_map, time, mem_mb, converged, iters, device), or None if
    torch is unavailable. `converged` is a sanity flag: finite, positive, and
    within a generous band of W2_ref when provided. Always best-effort -- any
    failure returns a dict with converged=False rather than raising.
    """
    if not _HAS_TORCH:
        return None
    try:
        dev = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        torch.manual_seed(seed)
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
        Yt = torch.as_tensor(np.asarray(Y, np.float32), device=dev)
        d = Xt.shape[1]
        f = ICNN(d, width, depth).to(dev)
        g = ICNN(d, width, depth).to(dev)
        opt_f = torch.optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.9))
        # cosine decay stabilizes the late maximin iterations
        sch_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, iters)
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, iters)

        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        def V():
            gg = _grad(g, Yt)                       # grad g(Y): (n,d)
            corr = (Yt * gg).sum(1).mean()          # <y, grad g(y)>
            return f(Xt).mean() + corr - f(gg).mean()

        for _ in range(iters):
            for _ in range(k_inner):                # inner: maximize V over g
                opt_g.zero_grad(set_to_none=True)
                (-V()).backward()
                opt_g.step(); g.clamp()
            opt_f.zero_grad(set_to_none=True)       # outer: minimize V over f
            V().backward()
            opt_f.step(); f.clamp()
            sch_f.step(); sch_g.step()

        corr_val = float(V().item())   # V() needs autograd (grad of g); no no_grad here
        ex2 = float((Xt ** 2).sum(1).mean().item())
        ey2 = float((Yt ** 2).sum(1).mean().item())
        est_dual = ex2 + ey2 - 2.0 * corr_val       # maximin W2^2 (can be noisy)

        # Brenier map T = grad f : mu -> nu, transport cost E|x - grad f(x)|^2
        Xg = Xt.detach().requires_grad_(True)
        Tx = torch.autograd.grad(f(Xg).sum(), Xg)[0]
        U_map = float(((Xt - Tx) ** 2).sum(1).mean().item())

        dt = time.perf_counter() - t0
        mem_mb = (float(torch.cuda.max_memory_allocated()) / 1e6) if dev == "cuda" else 0.0

        est = U_map                                  # stable primal estimate
        converged = bool(np.isfinite(est) and est > 0)
        if W2_ref is not None and W2_ref > 0:
            converged = converged and (0.5 <= est / W2_ref <= 6.0)

        # --- Lemma 1 (neural feasibility margin): turn the UNCERTIFIED neural
        # potentials into a CERTIFIED lower bound on the discrete OT*. ---
        cert = _certify_neural(f, g, Xt, Yt, dev)

        out = dict(est=float(est), est_dual=float(est_dual), U_map=float(U_map),
                   time=float(dt), mem_mb=float(mem_mb), converged=converged,
                   iters=int(iters), device=dev)
        out.update(cert)
        return out
    except Exception as e:
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
        return dict(error=str(e)[:120], converged=False)
