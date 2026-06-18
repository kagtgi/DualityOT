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
    outer (f) update."""
    y = y.requires_grad_(True)
    out = net(y).sum()
    g = torch.autograd.grad(out, y, create_graph=True)[0]
    return g


def neural_w2(X, Y, W2_ref=None, width=64, depth=2, iters=1500, k_inner=8,
              lr=1e-3, use_gpu=True, seed=0):
    """Train the two-ICNN maximin and return an UNCERTIFIED W2^2 estimate.

    Returns dict(est, U_map, time, mem_mb, converged, iters) or None if torch is
    unavailable / training fails. `est` is the maximin W2^2 estimate; `U_map` is
    the Brenier-map transport cost (a second uncertified estimate). `converged`
    is a sanity flag (finite and within a 5x band of W2_ref when provided).
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

        with torch.no_grad():
            corr_val = float(V().item())
        # W2^2 = E|x|^2 + E|y|^2 - 2 * max-correlation
        ex2 = float((Xt ** 2).sum(1).mean().item())
        ey2 = float((Yt ** 2).sum(1).mean().item())
        est = ex2 + ey2 - 2.0 * corr_val

        # Brenier map T = grad f : mu -> nu, transport cost E|x - grad f(x)|^2
        Xg = Xt.detach().requires_grad_(True)
        Tx = torch.autograd.grad(f(Xg).sum(), Xg)[0]
        U_map = float(((Xt - Tx) ** 2).sum(1).mean().item())

        dt = time.perf_counter() - t0
        mem_mb = (float(torch.cuda.max_memory_allocated()) / 1e6) if dev == "cuda" else 0.0

        converged = np.isfinite(est) and est > 0
        if W2_ref is not None and W2_ref > 0:
            converged = converged and (0.2 <= est / W2_ref <= 5.0)
        return dict(est=float(est), U_map=float(U_map), time=float(dt),
                    mem_mb=float(mem_mb), converged=bool(converged), iters=int(iters),
                    device=dev)
    except Exception as e:
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
        return dict(error=str(e)[:120], converged=False)
