"""Sanity tests for the certified Kantorovich bounds and the closed-form
ground truth. Requires POT (`pip install POT`); runs anywhere POT is installed
(e.g. the Colab environment). Usage:

    python tests/test_bounds.py        # plain runner, exits non-zero on failure
    pytest tests/                      # if pytest is available
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import otkit as K


def _problem(n=80, d=4, seed=0):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, d)); Y = g.standard_normal((n, d)) + 1.5
    a = np.ones(n) / n; M = K.cost_matrix(X, Y)
    return X, Y, a, M


def test_rounding_is_feasible():
    g = np.random.default_rng(1); n, m = 50, 40
    a = g.random(n); a /= a.sum(); b = g.random(m); b /= b.sum()
    P = K.round_to_marginals(g.random((n, m)), a, b)
    assert (P >= -1e-12).all()
    assert np.allclose(P.sum(1), a) and np.allclose(P.sum(0), b)


def test_sinkhorn_brackets_optimum():
    _, _, a, M = _problem()
    OT = K.exact_ot(M, a, a)
    for er in (0.2, 0.05, 0.01):
        est, U, L, _ = K.sinkhorn_gap(M, a, a, er, use_gpu=False)
        assert L <= OT + 1e-6, (er, L, OT)
        assert U >= OT - 1e-6, (er, U, OT)
        assert U - L >= -1e-9


def test_sliced_is_lower_bound():
    X, Y, a, M = _problem()
    OT = K.exact_ot(M, a, a)
    assert K.sliced_lb(X, Y, n_proj=200) <= OT + 1e-6


def test_lowrank_is_upper_bound():
    X, Y, a, M = _problem()
    OT = K.exact_ot(M, a, a)
    assert K.lowrank_ub(X, Y, a, a, 10, M) >= OT - 1e-6


def test_bures_shift_gaussian():
    d = 5; m0 = np.zeros(d); m1 = np.ones(d) * 2; I = np.eye(d)
    assert abs(K.bures_w2sq(m0, I, m1, I) - np.sum((m0 - m1) ** 2)) < 1e-9


def test_bures_geodesic_endpoints():
    C0 = np.array([[2.0, 0.0], [0.0, 1.0]]); C1 = np.array([[1.0, 0.0], [0.0, 3.0]])
    assert np.allclose(K.bw_geodesic_cov(C0, C1, 0.0), C0, atol=1e-9)
    assert np.allclose(K.bw_geodesic_cov(C0, C1, 1.0), C1, atol=1e-9)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in tests:
        try:
            fn(); print("ok   ", fn.__name__)
        except Exception:
            bad += 1; print("FAIL ", fn.__name__); traceback.print_exc()
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
