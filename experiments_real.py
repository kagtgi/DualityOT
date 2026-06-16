"""experiments_real.py -- run the duality-gap study on real public data.

Produces results_real.json with, per dataset:
  frontier (error vs cost), composed certificate, dimension sweep,
plus a large-N GPU scaling run and a MNIST->USPS gap-vs-accuracy study that
shows the certified gap predicts a real downstream metric.
"""
import time
import numpy as np
import ot
import otkit
import datasets as D


def _label_transfer_acc(P, ys, yt):
    """OT label propagation: target label = class with most transported mass."""
    classes = np.unique(ys)
    onehot = np.zeros((len(ys), len(classes)))
    for k, c in enumerate(classes):
        onehot[ys == c, k] = 1.0
    score = onehot.T @ P                      # (C, n_t)
    pred = classes[np.argmax(score, axis=0)]
    return float(np.mean(pred == yt))


def da_gap_vs_accuracy(Xs, ys, Xt, yt, N=1500, seed=0, use_gpu=True,
                       eps_list=(0.5, 0.2, 0.1, 0.05, 0.02, 0.01), log=print):
    """For MNIST->USPS: sweep entropic eps, record the certified gap AND the
    downstream label-transfer accuracy of the same plan."""
    g = np.random.default_rng(seed)
    si = g.choice(len(Xs), min(N, len(Xs)), replace=False)
    ti = g.choice(len(Xt), min(N, len(Xt)), replace=False)
    Xs, ys = Xs[si], ys[si]; Xt, yt = Xt[ti], yt[ti]
    ns, nt = len(Xs), len(Xt)
    a = np.ones(ns) / ns; b = np.ones(nt) / nt
    M = ot.dist(Xs, Xt, metric="sqeuclidean")
    OTstar = otkit.exact_ot(M, a, b)
    Pe = ot.emd(a, b, M)
    rows = [dict(method="exact", eps=None, gap=0.0, relgap=0.0,
                 acc=_label_transfer_acc(Pe, ys, yt))]
    for er in eps_list:
        # log-stabilized solve -> rigorous certified gap + the plan we transfer with
        est, U, L, _, Ps = otkit.sinkhorn_gap(M, a, b, er, use_gpu=use_gpu,
                                              numItermax=50000, stopThr=1e-9,
                                              return_plan=True)
        acc = _label_transfer_acc(Ps, ys, yt)
        rows.append(dict(method="sinkhorn", eps=er, gap=U - L, relgap=(U - L) / OTstar, acc=acc))
        log(f"   eps={er:<5} relgap={(U-L)/OTstar:.3f} transfer-acc={acc:.3f}")
    return dict(OTstar=OTstar, ns=ns, nt=nt, rows=rows)


def run_real(log=print, fast=False, use_gpu=True):
    out = {"gpu": otkit.gpu_info(), "datasets": {}}
    Nfr = 600 if fast else 1500
    dims = (2, 5, 10, 20) if fast else (2, 5, 10, 20, 50)

    specs = [
        ("single_cell", lambda: D.load_singlecell(dim=50, log=log), True),
        ("mnist_usps",  lambda: D.load_mnist_usps(dim=50, log=log), True),
        ("color",       lambda: D.load_color(n_pixels=Nfr + 300, log=log), False),
    ]
    for name, loader, do_dim in specs:
        log(f"\n=== REAL DATASET: {name} ===")
        try:
            d = loader()
        except Exception as e:
            log(f"  [skip {name}] loader failed: {e}"); continue
        rec = {"meta": d["meta"], "dim": d.get("dim")}
        t0 = time.perf_counter()
        rec["frontier"] = otkit.run_frontier(d["X"], d["Y"], N=Nfr, use_gpu=use_gpu)
        log(f"  frontier OT*={rec['frontier']['OTstar']:.3f} ({time.perf_counter()-t0:.1f}s)")
        rec["composed"] = otkit.composed_certificate(d["X"], d["Y"], N=Nfr)
        log(f"  composed certificate relgap={rec['composed']['relgap']:.3f} "
            f"(deficit={rec['composed']['dual_deficit']:.3f}) t={rec['composed']['time']:.3f}s")
        if do_dim:
            rec["dimension"] = otkit.dimension_sweep(d["X"], d["Y"], dims=dims)
            log("  dimension sweep: " + ", ".join(
                f"d{r['d']}:{r['sliced_rel_gap']:.2f}" for r in rec["dimension"]["rows"]))
        if name == "color":  # keep the image arrays for the transfer visual
            rec["color_viz"] = True
            out["_color_imgs"] = dict(src_img=d["src_img"], tgt_img=d["tgt_img"],
                                      src_lab_full=d["src_lab_full"],
                                      src_sample=d["src_sample"], tgt_sample=d["tgt_sample"])
        out["datasets"][name] = rec

    # ---- large-N GPU scaling on real MNIST (split into two halves) ----
    log("\n=== REAL SCALING (GPU Sinkhorn on MNIST halves) ===")
    try:
        Xm, ym = D._torch_images("mnist", n_max=40000, seed=3)
        Z = D._pca(Xm, 50, 0)
        half = len(Z) // 2
        Ns = (500, 1000, 2000, 4000) if fast else (500, 1000, 2000, 4000, 8000, 16000, 32000)
        out["scaling"] = otkit.scaling(Z[:half], Z[half:], Ns=Ns, lp_max_n=4000, use_gpu=use_gpu)
        out["scaling_meta"] = "MNIST halves (PCA-50), real images"
        log("  scaling done: Sinkhorn n=" + ",".join(str(r["n"]) for r in out["scaling"]["sinkhorn"]))
    except Exception as e:
        log(f"  [scaling fallback] {e}; using digits")
        from sklearn.datasets import load_digits
        dd = load_digits(); Z = D._pca(dd.data.astype(float), 50, 0)
        out["scaling"] = otkit.scaling(Z[:800], Z[800:1600],
                                       Ns=(200, 400, 800), lp_max_n=800, use_gpu=use_gpu)
        out["scaling_meta"] = "sklearn digits (fallback)"

    # ---- gap predicts downstream accuracy (MNIST -> USPS) ----
    log("\n=== REAL DOWNSTREAM: gap vs label-transfer accuracy (MNIST->USPS) ===")
    try:
        da = D.load_mnist_da_labeled(dim=50, log=log)
        out["da_accuracy"] = da_gap_vs_accuracy(da["Xs"], da["ys"], da["Xt"], da["yt"],
                                                N=Nfr, use_gpu=use_gpu, log=log)
        out["da_accuracy"]["meta"] = da["meta"]
    except Exception as e:
        log(f"  [da skip] {e}")

    return out
