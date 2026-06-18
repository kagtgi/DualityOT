"""datasets.py -- robust loaders for the real-data OT experiments.

Each loader returns a dict with X, Y (numpy float arrays) and a `meta` string
naming the actual source used (loaders fall back gracefully so the notebook
never crashes on a flaky download). Labeled variants also return y_src, y_tgt.

Public datasets used (all standard, all free):
  * PBMC scRNA-seq  (single-cell genomics; Waddington-OT's domain)
  * MNIST / USPS    (the classic OT domain-adaptation benchmark)
  * skimage photos  (textbook color-transfer OT, 3-D Lab space)
"""
import numpy as np


def _pca(Z, d, seed=0):
    from sklearn.decomposition import PCA
    d = int(min(d, Z.shape[1], Z.shape[0]))
    return PCA(n_components=d, random_state=seed).fit_transform(Z).astype(np.float64)


# --------------------------------------------------------- single cell (PBMC)
def _load_pbmc_anndata():
    """Load a processed PBMC AnnData (real cell-type labels). Raises on failure
    so callers can fall back."""
    import scanpy as sc
    try:
        ad = sc.datasets.pbmc3k_processed()
        return ad, "louvain", "scanpy pbmc3k_processed"
    except Exception:
        ad = sc.datasets.pbmc68k_reduced()
        return ad, "bulk_labels", "scanpy pbmc68k_reduced (bundled)"


def _diffmap_embedding(ad, n_comps=15, seed=0):
    """Nonlinear diffusion-map coordinates (drop the trivial steady-state
    component). Returns (n, n_comps-1) or None on failure."""
    try:
        import scanpy as sc
        if "X_diffmap" not in ad.obsm:
            if "neighbors" not in ad.uns and "distances" not in ad.obsp:
                rep = "X_pca" if "X_pca" in ad.obsm else None
                sc.pp.neighbors(ad, n_neighbors=15, use_rep=rep)
            sc.tl.diffmap(ad, n_comps=n_comps)
        D = np.asarray(ad.obsm["X_diffmap"])
        return D[:, 1:].astype(np.float64)   # drop component 0 (constant)
    except Exception:
        return None


def _umap2d(ad, seed=0):
    """2-D UMAP coords for the transport-quiver figure; fall back to PCA-2D."""
    try:
        if "X_umap" in ad.obsm:
            return np.asarray(ad.obsm["X_umap"])[:, :2].astype(np.float64)
        import scanpy as sc
        if "neighbors" not in ad.uns and "distances" not in ad.obsp:
            sc.pp.neighbors(ad, n_neighbors=15,
                            use_rep="X_pca" if "X_pca" in ad.obsm else None)
        sc.tl.umap(ad)
        return np.asarray(ad.obsm["X_umap"])[:, :2].astype(np.float64)
    except Exception:
        if "X_pca" in ad.obsm:
            return np.asarray(ad.obsm["X_pca"])[:, :2].astype(np.float64)
        return None


def _pbmc_pca(ad, dim, seed=0):
    if "X_pca" in ad.obsm and ad.obsm["X_pca"].shape[1] >= 2:
        return np.asarray(ad.obsm["X_pca"])[:, :dim].astype(np.float64)
    Xd = ad.X
    Xd = np.asarray(Xd.todense()) if hasattr(Xd, "todense") else np.asarray(Xd)
    return _pca(Xd, dim, seed)


def _raw_gene_matrix(ad):
    """Dense expression matrix (cells x genes), used for the high-dimensional
    'native gene space' OT point."""
    Xd = ad.X
    Xd = np.asarray(Xd.todense()) if hasattr(Xd, "todense") else np.asarray(Xd)
    return Xd.astype(np.float64)


def load_singlecell(dim=50, seed=0, log=print):
    """OT between two annotated cell types in PCA space (back-compatible)."""
    rich = load_singlecell_rich(dim=dim, seed=seed, log=log)
    return dict(X=rich["X"], Y=rich["Y"], meta=rich["meta"], dim=rich["dim"])


def load_singlecell_rich(dim=50, n_comps_diff=15, seed=0, log=print):
    """Rich single-cell loader: returns the two most-populous cell types in
    BOTH a linear (PCA) and a nonlinear (diffusion-map) embedding, plus the raw
    gene-space matrices, 2-D UMAP coordinates (for the transport-quiver figure),
    and labels. Falls back to a synthetic Gaussian mixture so the run never
    crashes (the fallback sets diffmap=pca and umap=pca-2D)."""
    try:
        ad, key, src = _load_pbmc_anndata()
        labels = ad.obs[key].astype(str).values
        uniq, cnt = np.unique(labels, return_counts=True)
        order = uniq[np.argsort(-cnt)]
        gA, gB = order[0], order[1]
        mA, mB = labels == gA, labels == gB

        Zpca = _pbmc_pca(ad, dim, seed)
        Zdiff = _diffmap_embedding(ad, n_comps=n_comps_diff, seed=seed)
        if Zdiff is None:
            Zdiff = Zpca
            diff_src = "pca (diffmap unavailable)"
        else:
            diff_src = "diffusion map"
        U2 = _umap2d(ad, seed=seed)
        Zraw = _raw_gene_matrix(ad)

        X = Zpca[mA]; Y = Zpca[mB]
        log(f"[single-cell] {src}: '{gA}' (n={mA.sum()}) vs '{gB}' (n={mB.sum()}), "
            f"pca-dim={Zpca.shape[1]}, diffmap-dim={Zdiff.shape[1]}, genes={Zraw.shape[1]}")
        return dict(
            X=X.astype(np.float64), Y=Y.astype(np.float64),
            X_diff=Zdiff[mA], Y_diff=Zdiff[mB],
            X_raw=Zraw[mA], Y_raw=Zraw[mB],
            umapA=(U2[mA] if U2 is not None else None),
            umapB=(U2[mB] if U2 is not None else None),
            gA=str(gA), gB=str(gB),
            meta=f"{src}: {gA} vs {gB} ({diff_src} + raw genes)",
            dim=Zpca.shape[1], n_genes=int(Zraw.shape[1]))
    except Exception as e:
        log(f"[single-cell] scanpy unavailable ({e}); synthetic Gaussian-mixture fallback")
        rng = np.random.default_rng(seed); d = dim
        def mix(centers):
            return np.vstack([rng.multivariate_normal(c, 0.6 * np.eye(d), 700) for c in centers])
        cA = [rng.standard_normal(d) * 2 for _ in range(3)]
        cB = [c + 1.5 for c in cA]
        X, Y = mix(cA), mix(cB)
        return dict(X=X, Y=Y, X_diff=X, Y_diff=Y, X_raw=X, Y_raw=Y,
                    umapA=X[:, :2], umapB=Y[:, :2], gA="A", gB="B",
                    meta="synthetic-fallback gaussian-mixture", dim=d, n_genes=d)


def load_singlecell_labeled(dim=50, k_types=4, shift_scale=0.6, seed=0, log=print):
    """Controlled-domain-shift single-cell DA for the gap-vs-accuracy study.

    Takes the top-`k_types` cell types, splits each into a source and a target
    half (so both domains carry every label), then applies a mild *controlled*
    batch effect (random near-identity linear map + offset) to the target. The
    certified gap of the source->target transport plan can then be related to
    cell-type label-transfer accuracy, mirroring the MNIST->USPS study in a
    biological setting. Falls back to a synthetic shift if scanpy is missing."""
    try:
        ad, key, src = _load_pbmc_anndata()
        labels = ad.obs[key].astype(str).values
        Z = _pbmc_pca(ad, dim, seed)
        uniq, cnt = np.unique(labels, return_counts=True)
        keep = uniq[np.argsort(-cnt)][:k_types]
        mask = np.isin(labels, keep)
        Z = Z[mask]; lab = labels[mask]
        # integer-code the labels
        code = {c: i for i, c in enumerate(keep)}
        y = np.array([code[c] for c in lab])
        rng = np.random.default_rng(seed)
        # split each class ~half into source/target
        si, ti = [], []
        for c in range(len(keep)):
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            h = len(idx) // 2; si += list(idx[:h]); ti += list(idx[h:])
        si = np.array(si); ti = np.array(ti)
        Xs, ys = Z[si], y[si]; Xt, yt = Z[ti], y[ti]
        # controlled batch effect on target: near-identity linear map + offset
        d = Z.shape[1]
        A = np.eye(d) + shift_scale * rng.standard_normal((d, d)) / np.sqrt(d)
        off = shift_scale * rng.standard_normal(d) * Z.std(0).mean()
        Xt = Xt @ A.T + off
        log(f"[single-cell DA] {src}: {k_types} cell types, controlled batch shift; "
            f"source n={len(Xs)}, target n={len(Xt)}, dim={d}")
        return dict(Xs=Xs.astype(np.float64), ys=ys, Xt=Xt.astype(np.float64), yt=yt,
                    meta=f"{src}: {k_types}-type PBMC, controlled batch shift", dim=d,
                    classes=[str(c) for c in keep])
    except Exception as e:
        log(f"[single-cell DA] unavailable ({e}); synthetic Gaussian-mixture DA")
        rng = np.random.default_rng(seed); d = dim; k = k_types
        cen = [rng.standard_normal(d) * 2 for _ in range(k)]
        def mk():
            X = np.vstack([rng.multivariate_normal(c, 0.5 * np.eye(d), 250) for c in cen])
            y = np.repeat(np.arange(k), 250); return X, y
        Xs, ys = mk(); Xt, yt = mk()
        A = np.eye(d) + 0.6 * rng.standard_normal((d, d)) / np.sqrt(d)
        Xt = Xt @ A.T + 0.6 * rng.standard_normal(d)
        return dict(Xs=Xs, ys=ys, Xt=Xt, yt=yt,
                    meta="synthetic-fallback PBMC-like DA", dim=d,
                    classes=[str(i) for i in range(k)])


# --------------------------------------------------------- MNIST / USPS images
def _torch_images(which, n_max=6000, seed=0):
    """Return (images Nx(HxW) in [0,1], labels) for 'mnist' or 'usps'."""
    import torch, torchvision
    from torchvision import transforms
    rs = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    if which == "mnist":
        ds = torchvision.datasets.MNIST(root="./_data", train=True, download=True, transform=rs)
    else:
        ds = torchvision.datasets.USPS(root="./_data", train=True, download=True, transform=rs)
    g = np.random.default_rng(seed)
    idx = g.choice(len(ds), min(n_max, len(ds)), replace=False)
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in idx]).astype(np.float64)
    y = np.array([int(ds[i][1]) for i in idx])
    return X, y


def load_mnist_usps(dim=50, seed=0, n_max=6000, log=print):
    """OT between MNIST and USPS digit-image distributions (domain shift)."""
    try:
        Xm, ym = _torch_images("mnist", n_max, seed)
        Xu, yu = _torch_images("usps", n_max, seed + 1)
        Z = _pca(np.vstack([Xm, Xu]), dim, seed)
        X, Y = Z[: len(Xm)], Z[len(Xm):]
        log(f"[mnist-usps] MNIST (n={len(X)}) vs USPS (n={len(Y)}), dim={Z.shape[1]}")
        return dict(X=X.astype(np.float64), Y=Y.astype(np.float64),
                    meta="MNIST vs USPS (16x16, PCA)", dim=Z.shape[1])
    except Exception as e:
        log(f"[mnist-usps] torchvision unavailable ({e}); falling back to sklearn digits classes")
        from sklearn.datasets import load_digits
        d = load_digits()
        Z = _pca(d.data.astype(np.float64), dim, seed)
        X = Z[d.target % 2 == 0]; Y = Z[d.target % 2 == 1]
        return dict(X=X, Y=Y, meta="sklearn digits even-vs-odd (PCA)", dim=Z.shape[1])


def load_mnist_da_labeled(dim=50, seed=0, n_max=4000, log=print):
    """Labeled MNIST(source) -> USPS(target) for the gap-vs-accuracy study."""
    try:
        Xm, ym = _torch_images("mnist", n_max, seed)
        Xu, yu = _torch_images("usps", n_max, seed + 1)
        Z = _pca(np.vstack([Xm, Xu]), dim, seed)
        Xs, Xt = Z[: len(Xm)], Z[len(Xm):]
        log(f"[mnist->usps DA] source MNIST n={len(Xs)}, target USPS n={len(Xt)}, dim={Z.shape[1]}")
        return dict(Xs=Xs.astype(np.float64), ys=ym, Xt=Xt.astype(np.float64), yt=yu,
                    meta="MNIST->USPS labeled DA", dim=Z.shape[1])
    except Exception as e:
        log(f"[mnist->usps DA] torchvision unavailable ({e}); using digits with a domain shift")
        from sklearn.datasets import load_digits
        from scipy.ndimage import rotate
        d = load_digits()
        imgs = d.images  # (n,8,8)
        rot = np.stack([rotate(im, 25, reshape=False, order=1) for im in imgs]).reshape(len(imgs), -1)
        Z = _pca(np.vstack([d.data, rot]).astype(np.float64), dim, seed)
        Xs, Xt = Z[: len(d.data)], Z[len(d.data):]
        return dict(Xs=Xs, ys=d.target, Xt=Xt, yt=d.target, meta="digits + 25deg-rotation DA", dim=Z.shape[1])


# --------------------------------------------------------- color transfer
def load_color(n_pixels=1800, seed=0, log=print):
    """OT between the color (CIE-Lab) distributions of two real photographs."""
    from skimage import data, color, transform
    src_img = transform.resize(data.astronaut(), (256, 256), anti_aliasing=True)
    tgt_img = transform.resize(data.coffee(), (256, 256), anti_aliasing=True)
    src_lab = color.rgb2lab(src_img).reshape(-1, 3)
    tgt_lab = color.rgb2lab(tgt_img).reshape(-1, 3)
    g = np.random.default_rng(seed)
    si = g.choice(src_lab.shape[0], n_pixels, replace=False)
    ti = g.choice(tgt_lab.shape[0], n_pixels, replace=False)
    log(f"[color] astronaut -> coffee, {n_pixels} Lab pixels each")
    return dict(X=src_lab[si].astype(np.float64), Y=tgt_lab[ti].astype(np.float64),
                meta="color transfer astronaut->coffee (Lab)", dim=3,
                src_img=src_img, tgt_img=tgt_img,
                src_lab_full=src_lab, src_sample=src_lab[si], tgt_sample=tgt_lab[ti])
