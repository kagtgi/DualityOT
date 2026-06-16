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
def load_singlecell(dim=50, seed=0, log=print):
    """OT between two annotated cell types in PCA space."""
    try:
        import scanpy as sc
        try:
            ad = sc.datasets.pbmc3k_processed()
            key = "louvain"
            src = "scanpy pbmc3k_processed"
        except Exception:
            ad = sc.datasets.pbmc68k_reduced()
            key = "bulk_labels"
            src = "scanpy pbmc68k_reduced (bundled)"
        labels = ad.obs[key].astype(str).values
        # pick the two most populous groups
        uniq, cnt = np.unique(labels, return_counts=True)
        order = uniq[np.argsort(-cnt)]
        gA, gB = order[0], order[1]
        if "X_pca" in ad.obsm and ad.obsm["X_pca"].shape[1] >= 2:
            Z = np.asarray(ad.obsm["X_pca"])[:, :dim]
        else:
            Xd = ad.X
            Xd = np.asarray(Xd.todense()) if hasattr(Xd, "todense") else np.asarray(Xd)
            Z = _pca(Xd, dim, seed)
        X = Z[labels == gA].astype(np.float64)
        Y = Z[labels == gB].astype(np.float64)
        log(f"[single-cell] {src}: '{gA}' (n={len(X)}) vs '{gB}' (n={len(Y)}), dim={Z.shape[1]}")
        return dict(X=X, Y=Y, meta=f"{src}: {gA} vs {gB}", dim=Z.shape[1])
    except Exception as e:
        log(f"[single-cell] scanpy unavailable ({e}); falling back to synthetic Gaussian mixtures")
        rng = np.random.default_rng(seed)
        d = dim
        def mix(centers):
            return np.vstack([rng.multivariate_normal(c, 0.6 * np.eye(d), 700) for c in centers])
        cA = [rng.standard_normal(d) * 2 for _ in range(3)]
        cB = [c + 1.5 for c in cA]
        return dict(X=mix(cA), Y=mix(cB), meta="synthetic-fallback gaussian-mixture", dim=d)


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
