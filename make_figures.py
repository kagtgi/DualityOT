"""make_figures.py -- regenerate every figure (Google/Material palette + Roboto).

generate_all(synth, real, outdir, font_dir) writes, as PDF + PNG:
  paper figures   : fig1_teaser, fig2_frontier, fig3_diagnostics, fig4_hybrid
  real-data figures: real_frontier, real_dimension, real_gap_accuracy, color_transfer
`synth` and `real` are the in-memory result dicts (real may carry numpy images).
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm

# ---------- palette ----------
C_EXACT = "#202124"; C_SINK = "#4285F4"; C_LOWR = "#34A853"
C_SLICE = "#EA4335"; C_MINI = "#A142F4"
C_GAP = "#F9AB00"; C_GAPTX = "#B06000"
GREY2 = "#E8EAED"; GREY5 = "#9AA0A6"; GREY7 = "#5F6368"; GREY9 = "#202124"
SURF_AMBER = "#FEF7E0"
STYLE = {"exact": (C_EXACT, "*", True), "entropic": (C_SINK, "o", True),
         "lowrank": (C_LOWR, "s", False), "sliced": (C_SLICE, "^", False),
         "minibatch": (C_MINI, "D", False)}
LABEL = {"exact": "Exact LP", "entropic": "Sinkhorn", "lowrank": "Low-rank",
         "sliced": "Sliced", "minibatch": "Minibatch"}
FP_REG = FP_MED = None


def _setup(font_dir):
    global FP_REG, FP_MED
    for f in glob.glob(os.path.join(font_dir, "Roboto-*.ttf")):
        fm.fontManager.addfont(f)
    reg = os.path.join(font_dir, "Roboto-Regular.ttf")
    med = os.path.join(font_dir, "Roboto-Medium.ttf")
    FP_REG = fm.FontProperties(fname=reg) if os.path.exists(reg) else fm.FontProperties()
    FP_MED = fm.FontProperties(fname=med) if os.path.exists(med) else FP_REG
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "Roboto" if os.path.exists(reg) else "DejaVu Sans",
        "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8,
        "font.size": 9, "legend.fontsize": 7.4, "xtick.labelsize": 8, "ytick.labelsize": 8})


def style_ax(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GREY5); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=GREY7, length=3, width=0.7)
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color(GREY9)
    ax.grid(True, color=GREY2, lw=0.7); ax.set_axisbelow(True)


def _leg(ax, **kw):
    leg = ax.legend(frameon=True, facecolor="white", edgecolor="#DADCE0",
                    framealpha=0.95, borderpad=0.45, **kw)
    for t in leg.get_texts(): t.set_fontproperties(FP_REG); t.set_color(GREY9)
    return leg


def mk(ax, x, y, fam, **kw):
    c, m, filled = STYLE[fam]
    return ax.plot(x, y, marker=m, color=c, ls=kw.pop("ls", "-"),
                   mfc=(c if filled else "white"), mec=c, mew=1.3,
                   ms=kw.pop("ms", 5.4), lw=kw.pop("lw", 1.3), **kw)


def _save(fig, outdir, name):
    import matplotlib.pyplot as plt
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _frontier_panel(ax, rows, errkey, order=("exact", "entropic", "lowrank", "minibatch", "sliced"),
                    hybrid=None):
    fams = {}
    for r in rows: fams.setdefault(r["family"], []).append(r)
    for fam in order:
        if fam not in fams: continue
        rs = fams[fam]
        xs = np.array([max(r["mem"], 1e-3) for r in rs])
        ys = np.array([max(r[errkey], 5e-4) for r in rs])
        o = np.argsort(xs)
        mk(ax, xs[o], ys[o], fam, label=LABEL[fam])
    if hybrid is not None:
        ax.plot([256 * 256 * 8 / 1e6], [hybrid["relgap"]], "*", ms=12, mfc=C_GAP,
                mec=C_GAPTX, mew=1.0, zorder=6, label="hybrid cert.")
    ax.set_xscale("log"); ax.set_yscale("log")


# =================================================== PAPER FIGURES (synthetic)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def fig1_teaser(outdir):
    # Compact single-column version: tighter coordinate range, slightly smaller
    # fonts. The saved PDF is designed to be included as a single-column \figure
    # in the AAAI template (width ~3.33 in).
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.set_xlim(0.2, 10.5); ax.set_ylim(0.55, 3.05); ax.axis("off")
    y0 = 1.62; L, S, U = 2.6, 5.35, 7.75
    ax.add_patch(FancyBboxPatch((L, y0 - 0.2), U - L, 0.4,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=SURF_AMBER, ec="none", zorder=0))
    ax.annotate("", xy=(9.95, y0), xytext=(0.4, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREY7, lw=1.0))
    ax.text(10.0, y0, "cost", va="center", ha="left", fontsize=7, color=GREY7,
            fontproperties=FP_REG)
    for x, m, c, lab, sub in [
        (L, "^", C_SLICE,  "$L$",               "dual lower bound"),
        (S, "*", C_EXACT,  r"$\mathrm{OT}^\star$", "true optimum"),
        (U, "s", C_LOWR,   "$U$",               "primal upper bound"),
    ]:
        ax.plot([x], [y0], m, ms=11 if m == "*" else 8,
                mfc=(c if m == "*" else "white"), mec=c, mew=1.5, zorder=6)
        ax.text(x, y0 + 0.30, lab, ha="center", va="bottom",
                fontsize=10.5, color=c)
        ax.text(x, y0 + 0.72, sub, ha="center", va="bottom",
                fontsize=6.5, color=GREY7, fontproperties=FP_REG)
    yb = y0 - 0.38
    ax.plot([L, U], [yb, yb], color=C_GAP, lw=1.3)
    for x in (L, U): ax.plot([x, x], [yb, yb + 0.09], color=C_GAP, lw=1.3)
    ax.annotate(r"duality gap  $G = U - L$",
                xy=((L + U) / 2, yb), xytext=((L + U) / 2, yb - 0.38),
                ha="center", va="top", fontsize=9.5, color=C_GAPTX,
                fontproperties=FP_MED,
                arrowprops=dict(arrowstyle="-", color=C_GAP, lw=0.7))
    ax.add_patch(FancyArrowPatch(
        (L - 0.85, y0 + 0.02), (L + 0.45, y0 + 0.02),
        arrowstyle="-|>", mutation_scale=11, lw=1.2, color=C_SLICE))
    ax.text(L - 0.9, y0 - 0.24, "relaxation\nraises $L$",
            ha="center", va="top", fontsize=6.8, color=C_SLICE,
            fontproperties=FP_MED)
    ax.add_patch(FancyArrowPatch(
        (U + 0.85, y0 + 0.02), (U - 0.45, y0 + 0.02),
        arrowstyle="-|>", mutation_scale=11, lw=1.2, color=C_LOWR))
    ax.text(U + 0.9, y0 - 0.24, "restriction\nlowers $U$",
            ha="center", va="top", fontsize=6.8, color=C_LOWR,
            fontproperties=FP_MED)
    ax.text(5.35, 2.97,
            "Every approximate-OT family controls one side of the Kantorovich duality gap",
            ha="center", va="top", fontsize=8.5, color=GREY9,
            fontproperties=FP_MED)
    fig.tight_layout(pad=0.25); _save(fig, outdir, "fig1_teaser")


def fig2_frontier(synth, outdir):
    rows = synth["frontier"]["rows"]; fams = {}
    for r in rows: fams.setdefault(r["family"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, xk, xl in [(axes[0], "mem", "peak memory (MB)"), (axes[1], "time", "wall-clock time (s)")]:
        for fam in ["exact", "entropic", "lowrank", "minibatch", "sliced"]:
            rs = fams[fam]; xs = np.array([max(r[xk], 1e-3) for r in rs]); ys = np.array([max(r["rel_err_disc"], 5e-4) for r in rs]); o = np.argsort(xs)
            mk(ax, xs[o], ys[o], fam, label=(LABEL[fam] if xk == "mem" else None))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl, color=GREY9, fontproperties=FP_REG); style_ax(ax)
    axes[0].set_ylabel(r"relative error vs.\ discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
    _leg(axes[0], loc="lower left", handlelength=2.0); fig.tight_layout(pad=0.4); _save(fig, outdir, "fig2_frontier")


def fig3_diagnostics(synth, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    gg = synth["gap_geometry"]["rows"]; gaps = np.array([r["gap"] for r in gg]); dist = np.array([r["geo_dist"] for r in gg]); eps = np.array([r["eps_rel"] for r in gg])
    ax = axes[0]; ax.plot(gaps, dist, "-", color=C_SINK, lw=1.3, zorder=2)
    ax.scatter(gaps, dist, c=np.log10(eps), cmap="Blues_r", s=34, edgecolor=C_SINK, linewidth=0.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"certified gap $G$", color=C_GAPTX, fontproperties=FP_MED); ax.set_ylabel("geodesic distortion", color=GREY9, fontproperties=FP_REG)
    ax.set_title("(a) gap tracks geometry", color=GREY9, fontproperties=FP_MED, fontsize=8.7); style_ax(ax)
    dd = synth["dimension"]["rows"]; ds = np.array([r["d"] for r in dd]); sg = np.array([r["sliced_rel_gap"] for r in dd]); st = np.array([r["stat_rel_gap"] for r in dd])
    ax = axes[1]; dgrid = np.array([1, 2, 4, 8, 16, 32, 64, 128.]); ax.plot(dgrid, 1 - 1 / dgrid, "--", color=GREY5, lw=1.1, label=r"$1-1/d$ (Prop. 2)")
    mk(ax, ds, sg, "sliced", label="sliced gap", ms=5.0); ax.plot(ds, st, "o:", color=GREY7, mfc="white", mec=GREY7, mew=1.2, ms=4.2, lw=1.1, label="stat. gap")
    ax.set_xscale("log", base=2); ax.set_xlabel(r"dimension $d$", color=GREY9, fontproperties=FP_REG); ax.set_ylabel("relative gap", color=GREY9, fontproperties=FP_REG); ax.set_ylim(-0.03, 1.05)
    ax.set_title("(b) dimension widens the gap", color=GREY9, fontproperties=FP_MED, fontsize=8.7); _leg(ax, loc="lower right", handlelength=1.7, fontsize=6.5); style_ax(ax)
    sc = synth["scaling"]; ax = axes[2]
    fmap = {"exact": "exact", "sinkhorn": "entropic", "lowrank": "lowrank",
            "sliced": "sliced", "minibatch": "minibatch"}
    lmap = {"exact": "exact LP", "sinkhorn": "Sinkhorn", "lowrank": "low-rank",
            "sliced": "sliced", "minibatch": "minibatch"}
    for k in ["exact", "sinkhorn", "lowrank", "minibatch", "sliced"]:
        v = sc.get(k, [])
        if not v: continue
        ns = np.array([r["n"] for r in v]); ts = np.array([r["time"] for r in v])
        mk(ax, ns, ts, fmap[k], label=lmap[k], ms=4.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"sample size $n$", color=GREY9, fontproperties=FP_REG); ax.set_ylabel("time (s)", color=GREY9, fontproperties=FP_REG)
    ax.set_title("(c) cost of certifiability", color=GREY9, fontproperties=FP_MED, fontsize=8.7); _leg(ax, loc="upper left", handlelength=1.7, fontsize=6.5); style_ax(ax)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "fig3_diagnostics")


def fig4_hybrid(synth, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    rows = synth["hybrid_dim"]["rows"]; d = np.array([r["d"] for r in rows], float)
    ax = axes[0]; ax.plot(d, 1 - 1 / d, ":", color=GREY5, lw=1.1, label=r"$1-1/d$ (Prop. 2)")
    ax.plot(d, [r["relgap_hyb"] for r in rows], "D-", color=C_GAP, mfc=C_GAP, mec=C_GAP, ms=5.2, lw=1.6, label="hybrid $U\\!-\\!L$ (near-lin.)")
    ax.plot(d, [r["dual_deficit"] for r in rows], "^--", color=C_SLICE, mfc="white", mec=C_SLICE, mew=1.2, ms=4.6, lw=1.1, label="dual deficit (sliced)")
    ax.plot(d, [r["relgap_ent"] for r in rows], "o-", color=C_SINK, mfc=C_SINK, mec=C_SINK, ms=4.6, lw=1.3, label="entropic $U\\!-\\!L$ ($O(n^2)$)")
    ax.set_xscale("log", base=2); ax.set_xlabel(r"dimension $d$", color=GREY9, fontproperties=FP_REG); ax.set_ylabel("relative certified gap", color=C_GAPTX, fontproperties=FP_MED); ax.set_ylim(-0.04, 1.12)
    ax.set_title("(a) cheap certificates by composition", color=GREY9, fontproperties=FP_MED, fontsize=8.5); _leg(ax, loc="lower right", handlelength=1.7, fontsize=6.2); style_ax(ax)
    ng = synth["nongauss"]; ax = axes[1]; _frontier_panel(ax, ng["rows"], "rel_err", hybrid=ng["hybrid"])
    ax.set_xlabel("peak memory (MB)", color=GREY9, fontproperties=FP_REG); ax.set_ylabel(r"rel. error vs.\ discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
    ax.set_title("(b) non-Gaussian frontier (two-moons)", color=GREY9, fontproperties=FP_MED, fontsize=8.5); _leg(ax, loc="lower left", handlelength=1.6, fontsize=6.2); style_ax(ax)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "fig4_hybrid")


# =================================================== REAL-DATA FIGURES
PRETTY = {"single_cell": "single-cell (PBMC)", "mnist_usps": "MNIST vs USPS", "color": "color transfer"}


def real_frontier(real, outdir):
    names = [n for n in ["single_cell", "mnist_usps", "color"] if n in real["datasets"]]
    fig, axes = plt.subplots(1, len(names), figsize=(2.55 * len(names), 2.7), squeeze=False)
    for ax, name in zip(axes[0], names):
        rec = real["datasets"][name]
        _frontier_panel(ax, rec["frontier"]["rows"], "rel_err", hybrid=rec.get("composed"))
        ax.set_xlabel("peak memory (MB)", color=GREY9, fontproperties=FP_REG)
        ax.set_title(PRETTY[name], color=GREY9, fontproperties=FP_MED, fontsize=8.7); style_ax(ax)
    axes[0][0].set_ylabel(r"rel. error vs.\ discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
    _leg(axes[0][0], loc="lower left", handlelength=1.6, fontsize=6.3)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "real_frontier")


def real_dimension(real, outdir):
    names = [n for n in ["single_cell", "mnist_usps"] if n in real["datasets"] and "dimension" in real["datasets"][n]]
    if not names: return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    colors = {"single_cell": C_LOWR, "mnist_usps": C_SINK}
    marks = {"single_cell": "s", "mnist_usps": "o"}
    dmax = 1
    for name in names:
        rows = real["datasets"][name]["dimension"]["rows"]
        d = np.array([r["d"] for r in rows], float); sg = np.array([r["sliced_rel_gap"] for r in rows])
        dmax = max(dmax, d.max())
        ax.plot(d, sg, marker=marks[name], color=colors[name], mfc=colors[name], mec=colors[name], ms=5, lw=1.4, label=PRETTY[name])
    dg = np.logspace(0, np.log2(dmax), 50, base=2)
    ax.plot(dg, 1 - 1 / dg, ":", color=GREY5, lw=1.2, label=r"$1-1/d$ (Prop. 2)")
    ax.set_xscale("log", base=2); ax.set_xlabel(r"effective dimension $d$ (PCA)", color=GREY9, fontproperties=FP_REG)
    ax.set_ylabel("sliced relative gap", color=GREY9, fontproperties=FP_REG); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Dimensional deficit on real data", color=GREY9, fontproperties=FP_MED, fontsize=9)
    _leg(ax, loc="lower right", handlelength=1.7, fontsize=6.6); style_ax(ax)
    fig.tight_layout(pad=0.4); _save(fig, outdir, "real_dimension")


def real_gap_accuracy(real, outdir):
    has_da = "da_accuracy" in real; has_sc = "scaling" in real and len(real["scaling"].get("sinkhorn", [])) > 1
    ncol = has_da + has_sc
    if ncol == 0: return
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 2.7), squeeze=False); k = 0
    if has_da:
        ax = axes[0][k]; k += 1
        rows = [r for r in real["da_accuracy"]["rows"] if r["method"] == "sinkhorn"]
        g = np.array([r["relgap"] for r in rows]); acc = np.array([r["acc"] for r in rows])
        sc = ax.scatter(g, acc, c=np.log10([r["eps"] for r in rows]), cmap="Blues_r", s=46, edgecolor=C_SINK, linewidth=1.0, zorder=3)
        o = np.argsort(g); ax.plot(g[o], acc[o], "-", color=C_SINK, lw=1.2, zorder=2)
        ex = [r for r in real["da_accuracy"]["rows"] if r["method"] == "exact"]
        if ex: ax.axhline(ex[0]["acc"], color=C_EXACT, ls="--", lw=1.0, label="exact-OT transfer")
        ax.set_xlabel(r"certified gap $G/\mathrm{OT}^\star$", color=C_GAPTX, fontproperties=FP_MED)
        ax.set_ylabel("label-transfer accuracy", color=GREY9, fontproperties=FP_REG)
        ax.set_title("(a) gap predicts transfer (MNIST$\\to$USPS)", color=GREY9, fontproperties=FP_MED, fontsize=8.3)
        _leg(ax, loc="lower left", handlelength=1.6, fontsize=6.4); style_ax(ax)
    if has_sc:
        ax = axes[0][k]
        sc = real["scaling"]
        fmap = {"exact": "exact", "sinkhorn": "entropic", "lowrank": "lowrank",
                "sliced": "sliced", "minibatch": "minibatch"}
        lmap = {"exact": "exact LP", "sinkhorn": "Sinkhorn (GPU)", "lowrank": "low-rank",
                "sliced": "sliced", "minibatch": "minibatch"}
        for key in ["exact", "sinkhorn", "lowrank", "sliced"]:
            v = sc.get(key, [])
            if len(v) < 1: continue
            ns = np.array([r["n"] for r in v]); ts = np.array([r["time"] for r in v]); mk(ax, ns, ts, fmap[key], label=lmap[key], ms=4.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"sample size $n$", color=GREY9, fontproperties=FP_REG); ax.set_ylabel("time (s)", color=GREY9, fontproperties=FP_REG)
        ax.set_title("(b) cost of certifiability at scale", color=GREY9, fontproperties=FP_MED, fontsize=8.3)
        _leg(ax, loc="upper left", handlelength=1.7, fontsize=6.4); style_ax(ax)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "real_gap_accuracy")


def color_transfer(real, outdir):
    imgs = real.get("_color_imgs")
    if imgs is None: return
    import ot
    from skimage import color
    src = imgs["src_sample"]; tgt = imgs["tgt_sample"]; n = len(src)
    a = np.ones(n) / n; b = np.ones(n) / n
    M = ot.dist(src, tgt, metric="sqeuclidean")
    P = ot.emd(a, b, M)
    mapped = (P @ tgt) / np.maximum(P.sum(1, keepdims=True), 1e-12)  # barycentric map of samples
    full = imgs["src_lab_full"]
    # nearest source-sample for every full pixel, then take its transported color
    from scipy.spatial import cKDTree
    tree = cKDTree(src); _, idx = tree.query(full, k=1)
    out_lab = mapped[idx].reshape(imgs["src_img"].shape)
    out_rgb = np.clip(color.lab2rgb(out_lab), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, im, ttl in [(axes[0], imgs["src_img"], "source"), (axes[1], imgs["tgt_img"], "target palette"), (axes[2], out_rgb, "source recolored via OT")]:
        ax.imshow(im); ax.set_title(ttl, color=GREY9, fontproperties=FP_MED, fontsize=9); ax.axis("off")
    fig.tight_layout(pad=0.3); _save(fig, outdir, "color_transfer")


# =================================================== driver
def generate_all(synth, real, outdir, font_dir="."):
    _setup(font_dir); os.makedirs(outdir, exist_ok=True); made = []
    try: fig1_teaser(outdir); made.append("fig1_teaser")
    except Exception as e: print("fig1 failed:", e)
    for fn, nm in [(fig2_frontier, "fig2_frontier"), (fig3_diagnostics, "fig3_diagnostics"), (fig4_hybrid, "fig4_hybrid")]:
        try: fn(synth, outdir); made.append(nm)
        except Exception as e: print(nm, "failed:", e)
    for fn, nm in [(real_frontier, "real_frontier"), (real_dimension, "real_dimension"), (real_gap_accuracy, "real_gap_accuracy"), (color_transfer, "color_transfer")]:
        try: fn(real, outdir); made.append(nm)
        except Exception as e: print(nm, "failed:", e)
    return made
