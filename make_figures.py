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
C_NEURAL = "#FB8C00"
GREY2 = "#E8EAED"; GREY5 = "#9AA0A6"; GREY7 = "#5F6368"; GREY9 = "#202124"
SURF_AMBER = "#FEF7E0"
STYLE = {"exact": (C_EXACT, "*", True), "entropic": (C_SINK, "o", True),
         "lowrank": (C_LOWR, "s", False), "sliced": (C_SLICE, "^", False),
         "minibatch": (C_MINI, "D", False), "neural": (C_NEURAL, "P", False)}
LABEL = {"exact": "Exact LP", "entropic": "Sinkhorn", "lowrank": "Low-rank",
         "sliced": "Sliced", "minibatch": "Minibatch", "neural": "Neural (ICNN)"}
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


def _frontier_panel(ax, rows, errkey,
                    order=("exact", "entropic", "lowrank", "minibatch", "sliced", "neural"),
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
    # Single-column concept figure (AAAI ~3.33 in wide). A clean number-line:
    # OT* sits between a dual lower bound L and a primal upper bound U; relaxation
    # pushes L up, restriction pushes U down, and the amber span is the gap. The
    # mechanism arrows sit well above the L/U symbols so nothing overlaps.
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    ax.set_xlim(0, 10); ax.set_ylim(0.6, 8.7); ax.axis("off")
    axis_y = 3.9
    Lx, Sx, Ux = 2.45, 5.0, 7.55

    # title (two centered lines)
    ax.text(5.0, 8.5, "Every approximate-OT family controls one\n"
            "side of the Kantorovich duality gap",
            ha="center", va="top", fontsize=8.8, color=GREY9,
            fontproperties=FP_MED, linespacing=1.35)

    # soft amber gap band between L and U
    ax.add_patch(FancyBboxPatch((Lx, axis_y - 0.5), Ux - Lx, 1.0,
                 boxstyle="round,pad=0,rounding_size=0.14",
                 fc=SURF_AMBER, ec="none", zorder=0))

    # transport-cost axis
    ax.annotate("", xy=(9.35, axis_y), xytext=(0.65, axis_y),
                arrowprops=dict(arrowstyle="-|>", color=GREY7, lw=1.1))
    ax.text(9.5, axis_y, "transport\ncost", va="center", ha="left",
            fontsize=6.0, color=GREY7, fontproperties=FP_REG, linespacing=1.1)

    # mechanism push-arrows, placed high above the axis (clear of the L/U symbols)
    ar_y = axis_y + 2.05
    ax.add_patch(FancyArrowPatch((Lx - 0.7, ar_y), (Lx + 0.7, ar_y),
                 arrowstyle="-|>", mutation_scale=13, lw=1.8, color=C_SLICE))
    ax.text(Lx, ar_y + 0.32, "relaxation raises $L$", ha="center",
            va="bottom", fontsize=6.8, color=C_SLICE, fontproperties=FP_MED)
    ax.add_patch(FancyArrowPatch((Ux + 0.7, ar_y), (Ux - 0.7, ar_y),
                 arrowstyle="-|>", mutation_scale=13, lw=1.8, color=C_LOWR))
    ax.text(Ux, ar_y + 0.32, "restriction lowers $U$", ha="center",
            va="bottom", fontsize=6.8, color=C_LOWR, fontproperties=FP_MED)

    # markers + symbol labels (symbols sit just above the markers, well below arrows)
    for x, m, c, lab, sub, filled, ms in [
        (Lx, "^", C_SLICE, "$L$", "lower bound", False, 11),
        (Sx, "*", C_EXACT, r"$\mathrm{OT}^\star$", "true optimum", True, 16),
        (Ux, "s", C_LOWR, "$U$", "upper bound", False, 10),
    ]:
        ax.plot([x], [axis_y], m, ms=ms, mfc=(c if filled else "white"),
                mec=c, mew=1.8, zorder=6)
        ax.text(x, axis_y + 0.48, lab, ha="center", va="bottom",
                fontsize=12, color=c)
        ax.text(x, axis_y - 0.72, sub, ha="center", va="top",
                fontsize=6.2, color=GREY7, fontproperties=FP_REG)

    # gap bracket below
    yb = axis_y - 1.75
    ax.plot([Lx, Ux], [yb, yb], color=C_GAP, lw=1.8, solid_capstyle="round", zorder=4)
    for x in (Lx, Ux):
        ax.plot([x, x], [yb, yb + 0.2], color=C_GAP, lw=1.8, solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(Sx, yb), xytext=(Sx, axis_y - 0.95),
                arrowprops=dict(arrowstyle="-", color=C_GAP, lw=0.8, ls=(0, (2, 2))))
    ax.text(Sx, yb - 0.42, r"duality gap  $G = U - L$", ha="center", va="top",
            fontsize=9.5, color=C_GAPTX, fontproperties=FP_MED)

    fig.tight_layout(pad=0.15); _save(fig, outdir, "fig1_teaser")


def fig2_frontier(synth, outdir):
    rows = synth["frontier"]["rows"]; fams = {}
    for r in rows: fams.setdefault(r["family"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    floor = 1.0 - 1.0 / 5  # sliced structural deficit in d=5 (Thm 1)
    panels = [(axes[0], "mem", "peak memory (MB)", "(a) error vs. memory"),
              (axes[1], "time", "wall-clock time (s)", "(b) error vs. time")]
    for ax, xk, xl, ttl in panels:
        # structural sliced-floor guide
        ax.axhline(floor, color=C_SLICE, ls=(0, (5, 3)), lw=0.9, alpha=0.45, zorder=1)
        for fam in ["exact", "entropic", "lowrank", "minibatch", "sliced", "neural"]:
            if fam not in fams: continue
            rs = fams[fam]; xs = np.array([max(r[xk], 1e-3) for r in rs]); ys = np.array([max(r["rel_err_disc"], 5e-4) for r in rs]); o = np.argsort(xs)
            mk(ax, xs[o], ys[o], fam, label=(LABEL[fam] if xk == "mem" else None))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(4e-4, 1.5)
        ax.set_xlabel(xl, color=GREY9, fontproperties=FP_REG)
        ax.set_title(ttl, color=GREY9, fontproperties=FP_MED, fontsize=8.6, loc="left", pad=4)
        style_ax(ax)

    # annotations on the memory panel (left): the structural sliced floor, the
    # certified-but-expensive corner, and Sinkhorn's fixed-memory descent.
    a0 = axes[0]
    a0.text(2e-3, floor * 1.12, r"sliced floor $1\!-\!1/d$", color=C_SLICE,
            fontsize=6.3, fontproperties=FP_MED, va="bottom")
    ent = sorted(fams.get("entropic", []), key=lambda r: r["mem"])
    if ent:
        xe = ent[0]["mem"]
        a0.annotate(r"Sinkhorn: $\varepsilon\!\downarrow$" "\n" "(fixed memory)",
                    xy=(xe * 0.97, 0.32), xytext=(xe * 0.18, 0.30),
                    color=C_SINK, fontsize=6.0, fontproperties=FP_REG, ha="right",
                    va="center", linespacing=1.2,
                    arrowprops=dict(arrowstyle="-|>", color=C_SINK, lw=1.0))
    ex = fams.get("exact", [{}])[0]
    if ex.get("mem"):
        a0.annotate(r"exact LP ($G\!=\!0$)", xy=(ex["mem"], max(ex["rel_err_disc"], 5e-4)),
                    xytext=(ex["mem"] * 0.4, 3e-3), color=C_EXACT, fontsize=6.0,
                    fontproperties=FP_REG, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=GREY5, lw=0.6))

    axes[0].set_ylabel(r"relative error vs. discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
    leg = _leg(axes[0], loc="lower left", handlelength=1.8, fontsize=6.6,
               title="filled $=$ two-sided (certified)")
    plt.setp(leg.get_title(), fontsize=6.0, color=GREY7, fontproperties=FP_REG)
    fig.tight_layout(pad=0.4); _save(fig, outdir, "fig2_frontier")


def fig3_diagnostics(synth, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    gg = synth["gap_geometry"]["rows"]; gaps = np.array([r["gap"] for r in gg]); dist = np.array([r["geo_dist"] for r in gg]); eps = np.array([r["eps_rel"] for r in gg])
    ax = axes[0]; ax.plot(gaps, dist, "-", color=C_SINK, lw=1.3, zorder=2)
    ax.scatter(gaps, dist, c=np.log10(eps), cmap="Blues_r", s=34, edgecolor=C_SINK, linewidth=0.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"certified gap $G$", color=C_GAPTX, fontproperties=FP_MED); ax.set_ylabel("geodesic distortion", color=GREY9, fontproperties=FP_REG)
    ax.set_title("(a) gap tracks geometry", color=GREY9, fontproperties=FP_MED, fontsize=8.7); style_ax(ax)
    dd = synth["dimension"]["rows"]; ds = np.array([r["d"] for r in dd]); sg = np.array([r["sliced_rel_gap"] for r in dd]); st = np.array([r["stat_rel_gap"] for r in dd])
    sgs = np.array([r.get("sliced_rel_gap_std", 0.0) for r in dd])
    ax = axes[1]; dgrid = np.array([1, 2, 4, 8, 16, 32, 64, 128.]); ax.plot(dgrid, 1 - 1 / dgrid, "--", color=GREY5, lw=1.1, label=r"$1-1/d$ (Thm. 1)")
    if np.any(sgs > 0): ax.fill_between(ds, sg - sgs, sg + sgs, color=C_SLICE, alpha=0.18, lw=0, zorder=1)
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
    hyb = np.array([r["relgap_hyb"] for r in rows]); hyb_s = np.array([r.get("relgap_hyb_std", 0.0) for r in rows])
    ent = np.array([r["relgap_ent"] for r in rows]); ent_s = np.array([r.get("relgap_ent_std", 0.0) for r in rows])
    ax = axes[0]; ax.plot(d, 1 - 1 / d, ":", color=GREY5, lw=1.1, label=r"$1-1/d$ (Thm. 1)")
    if np.any(hyb_s > 0): ax.fill_between(d, hyb - hyb_s, hyb + hyb_s, color=C_GAP, alpha=0.18, lw=0, zorder=1)
    if np.any(ent_s > 0): ax.fill_between(d, ent - ent_s, ent + ent_s, color=C_SINK, alpha=0.15, lw=0, zorder=1)
    ax.plot(d, hyb, "D-", color=C_GAP, mfc=C_GAP, mec=C_GAP, ms=5.2, lw=1.6, label="hybrid $U\\!-\\!L$ (near-lin.)")
    ax.plot(d, [r["dual_deficit"] for r in rows], "^--", color=C_SLICE, mfc="white", mec=C_SLICE, mew=1.2, ms=4.6, lw=1.1, label="dual deficit (sliced)")
    ax.plot(d, ent, "o-", color=C_SINK, mfc=C_SINK, mec=C_SINK, ms=4.6, lw=1.3, label="entropic $U\\!-\\!L$ ($O(n^2)$)")
    ax.set_xscale("log", base=2); ax.set_xlabel(r"dimension $d$", color=GREY9, fontproperties=FP_REG); ax.set_ylabel("relative certified gap", color=C_GAPTX, fontproperties=FP_MED); ax.set_ylim(-0.04, 1.12)
    ax.set_title("(a) cheap certificates by composition", color=GREY9, fontproperties=FP_MED, fontsize=8.5); _leg(ax, loc="lower right", handlelength=1.7, fontsize=6.2); style_ax(ax)
    ng = synth["nongauss"]; ax = axes[1]; _frontier_panel(ax, ng["rows"], "rel_err", hybrid=ng["hybrid"])
    ax.set_xlabel("peak memory (MB)", color=GREY9, fontproperties=FP_REG); ax.set_ylabel(r"rel. error vs. discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
    ax.set_title("(b) non-Gaussian frontier (two-moons)", color=GREY9, fontproperties=FP_MED, fontsize=8.5); _leg(ax, loc="lower left", handlelength=1.6, fontsize=6.2); style_ax(ax)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "fig4_hybrid")


def fig_composed_schematic(outdir):
    """Block diagram of the composed certificate: two independent near-linear
    one-sided methods feed a two-sided gap G = U - L (single column)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, fc, ec, txt, sub, tc):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(x, y + 0.32, txt, ha="center", va="center", fontsize=8.5,
                color=tc, fontproperties=FP_MED, zorder=4)
        ax.text(x, y - 0.42, sub, ha="center", va="center", fontsize=6.4,
                color=GREY7, fontproperties=FP_REG, zorder=4)

    # two independent one-sided inputs
    box(2.6, 7.8, 4.0, 1.6, "#FDECEA", C_SLICE, r"Sliced  $L \leq \mathrm{OT}^\star$", "relaxation (near-linear)", C_SLICE)
    box(2.6, 2.2, 4.0, 1.6, "#E6F4EA", C_LOWR, r"Minibatch  $U \geq \mathrm{OT}^\star$", "restriction (near-linear)", C_LOWR)
    # combine node
    box(7.6, 5.0, 3.4, 1.8, SURF_AMBER, C_GAPTX, r"$G = U - L$", "two-sided certificate", C_GAPTX)
    # arrows
    ax.annotate("", xy=(5.9, 5.55), xytext=(4.6, 7.6),
                arrowprops=dict(arrowstyle="-|>", color=C_SLICE, lw=1.5))
    ax.annotate("", xy=(5.9, 4.45), xytext=(4.6, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=C_LOWR, lw=1.5))
    ax.text(5.0, 9.4, "Compose two one-sided methods into a certificate",
            ha="center", va="center", fontsize=8.3, color=GREY9, fontproperties=FP_MED)
    ax.text(7.6, 2.7, "valid for ANY feasible\n" r"$\pi$ and $(f,g)$ (weak duality)",
            ha="center", va="center", fontsize=6.2, color=GREY7, fontproperties=FP_REG)
    fig.tight_layout(pad=0.2); _save(fig, outdir, "fig_composed_schematic")


def fig_memory_scaling(synth, outdir):
    """The OOM wall in two precisions: dense (certifiable) memory grows as O(n^2)
    and hits the GPU VRAM ceiling (float64 first, float32 ~sqrt(2) later), while
    one-sided surrogates stay flat out to n=1e6."""
    ms = synth.get("memory_stress")
    if not ms: return
    dense = ms.get("dense", {})
    # backward-compat: a plain list means single-precision float64
    if isinstance(dense, list):
        dense = {"float64": dense}
        oom_at = {"float64": ms.get("oom_at")}
    else:
        oom_at = ms.get("oom_at", {})
    nbytes = ms.get("bytes", {"float64": 8, "float32": 4})
    dt_style = {"float64": (C_SINK, "o", "-", "float64 (certified)"),
                "float32": (C_MINI, "s", "--", "float32")}
    cap = ms.get("gpu_total_gb")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ng = np.logspace(4, 6, 60)
    _ann = {"float64": (0.42, 3.0), "float32": (1.25, 6.0)}  # stagger OOM labels
    for dt_name, rows in dense.items():
        col, mk_, ls, lab = dt_style.get(dt_name, (C_SINK, "o", "-", dt_name))
        bpe = nbytes.get(dt_name, 8)
        ax.plot(ng, bpe * ng ** 2 / 1e9, ls, color=col, lw=1.0, alpha=0.45)
        dok = [r for r in rows if r.get("ok") and r.get("mem_gb")]
        if dok:
            ax.plot([r["n"] for r in dok], [r["mem_gb"] for r in dok], marker=mk_,
                    color=col, mfc=col, mec=col, ms=5, lw=1.3, label="Sinkhorn " + lab)
        wall = oom_at.get(dt_name) if isinstance(oom_at, dict) else None
        if wall:
            yk = cap if cap else bpe * wall ** 2 / 1e9
            ax.plot([wall], [yk], "X", ms=10, mfc=C_SLICE, mec="white", mew=1.0, zorder=6)
            fx, fy = _ann.get(dt_name, (1.15, 2.2))
            ax.annotate(f"OOM {dt_name[-2:]}\nn={wall/1000:.0f}k", xy=(wall, yk),
                        xytext=(wall * fx, yk * fy), fontsize=5.8, color=C_SLICE,
                        fontproperties=FP_REG, zorder=7)
    if cap:
        ax.axhline(cap, color=GREY7, ls=":", lw=1.1)
        ax.text(1.1e4, cap * 1.1, f"GPU VRAM {cap:.0f} GB", fontsize=6.3,
                color=GREY7, fontproperties=FP_REG)
    lok = [r for r in ms.get("light", []) if r.get("mem_gb")]
    if lok:
        ax.plot([r["n"] for r in lok], [r["mem_gb"] for r in lok], "^-",
                color=C_SLICE, mfc="white", mec=C_SLICE, mew=1.2, ms=5, lw=1.3,
                label="sliced / minibatch")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=3e4)
    ax.set_xlabel(r"sample size $n$", color=GREY9, fontproperties=FP_REG)
    ax.set_ylabel("peak memory (GB)", color=GREY9, fontproperties=FP_REG)
    ax.set_title("Memory wall: certifiable vs one-sided", color=GREY9,
                 fontproperties=FP_MED, fontsize=9)
    leg = ax.legend(loc="lower right", frameon=True, facecolor="white",
                    edgecolor="#DADCE0", framealpha=0.95, borderpad=0.4,
                    handlelength=1.4, fontsize=6.0)
    for t in leg.get_texts(): t.set_fontproperties(FP_REG); t.set_color(GREY9)
    style_ax(ax)
    fig.tight_layout(pad=0.4); _save(fig, outdir, "fig_memory_scaling")


def fig_singlecell_transport(real, outdir):
    """Optimal-transport plan between two cell types in single-cell UMAP space:
    arrows show where the exact plan sends each source cell (geometric distortion
    in a biological embedding)."""
    viz = real.get("_singlecell_viz")
    if viz is None: return
    import ot
    A = np.asarray(viz["umapA"]); B = np.asarray(viz["umapB"])
    Xp = np.asarray(viz["Xpca"]); Yp = np.asarray(viz["Ypca"])
    nA = len(Xp); nB = len(Yp)
    a = np.ones(nA) / nA; b = np.ones(nB) / nB
    M = ot.dist(Xp, Yp, metric="sqeuclidean")
    P = ot.emd(a, b, M)
    mapped = (P @ B) / np.maximum(P.sum(1, keepdims=True), 1e-12)  # barycentric in UMAP
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(A[:, 0], A[:, 1], s=12, c=C_LOWR, alpha=0.7, edgecolor="none",
               label=viz.get("gA", "source"), zorder=3)
    ax.scatter(B[:, 0], B[:, 1], s=12, c=C_SINK, alpha=0.7, edgecolor="none",
               label=viz.get("gB", "target"), zorder=3)
    for i in range(nA):
        ax.annotate("", xy=(mapped[i, 0], mapped[i, 1]), xytext=(A[i, 0], A[i, 1]),
                    arrowprops=dict(arrowstyle="-", color=C_GAP, lw=0.45, alpha=0.5), zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1", color=GREY9, fontproperties=FP_REG)
    ax.set_ylabel("UMAP-2", color=GREY9, fontproperties=FP_REG)
    ax.set_title("OT plan between cell types (UMAP)", color=GREY9,
                 fontproperties=FP_MED, fontsize=9)
    _leg(ax, loc="best", handlelength=1.2, fontsize=6.6)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_color(GREY5)
    fig.tight_layout(pad=0.3); _save(fig, outdir, "fig_singlecell_transport")


def fig_aniso_deficit(synth, outdir):
    """Theorem 1 / E5: sliced deficit vs covariance condition number at fixed d.
    The anisotropic deficit stays high and above the (1-1/d)*rho lower bound; the
    common-covariance case sits exactly on 1-1/d, independent of conditioning."""
    an = synth.get("aniso")
    if not an or not an.get("rows"): return
    r = an["rows"]; d = an.get("d", "?")
    cond = np.array([x["cond"] for x in r]); dfc = np.array([x["deficit"] for x in r])
    dfs = np.array([x.get("deficit_std", 0.0) for x in r])
    comm = np.array([x["deficit_common"] for x in r]); lb = np.array([x["lower_bound"] for x in r])
    omid = r[0]["one_minus_inv_d"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axhline(omid, color=GREY5, ls=":", lw=1.1, label=r"$1-1/d$")
    if np.any(dfs > 0): ax.fill_between(cond, dfc - dfs, dfc + dfs, color=C_SLICE, alpha=0.18, lw=0)
    ax.plot(cond, dfc, "^-", color=C_SLICE, mfc=C_SLICE, mec=C_SLICE, ms=5.4, lw=1.4,
            label=r"anisotropic ($\Sigma_0\!\neq\!\Sigma_1$)")
    ax.plot(cond, comm, "s--", color=C_LOWR, mfc="white", mec=C_LOWR, mew=1.3, ms=5, lw=1.2,
            label=r"common $\Sigma$ (Thm 1(i))")
    ax.plot(cond, lb, "o:", color=C_GAPTX, mfc="white", mec=C_GAPTX, mew=1.2, ms=4.4, lw=1.1,
            label=r"lower bound $(1\!-\!1/d)\rho$ (Thm 1(ii))")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"covariance condition number $\kappa$", color=GREY9, fontproperties=FP_REG)
    ax.set_ylabel("sliced relative deficit", color=GREY9, fontproperties=FP_REG)
    ax.set_title(f"Anisotropic deficit ($d={d}$)", color=GREY9, fontproperties=FP_MED, fontsize=9)
    _leg(ax, loc="lower right", handlelength=1.6, fontsize=6.0); style_ax(ax)
    fig.tight_layout(pad=0.4); _save(fig, outdir, "fig_aniso_deficit")


def fig_gap_selection(synth, real, outdir):
    """(a) Only the certified gap upper-bounds the two-sided error (proxy validity);
    (b) gap-selected eps matches the oracle and beats unsupervised heuristics."""
    px = synth.get("proxy"); dec = (real or {}).get("decision", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    # --- panel (a): proxy validity scatter ---
    # Among proxies COMPUTABLE without the ground-truth OT* (the certified gap and
    # the Sinkhorn divergence), only the gap is a valid bound on the two-sided
    # error. The primal excess / dual deficit need OT* (oracle-only); we draw them
    # faintly to expose the decomposition.
    ax = axes[0]
    if px and px.get("rows"):
        rows = px["rows"]
        ce = np.array([r["cert_error"] for r in rows])
        lim = max(ce.max(), max(np.max([r[k] for r in rows]) for k in ["gap", "primal", "dual", "sinkdiv"])) * 1.2
        ax.plot([1e-3, lim], [1e-3, lim], "-", color=GREY7, lw=0.9, zorder=1)
        # oracle-only quantities, faint
        for k, c, mk_ in [("primal", C_LOWR, "s"), ("dual", C_SLICE, "^")]:
            ax.scatter(ce, [r[k] for r in rows], s=16, c=c, marker=mk_, edgecolor="none",
                       alpha=0.25, zorder=2)
        # computable proxies, prominent
        ax.scatter(ce, [r["sinkdiv"] for r in rows], s=26, c=C_MINI, marker="o", edgecolor="none",
                   alpha=0.85, label="Sinkhorn div. (computable)", zorder=3)
        ax.scatter(ce, [r["gap"] for r in rows], s=30, c=C_GAP, marker="D", edgecolor="none",
                   alpha=0.9, label="certified gap (computable)", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"two-sided error $\max(U\!-\!\mathrm{OT}^\star,\mathrm{OT}^\star\!-\!L)/\mathrm{OT}^\star$",
                      color=GREY9, fontproperties=FP_REG, fontsize=7.0)
        ax.set_ylabel("proxy value", color=GREY9, fontproperties=FP_REG)
        ax.set_title("(a) the gap is the only computable bound", color=GREY9, fontproperties=FP_MED, fontsize=8.3)
        _leg(ax, loc="lower right", handlelength=1.3, fontsize=6.0); style_ax(ax)
        vb = px.get("valid_bound_fraction", {})
        ax.text(0.04, 0.96,
                f"valid bound: gap {100*vb.get('gap',0):.0f}%,  "
                f"Sinkhorn-div {100*vb.get('sinkdiv',0):.0f}%\n(primal/dual need $\\mathrm{{OT}}^\\star$; shown faint)",
                transform=ax.transAxes, va="top", fontsize=5.6, color=GREY7, fontproperties=FP_REG)
    # --- panel (b): decision usefulness grouped bars ---
    ax = axes[1]
    order = ["gap_tol", "gap_knee", "obj_elbow", "sinkdiv", "fixed", "oracle_best"]
    pretty = {"gap_tol": "gap-tol", "gap_knee": "gap-knee", "obj_elbow": "obj-elbow",
              "sinkdiv": "sinkdiv", "fixed": r"fixed $\varepsilon$", "oracle_best": "oracle"}
    names = [n for n in ["mnist_usps", "single_cell"] if n in dec]
    if names:
        x = np.arange(len(order)); w = 0.8 / max(len(names), 1)
        cols = {"mnist_usps": C_SINK, "single_cell": C_LOWR}
        for j, nm in enumerate(names):
            picks = dec[nm]["picks"]
            ys = [picks.get(r, {}).get("acc", np.nan) for r in order]
            bars = ax.bar(x + j * w, ys, w, color=cols.get(nm, GREY5),
                          label=PRETTY.get(nm, nm), edgecolor="white", linewidth=0.6)
            # mark our gap rules
        ax.set_xticks(x + w * (len(names) - 1) / 2)
        ax.set_xticklabels([pretty[r] for r in order], rotation=30, ha="right", fontsize=6.4)
        ax.set_ylabel("transfer accuracy", color=GREY9, fontproperties=FP_REG)
        ax.set_title("(b) gap-selected $\\varepsilon$ vs heuristics", color=GREY9,
                     fontproperties=FP_MED, fontsize=8.5)
        _leg(ax, loc="lower left", handlelength=1.2, fontsize=6.2); style_ax(ax)
    fig.tight_layout(pad=0.5); _save(fig, outdir, "fig_gap_selection")


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
    axes[0][0].set_ylabel(r"rel. error vs. discrete $\mathrm{OT}^\star$", color=GREY9, fontproperties=FP_REG)
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
        ax.plot(d, sg, marker=marks[name], color=colors[name], mfc=colors[name], mec=colors[name], ms=5, lw=1.4, label=PRETTY[name] + " (PCA)")
    # nonlinear (diffusion-map) overlay for single-cell: does 1-1/d survive?
    sc = real["datasets"].get("single_cell", {})
    if "dimension_nonlinear" in sc:
        rows = sc["dimension_nonlinear"]["rows"]
        d = np.array([r["d"] for r in rows], float); sg = np.array([r["sliced_rel_gap"] for r in rows])
        dmax = max(dmax, d.max())
        ax.plot(d, sg, marker="s", color=C_LOWR, mfc="white", mec=C_LOWR, mew=1.3,
                ms=5, lw=1.2, ls="--", label="single-cell (diffmap)")
    dg = np.logspace(0, np.log2(dmax), 50, base=2)
    ax.plot(dg, 1 - 1 / dg, ":", color=GREY5, lw=1.2, label=r"$1-1/d$ (Thm. 1)")
    ax.set_xscale("log", base=2); ax.set_xlabel(r"effective dimension $d$ (PCA / diffmap)", color=GREY9, fontproperties=FP_REG)
    ax.set_ylabel("sliced relative gap", color=GREY9, fontproperties=FP_REG); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Dimensional deficit on real data", color=GREY9, fontproperties=FP_MED, fontsize=9)
    _leg(ax, loc="lower right", handlelength=1.7, fontsize=6.6); style_ax(ax)
    fig.tight_layout(pad=0.4); _save(fig, outdir, "real_dimension")


def _gap_acc_panel(ax, res, title):
    """One gap-vs-accuracy panel from a da_gap_vs_accuracy result dict.
    Draws error bars when the result carries per-eps std (multi-seed)."""
    rows = [r for r in res["rows"] if r["method"] == "sinkhorn"]
    g = np.array([r["relgap"] for r in rows]); acc = np.array([r["acc"] for r in rows])
    gs = np.array([r.get("relgap_std", 0.0) for r in rows])
    accs = np.array([r.get("acc_std", 0.0) for r in rows])
    if np.any(gs > 0) or np.any(accs > 0):
        ax.errorbar(g, acc, xerr=gs, yerr=accs, fmt="none", ecolor=GREY5,
                    elinewidth=0.8, capsize=2, zorder=2)
    ax.scatter(g, acc, c=np.log10([r["eps"] for r in rows]), cmap="Blues_r",
               s=46, edgecolor=C_SINK, linewidth=1.0, zorder=3)
    o = np.argsort(g); ax.plot(g[o], acc[o], "-", color=C_SINK, lw=1.2, zorder=2)
    ex = [r for r in res["rows"] if r["method"] == "exact"]
    if ex: ax.axhline(ex[0]["acc"], color=C_EXACT, ls="--", lw=1.0, label="exact-OT transfer")
    ax.set_xlabel(r"certified gap $G/\mathrm{OT}^\star$", color=C_GAPTX, fontproperties=FP_MED)
    ax.set_ylabel("label-transfer accuracy", color=GREY9, fontproperties=FP_REG)
    ax.set_title(title, color=GREY9, fontproperties=FP_MED, fontsize=8.3)
    _leg(ax, loc="lower left", handlelength=1.6, fontsize=6.4); style_ax(ax)


def real_gap_accuracy(real, outdir):
    has_da = "da_accuracy" in real
    has_scda = "sc_da_accuracy" in real
    has_sc = "scaling" in real and len(real["scaling"].get("sinkhorn", [])) > 1
    panels = []
    if has_da:   panels.append("da")
    if has_scda: panels.append("scda")
    if has_sc:   panels.append("scaling")
    ncol = len(panels)
    if ncol == 0: return
    tags = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 2.7), squeeze=False)
    for i, p in enumerate(panels):
        ax = axes[0][i]; tag = tags[i]
        if p == "da":
            _gap_acc_panel(ax, real["da_accuracy"], f"{tag} gap predicts transfer (MNIST$\\to$USPS)")
        elif p == "scda":
            _gap_acc_panel(ax, real["sc_da_accuracy"], f"{tag} gap predicts cell-type transfer")
        elif p == "scaling":
            sc = real["scaling"]
            fmap = {"exact": "exact", "sinkhorn": "entropic", "lowrank": "lowrank",
                    "sliced": "sliced", "minibatch": "minibatch"}
            lmap = {"exact": "exact LP", "sinkhorn": "Sinkhorn (GPU)", "lowrank": "low-rank",
                    "sliced": "sliced", "minibatch": "minibatch"}
            for key in ["exact", "sinkhorn", "lowrank", "sliced"]:
                v = sc.get(key, [])
                if len(v) < 1: continue
                ns = np.array([r["n"] for r in v]); ts = np.array([r["time"] for r in v])
                mk(ax, ns, ts, fmap[key], label=lmap[key], ms=4.6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"sample size $n$", color=GREY9, fontproperties=FP_REG)
            ax.set_ylabel("time (s)", color=GREY9, fontproperties=FP_REG)
            ax.set_title(f"{tag} cost of certifiability at scale", color=GREY9,
                         fontproperties=FP_MED, fontsize=8.3)
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


def fig_proxy_scatter(synth, outdir):
    """4-panel full proxy-validity scatter (appendix).
    Each panel shows one proxy vs the true two-sided certification error with the
    identity line; the red-shaded region below it marks 'not a valid bound'.
    (a) certified gap: computable AND always above the line.
    (b) Sinkhorn divergence: computable but frequently below the line.
    (c) primal excess / (d) dual deficit: valid bounds but require OT*."""
    px = synth.get("proxy")
    if not px or not px.get("rows"):
        return
    rows = px["rows"]
    ce = np.array([r["cert_error"] for r in rows])
    eps_log = np.log10([r["eps"] for r in rows])
    vb = px.get("valid_bound_fraction", {})
    sp = px.get("spearman", {})
    panel_specs = [
        ("gap",     r"certified gap $G/\mathrm{OT}^\star$",
         C_GAP,   "D", r"(computable — no $\mathrm{OT}^\star$ needed)"),
        ("sinkdiv", r"Sinkhorn divergence $S_\varepsilon/\mathrm{OT}^\star$",
         C_MINI,  "o", r"(computable — no $\mathrm{OT}^\star$ needed)"),
        ("primal",  r"primal excess $(U\!-\!\mathrm{OT}^\star)/\mathrm{OT}^\star$",
         C_LOWR,  "s", r"(requires $\mathrm{OT}^\star$)"),
        ("dual",    r"dual deficit $(\mathrm{OT}^\star\!-\!L)/\mathrm{OT}^\star$",
         C_SLICE, "^", r"(requires $\mathrm{OT}^\star$)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8))
    axes = axes.ravel()
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for i, (key, ylabel, col, mk_, note) in enumerate(panel_specs):
        ax = axes[i]
        vals = np.array([r[key] for r in rows])
        lim = max(float(ce.max()), float(vals.max())) * 1.18
        lo = -lim * 0.03
        ax.plot([lo, lim], [lo, lim], "-", color=GREY7, lw=1.0, zorder=1)
        ax.fill_between([lo, lim], [lo, lo], [lo, lim],
                        color="#FCE8E6", alpha=0.50, zorder=0)
        ax.scatter(ce, vals, c=eps_log, cmap="Blues_r", s=22, marker=mk_,
                   edgecolor=col, linewidth=0.7, alpha=0.85, zorder=3)
        ax.set_xlim(lo, lim); ax.set_ylim(lo, lim)
        pct = 100.0 * vb.get(key, float("nan"))
        rho = sp.get(key, float("nan"))
        ax.set_title(
            f"{tags[i]}  valid bound: {pct:.0f}%   Spearman: {rho:+.2f}\n{note}",
            color=GREY9, fontproperties=FP_REG, fontsize=6.8, linespacing=1.35)
        ax.set_xlabel("two-sided error", color=GREY9,
                      fontproperties=FP_REG, fontsize=6.5)
        ax.set_ylabel(ylabel, color=col, fontproperties=FP_MED, fontsize=6.8)
        style_ax(ax)
    fig.tight_layout(pad=0.9)
    _save(fig, outdir, "fig_proxy_scatter")


def color_transfer_methods(real, outdir):
    """4-panel: source | exact LP (G=0) | Sinkhorn tight | Sinkhorn loose.
    Each panel's border and title color matches its method; relgap is annotated.
    This makes the gap–fidelity link directly visible: tight gap = faithful colors,
    loose gap = washed-out / wrong hues."""
    imgs = real.get("_color_imgs")
    if imgs is None:
        return
    import ot
    from skimage import color as skcolor
    from scipy.spatial import cKDTree
    import otkit as K
    src = np.asarray(imgs["src_sample"], float)
    tgt = np.asarray(imgs["tgt_sample"], float)
    n = len(src); a = np.ones(n) / n; b = np.ones(n) / n
    M = ot.dist(src, tgt, metric="sqeuclidean")
    full = np.asarray(imgs["src_lab_full"])
    tree = cKDTree(src); _, idx = tree.query(full, k=1)
    src_img = np.asarray(imgs["src_img"])

    def bary(P):
        P = np.asarray(P, float)
        mapped = (P @ tgt) / np.maximum(P.sum(1, keepdims=True), 1e-12)
        return np.clip(skcolor.lab2rgb(mapped[idx].reshape(src_img.shape)), 0, 1)

    P_ex = ot.emd(a, b, M)
    OTstar = float(np.sum(P_ex * M))
    panels = [(src_img, "source\n(original)", GREY7)]
    panels.append((bary(P_ex), "Exact LP\n$G=0$   (relgap = 0.00)", C_EXACT))
    for eps_rel, lbl in [(0.02, r"$\varepsilon\!=\!0.02$  (tight)"),
                          (0.50, r"$\varepsilon\!=\!0.50$  (loose)")]:
        _, U, L, _, Ps = K.sinkhorn_gap(M, a, b, eps_rel, use_gpu=False,
                                          numItermax=5000, stopThr=1e-7,
                                          return_plan=True)
        rg = (U - L) / max(OTstar, 1e-12)
        col = C_SINK if rg < 0.3 else C_SLICE
        panels.append((bary(Ps),
                        f"Sinkhorn  {lbl}\nrelgap = {rg:.2f}", col))

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.8),
                              gridspec_kw={"wspace": 0.06})
    for ax, (img, ttl, col) in zip(axes, panels):
        ax.imshow(img); ax.axis("off")
        ax.set_title(ttl, color=col, fontproperties=FP_MED, fontsize=7.4,
                     linespacing=1.25, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2.2)
    fig.tight_layout(pad=0.2)
    _save(fig, outdir, "color_transfer_methods")


# =================================================== driver
def generate_all(synth, real, outdir, font_dir="."):
    _setup(font_dir); os.makedirs(outdir, exist_ok=True); made = []
    try: fig1_teaser(outdir); made.append("fig1_teaser")
    except Exception as e: print("fig1 failed:", e)
    try: fig_composed_schematic(outdir); made.append("fig_composed_schematic")
    except Exception as e: print("fig_composed_schematic failed:", e)
    for fn, nm in [(fig2_frontier, "fig2_frontier"), (fig3_diagnostics, "fig3_diagnostics"),
                   (fig4_hybrid, "fig4_hybrid"), (fig_memory_scaling, "fig_memory_scaling"),
                   (fig_aniso_deficit, "fig_aniso_deficit")]:
        try: fn(synth, outdir); made.append(nm)
        except Exception as e: print(nm, "failed:", e)
    try: fig_gap_selection(synth, real, outdir); made.append("fig_gap_selection")
    except Exception as e: print("fig_gap_selection failed:", e)
    try: fig_proxy_scatter(synth, outdir); made.append("fig_proxy_scatter")
    except Exception as e: print("fig_proxy_scatter failed:", e)
    for fn, nm in [(real_frontier, "real_frontier"), (real_dimension, "real_dimension"),
                   (real_gap_accuracy, "real_gap_accuracy"), (color_transfer, "color_transfer"),
                   (color_transfer_methods, "color_transfer_methods"),
                   (fig_singlecell_transport, "fig_singlecell_transport")]:
        try: fn(real, outdir); made.append(nm)
        except Exception as e: print(nm, "failed:", e)
    return made
