# DualityOT — certified Kantorovich-gap experiments

Companion code for **"A Duality Principle for Optimal Transport — Bridging
Geometric and Computational Perspectives."** It implements the certified
Kantorovich duality-gap coordinate

```
G = U − L ,   with   L ≤ OT* ≤ U
```

(`U` = cost of a marginal-rounded coupling, a rigorous **upper** bound;
`L = a·f + b·g` for a c-transform dual pair, a rigorous **lower** bound), and
uses it to (i) reproduce the paper's **synthetic** figures on closed-form
Bures–Wasserstein ground truth and (ii) run the same framework on **real public
datasets**.

All Sinkhorn solves run in the **log domain** (`method="sinkhorn_log"`), so the
certified bounds stay valid even at the smallest `eps` swept — where the plain
kernel `exp(−M/reg)` would underflow.

## Run it on Google Colab (A100), from a clean clone

1. Open **`run_all.ipynb`** in Colab
   ([github.com/kagtgi/DualityOT](https://github.com/kagtgi/DualityOT) → open in
   Colab, or upload the notebook).
2. **Runtime ▸ Change runtime type ▸ A100 GPU** (any GPU works; A100 is fastest; CPU also works).
3. **Runtime ▸ Run all.**

The first cell `git clone`s this repo, the second installs only the Colab-missing
extras (so Colab's CUDA build of `torch` is preserved), and the rest runs every
experiment and writes results. A full run is ≈30–60 min on an A100; set
`FAST = True` in the config cell for a ~3-minute smoke test.

When it finishes, all results are in **`results/`** and packaged into
**`ot_duality_results.zip`**, which downloads automatically:

```
results/metrics/results_synth.json   all synthetic numbers
results/metrics/results_real.json    all real-data numbers
results/figures/*.pdf, *.png         fig1–4 (paper) + real_frontier, real_dimension,
                                     real_gap_accuracy, color_transfer
results/color_images.npz             source/target/recolored arrays (color transfer)
results/logs/run.log                 full run log
results/manifest.json                timings, GPU, dataset provenance, versions
```

## What it runs

**Synthetic** (Bures–Wasserstein closed form is the exact ground truth):
fidelity–cost frontier, gap-tracks-geometry, dimensional `1 − 1/d` deficit,
cost-of-certifiability scaling, composed certificate vs. dimension, and a
non-Gaussian (two-moons) frontier.

**Real public datasets** (OT is genuinely used in each; subsamples are kept small
enough that an *exact discrete LP* optimum is computable as ground truth):

| dataset | domain | dim | OT use |
|---|---|---|---|
| PBMC scRNA-seq | single-cell genomics | ~50 (PCA) | Waddington-OT cell transport |
| MNIST ↔ USPS | image domain adaptation | 50 (PCA) | OT domain adaptation |
| astronaut → coffee | color transfer | 3 (Lab) | OT color matching |

Real-data experiments (each maps onto a paper claim):
- **Frontier** — relative error vs. cost for every family, scored against the
  exact discrete optimum (two-sided families certify low error; one-sided trade off).
- **Composed certificate** — the cheap two-sided certificate `G = U_minibatch − L_sliced`.
- **Dimensional deficit** — sliced relative gap vs. PCA dimension (the `1 − 1/d` law).
- **Cost of certifiability at scale** — GPU Sinkhorn pushed to n ≈ 32k on real MNIST
  while the exact LP becomes infeasible.
- **Gap predicts downstream quality** — on MNIST→USPS the *label-free* certified gap
  tracks label-transfer accuracy.

Every dataset loader **falls back gracefully** if a download is unavailable, so a
run never crashes.

## Repository layout

| file | role |
|---|---|
| `otkit.py` | gap machinery (ANWR rounding → `U`, c-transform → `L`), estimators (exact LP, log-stabilized GPU Sinkhorn, low-rank, sliced, minibatch), runners, Bures/BW closed forms |
| `datasets.py` | real-data loaders with graceful fallbacks |
| `experiments_synth.py` | synthetic suite (`run_synth`) |
| `experiments_real.py` | real-data suite (`run_real`) |
| `make_figures.py` | all figures (PDF + PNG; Google/Material palette + bundled Roboto) |
| `run_all.ipynb` | end-to-end Colab driver (clone → install → run → save → zip) |
| `tests/test_bounds.py` | asserts `L ≤ OT* ≤ U` and the closed-form identities |
| `requirements.txt` | full local environment |
| `requirements-colab.txt` | Colab-only extras (never reinstalls torch) |
| `Roboto-*.ttf` | fonts so the figures match the paper |

## Local use

```bash
git clone https://github.com/kagtgi/DualityOT.git && cd DualityOT
pip install -r requirements.txt          # NOT on Colab (would replace torch)
python tests/test_bounds.py              # quick correctness check
python -c "import experiments_synth as S, experiments_real as R, make_figures as F; \
           s=S.run_synth(); r=R.run_real(); F.generate_all(s, r, 'results/figures', font_dir='.')"
```

GPU is used automatically when PyTorch + CUDA are available, and falls back to CPU
otherwise.
