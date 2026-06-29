# ATLAS: Adaptive Testing for LLM Evaluation

<p align="center">
  <a href="https://icml.cc/virtual/2026/poster/64880"><img src="https://img.shields.io/badge/ICML_2026-Spotlight-blue.svg" alt="ICML 2026 Spotlight"></a>
  <a href="https://arxiv.org/abs/2511.04689"><img src="https://img.shields.io/badge/arXiv-2511.04689-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Peiyu-Georgia-Li/ATLAS"><img src="https://img.shields.io/github/stars/Peiyu-Georgia-Li/ATLAS?style=social" alt="GitHub Stars"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/文档-中文版-red.svg" alt="中文版"></a>
</p>

> **TL;DR** — ATLAS replaces static LLM benchmarks with adaptive testing grounded in **Item Response Theory (IRT)**. It reduces the number of required evaluation items by up to **90%** — matching full HellaSwag results with just **41 out of 5,608 items** — while providing finer ability estimates (θ) that reveal ranking shifts for **23–31% of models** compared to standard accuracy metrics.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Supported Benchmarks](#supported-benchmarks)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Output Files](#output-files)
- [Reproducibility](#reproducibility)
- [IRT Model](#irt-model)
- [Citation](#citation)

---

## How It Works

1. **IRT Calibration** — Fit a 3PL IRT model to a training set of LLM responses, partitioned into chunks for scalability. Chunk parameters are linked to a common scale via mean–sigma transformations.
2. **WLE Scoring** — Estimate full-test baseline ability (θ) for each model using Weighted Likelihood Estimation over all items. This serves as the ground-truth θ for evaluation.
3. **Adaptive Testing (ATLAS)** — For each model, adaptively select items by Maximum Fisher Information starting from θ = 0. Stop when SE(θ) ≤ threshold.
4. **p-IRT Accuracy** — Estimate full-benchmark accuracy from the adaptive subset using observed scores + IRT-predicted probabilities for unseen items.
5. **Actual Accuracy** — Compute ground-truth accuracy from the full test response matrix for comparison.
6. **Analysis** — Compare p-IRT estimates against actual accuracy; summarize MAE across benchmarks and methods.

### Supported Benchmarks

| Benchmark | Models | Items | Chunks |
|-----------|--------|-------|--------|
| ARC | 4,162 | 842 | 8 |
| GSM8K | 4,195 | 1,307 | 12 |
| HellaSwag | 3,467 | 5,608 | 50 |
| TruthfulQA | 4,635 | 628 | 6 |
| WinoGrande | 4,680 | 1,046 | 10 |

---

## Repository Structure

```
ATLAS/
├── scripts/
│   ├── 01_fit_irt.r                 # Partition-based 3PL IRT fitting
│   ├── 02_wle_scoring.r             # Mean–sigma linking + WLE estimation
│   ├── 03_atlas_cat.r               # Adaptive testing (ATLAS)
│   ├── 04_pirt_accuracy.r           # p-IRT accuracy estimation
│   ├── 05_compute_actual_acc.r      # Ground-truth accuracy from full matrix
│   ├── submit_pipeline.sh           # SGE cluster job submission
│   └── analysis/
│       ├── compare_pirt_actual.r    # p-IRT vs actual accuracy comparison
│       ├── summarize_pirt_for_all_method.py   # MAE summary across benchmarks
│       └── summarize_theta_for_all_method.py  # θ MAE vs WLE ground truth
│
├── arc/                             # Per-benchmark results (same structure for all)
│   ├── irt_item_parameters_combined.csv      # Linked 3PL item parameters
│   ├── irt_person_scores_WLE_SE_test.csv     # Full-test WLE θ (ground truth)
│   ├── actual_accuracy.csv                   # Ground-truth accuracy (step 5)
│   ├── pirt_accuracy_se_0.1.csv              # p-IRT accuracy estimates (step 4)
│   ├── pirt_accuracy_se_0.2.csv
│   ├── pirt_accuracy_se_0.3.csv
│   ├── pirt_vs_actual_se_0.1.csv             # Merged comparison + error metrics
│   ├── pirt_vs_actual_se_0.2.csv
│   ├── pirt_vs_actual_se_0.3.csv
│   └── atlas_arc_random/
│       ├── irt_person_scores_ATLAS_<SE>.csv  # Adaptive θ estimates
│       ├── item_selection_frequency_<SE>.csv # Item selection counts
│       └── selected_items.zip                # Per-model item sequences (all SE thresholds)
├── gsm8k/
├── hellaswag/
├── truthfulqa/
├── winogrande/
│
├── experiments/                     # Ablation studies
│   ├── arc_2pl/                     # 2PL model comparison
│   ├── baseline_compare/            # TinyBenchmark / MetaBench baselines
│   └── ...
│
├── data/                            # Response matrices (not committed; see Setup)
│
├── rmse_cat_summary.csv             # RMSE summary across benchmarks and SE thresholds
├── summary_pirt_mae_sd_se.csv       # Cross-benchmark p-IRT MAE summary
├── summary_theta_mae_sd_se.csv      # Cross-benchmark θ MAE summary
├── run_pipeline.sh                  # Full pipeline (local, parallel)
└── run_pirt_all_benchmarks.sh       # Steps 4–6 only (post-ATLAS)
```

---

## Setup

### 1. Data

Response matrices are not committed to the repository. Place the CSV files in the `data/` directory:

```
data/
├── gaussian_sampled_arc_response_matrix_test.csv
├── gaussian_sampled_gsm8k_response_matrix_test.csv
├── gaussian_sampled_hellaswag_response_matrix_test.csv
└── ...
```

### 2. R Dependencies

```r
install.packages(c("mirt", "catR", "parallel", "doParallel", "foreach"))
```

R ≥ 4.4 recommended.

### 3. Python Dependencies

```bash
pip install pandas numpy
```

---

## Running the Pipeline

### Option A — Local (all benchmarks, fully parallel)

```bash
bash run_pipeline.sh --benchmarks=arc,gsm8k,hellaswag,truthfulqa,winogrande \
                     --se=0.1,0.2,0.3 --n_cores=8
```

### Option B — SGE Cluster (one benchmark per submission)

```bash
# Submit all steps for one benchmark as chained SGE jobs
bash scripts/submit_pipeline.sh truthfulqa long

# After all benchmarks complete, run global summaries manually:
python3 scripts/analysis/summarize_pirt_for_all_method.py
python3 scripts/analysis/summarize_theta_for_all_method.py
```

### Option C — Steps 4–6 only (when ATLAS is already done)

```bash
bash run_pirt_all_benchmarks.sh
```

---

### Step-by-Step

**Step 1 — IRT fitting** (one chunk per call, run all chunks in parallel):

```bash
# TruthfulQA: 6 chunks
for i in 105 209 313 418 523 628; do
  Rscript scripts/01_fit_irt.r --benchmark=truthfulqa --chunk_end=$i &
done; wait

# ARC: 8 chunks
for i in $(seq 106 105 737) 840; do
  Rscript scripts/01_fit_irt.r --benchmark=arc --chunk_end=$i &
done; wait
```

Chunk indices per benchmark:

| Benchmark | Chunk ends |
|-----------|-----------|
| ARC | `seq(106, 737, 105)` + 840 |
| GSM8K | `seq(110, 1200, 109)` + 1307 |
| HellaSwag | `seq(113, 5501, 112)` + 5608 |
| TruthfulQA | 105, 209, 313, 418, 523, 628 |
| WinoGrande | 105, 209, 313, 417, 521, 626, 731, 836, 941, 1046 |

**Step 2 — WLE scoring** (links chunks, estimates full-test θ):

```bash
Rscript scripts/02_wle_scoring.r --benchmark=truthfulqa
```

**Step 3 — Adaptive testing**:

```bash
Rscript scripts/03_atlas_cat.r --benchmark_name=truthfulqa \
        --se_theta_stop=0.1 --n_cores=8
```

`--se_theta_stop`: stopping threshold (0.1 = high precision, 0.3 = fast).

**Step 4 — p-IRT accuracy**:

```bash
Rscript scripts/04_pirt_accuracy.r --benchmark=truthfulqa --se_theta_stop=0.1
```

**Step 5 — Actual accuracy** (ground truth for comparison):

```bash
Rscript scripts/05_compute_actual_acc.r --benchmark=truthfulqa
```

**Step 6 — Compare and summarize**:

```bash
# Per-benchmark comparison (repeat for each SE and benchmark)
Rscript scripts/analysis/compare_pirt_actual.r \
        --benchmark=truthfulqa --se_theta_stop=0.1

# Global MAE summaries across all benchmarks (run once all benchmarks are done)
python3 scripts/analysis/summarize_pirt_for_all_method.py
python3 scripts/analysis/summarize_theta_for_all_method.py
```

---

## Output Files

### Per-benchmark (`{benchmark}/`)

| File | Description |
|------|-------------|
| `irt_item_parameters_combined.csv` | Linked 3PL item parameters (a1, d, g, u) |
| `irt_person_scores_WLE_SE_test.csv` | Full-test WLE θ — ground truth for evaluation |
| `pirt_accuracy_se_<SE>.csv` | p-IRT accuracy estimates per model |
| `actual_accuracy.csv` | Ground-truth accuracy across all items |
| `pirt_vs_actual_se_<SE>.csv` | Merged comparison with error, abs_error, squared_error |
| `pirt_vs_actual_se_<SE>.png` | Scatter plot: p-IRT vs actual accuracy |
| `atlas_*/irt_person_scores_ATLAS_<SE>.csv` | Adaptive θ estimates with item counts |
| `atlas_*/item_selection_frequency_<SE>.csv` | How often each item was selected |
| `atlas_*/selected_items.zip` | Per-model item sequences for all SE thresholds |

### Global (repo root)

| File | Description |
|------|-------------|
| `rmse_cat_summary.csv` | RMSE across benchmarks and SE thresholds |
| `summary_pirt_mae_sd_se.csv` | p-IRT MAE/SD/SE across all benchmarks and SE thresholds |
| `summary_theta_mae_sd_se.csv` | θ MAE vs WLE ground truth across methods and benchmarks |

---

## Reproducibility

The adaptive item selection in Step 3 uses a per-model random seed, making results deterministic given the same seed. Pre-computed item sequences are provided in `atlas_*/selected_items.zip` for exact reproduction of reported numbers without re-running ATLAS.

To unzip and use the pre-computed selections:

```bash
# Unzip for one benchmark
cd arc/atlas_arc_random && unzip selected_items.zip

# Then run steps 4–6 directly
Rscript scripts/04_pirt_accuracy.r --benchmark=arc --se_theta_stop=0.1
```

To reproduce from scratch (re-runs Step 3):

```bash
Rscript scripts/03_atlas_cat.r --benchmark_name=arc --se_theta_stop=0.1 --n_cores=8
```

---

## IRT Model

**3-Parameter Logistic (3PL):**

$$P(X_{ij}=1 \mid \theta_j) = g_i + (1-g_i)\cdot\frac{1}{1+\exp(-(a_i\theta_j + d_i))}$$

- **$a_i$**: discrimination
- **$d_i$**: difficulty-related ($b_i = -d_i/a_i$)
- **$g_i$**: pseudo-guessing (lower asymptote)

**Mean–sigma linking** from partition *k* to reference:

$$A = \frac{\text{sd}(\theta_\text{ref})}{\text{sd}(\theta_k)}, \quad B = \bar\theta_\text{ref} - A\bar\theta_k$$

$$a_i^* = a_i/A, \quad d_i^* = A\cdot d_i + B\cdot a_i$$

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026atlas,
  title     = {Adaptive Testing for {LLM} Evaluation: A Psychometric Alternative to Static Benchmarks},
  author    = {Li, Peiyu and Tang, Xiuxiu and Chen, Si and Cheng, Ying and Metoyer, Ronald and Hua, Ting and Chawla, Nitesh},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026},
  note      = {Spotlight},
  eprint    = {2511.04689},
  archivePrefix = {arXiv}
}
```
