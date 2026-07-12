# ATLAS：大语言模型自适应评测框架

<p align="center">
  <a href="https://icml.cc/virtual/2026/poster/64880"><img src="https://img.shields.io/badge/ICML_2026-Spotlight-blue.svg" alt="ICML 2026 Spotlight"></a>
  <a href="https://arxiv.org/abs/2511.04689"><img src="https://img.shields.io/badge/arXiv-2511.04689-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Peiyu-Georgia-Li/ATLAS"><img src="https://img.shields.io/github/stars/Peiyu-Georgia-Li/ATLAS?style=social" alt="GitHub Stars"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Docs-English-blue.svg" alt="English"></a>
</p>

> **TL;DR** — ATLAS 以**项目反应理论（IRT）**为基础，将静态基准测评替换为自适应测试。测评所需题目数量最多减少 **90%** —— 在 HellaSwag 仅用 **41 道题（共 5,608 道）**即可匹配全量结果 —— 同时提供更细粒度的能力值（θ）估计，发现 **23–31% 的模型**在排名上与传统准确率指标存在显著偏移。

---

## 目录

- [工作原理](#工作原理)
- [支持的基准](#支持的基准)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [运行流程](#运行流程)
- [输出文件说明](#输出文件说明)
- [可复现性说明](#可复现性说明)
- [IRT 模型说明](#irt-模型说明)
- [引用](#引用)

---

## 工作原理

1. **IRT 参数估计** — 对 LLM 训练集响应矩阵拟合三参数逻辑斯蒂（3PL）IRT 模型，采用分块策略提升可扩展性，并通过均值-标准差变换将各块参数链接到统一量表。
2. **WLE 全量打分** — 用加权似然估计（WLE）对所有题目计算每个模型的基准能力值 θ，作为评估的 ground truth。
3. **自适应测试（ATLAS）** — 对每个模型，从 θ = 0 出发，按最大 Fisher 信息准则自适应选题，当 SE(θ) ≤ 阈值时停止。
4. **p-IRT 准确率估计** — 用自适应子集的观测得分 + IRT 预测概率估计全量基准的准确率。
5. **实际准确率** — 从完整测试响应矩阵计算真实准确率，用于对比评估。
6. **分析汇总** — 对比 p-IRT 估计与实际准确率，汇总各基准和方法的 MAE 指标。

### 支持的基准

| 基准 | 模型数 | 题目数 | Chunk 数 |
|------|--------|--------|----------|
| ARC | 4,162 | 842 | 8 |
| GSM8K | 4,195 | 1,307 | 12 |
| HellaSwag | 3,467 | 5,608 | 50 |
| TruthfulQA | 4,635 | 628 | 6 |
| WinoGrande | 4,680 | 1,046 | 10 |

---

## 目录结构

```
ATLAS/
├── scripts/
│   ├── 01_fit_irt.r                 # 分块 3PL IRT 参数拟合
│   ├── 02_wle_scoring.r             # 均值-标准差链接 + WLE 全量打分
│   ├── 03_atlas_cat.r               # 自适应测试（ATLAS）
│   ├── 04_pirt_accuracy.r           # p-IRT 准确率估计
│   ├── 05_compute_actual_acc.r      # 全量实际准确率（对比基准）
│   ├── submit_pipeline.sh           # SGE 集群作业提交脚本
│   └── analysis/
│       ├── compare_pirt_actual.r    # p-IRT 与实际准确率对比
│       ├── summarize_pirt_for_all_method.py   # 各基准 p-IRT MAE 汇总
│       └── summarize_theta_for_all_method.py  # θ MAE 与 WLE ground truth 对比汇总
│
├── arc/                             # 各基准结果目录（结构相同）
│   ├── irt_item_parameters_combined.csv      # 链接后的 3PL 题目参数
│   ├── irt_person_scores_WLE_SE_test.csv     # 全量 WLE 能力估计（ground truth）
│   ├── actual_accuracy.csv                   # 真实准确率（第五步）
│   ├── pirt_accuracy_se_0.1.csv              # p-IRT 准确率估计（第四步）
│   ├── pirt_accuracy_se_0.2.csv
│   ├── pirt_accuracy_se_0.3.csv
│   ├── pirt_vs_actual_se_0.1.csv             # 合并对比结果 + 误差指标
│   ├── pirt_vs_actual_se_0.2.csv
│   ├── pirt_vs_actual_se_0.3.csv
│   └── atlas_arc_random/
│       ├── irt_person_scores_ATLAS_<SE>.csv  # 自适应 θ 估计及使用题目数
│       ├── item_selection_frequency_<SE>.csv # 各题目被选中频率
│       └── selected_items.zip                # 各模型逐题选题记录（含全部 SE 阈值）
├── gsm8k/
├── hellaswag/
├── truthfulqa/
├── winogrande/
│
├── experiments/                     # 对比实验
│   ├── arc_2pl/                     # 2PL 模型消融实验
│   ├── baseline_compare/            # TinyBenchmark / MetaBench 基线对比
│   └── ...
│
├── data/                            # 响应矩阵（未提交，见环境配置）
│
├── rmse_cat_summary.csv             # 各基准 × SE 阈值的 RMSE 汇总
├── summary_pirt_mae_sd_se.csv       # 各基准 p-IRT MAE 汇总
├── summary_theta_mae_sd_se.csv      # 各基准 θ MAE 汇总
├── run_pipeline.sh                  # 全流程本地并行运行脚本
└── run_pirt_all_benchmarks.sh       # 仅运行第四至六步（ATLAS 已完成时使用）
```

---

## 环境配置

### 1. 数据

响应矩阵未提交至仓库，需手动将 CSV 文件放置到 `data/` 目录：

```
data/
├── gaussian_sampled_arc_response_matrix_test.csv
├── gaussian_sampled_gsm8k_response_matrix_test.csv
├── gaussian_sampled_hellaswag_response_matrix_test.csv
└── ...
```

### 2. R 依赖

```r
install.packages(c("mirt", "catR", "parallel", "doParallel", "foreach"))
```

推荐使用 R ≥ 4.4。

### 3. Python 依赖

```bash
pip install pandas numpy
```

---

## 运行流程

### 方式 A — 本地运行（所有基准，全并行）

```bash
bash run_pipeline.sh --benchmarks=arc,gsm8k,hellaswag,truthfulqa,winogrande \
                     --se=0.1,0.2,0.3 --n_cores=8
```

### 方式 B — SGE 集群提交（每次提交一个基准）

```bash
# 将某基准的全部步骤作为链式 SGE 作业提交
bash scripts/submit_pipeline.sh truthfulqa long

# 所有基准完成后，手动运行全局汇总：
python3 scripts/analysis/summarize_pirt_for_all_method.py
python3 scripts/analysis/summarize_theta_for_all_method.py
```

### 方式 C — 仅运行分析步骤（ATLAS 已完成时）

```bash
bash run_pirt_all_benchmarks.sh
```

---

### 分步运行

**第一步 — IRT 参数拟合**（每次处理一个 chunk，所有 chunk 并行）：

```bash
# TruthfulQA：6 个 chunk
for i in 105 209 313 418 523 628; do
  Rscript scripts/01_fit_irt.r --benchmark=truthfulqa --chunk_end=$i &
done; wait

# ARC：8 个 chunk
for i in $(seq 106 105 737) 840; do
  Rscript scripts/01_fit_irt.r --benchmark=arc --chunk_end=$i &
done; wait
```

各基准的 chunk 端点：

| 基准 | Chunk 端点 |
|------|-----------|
| ARC | `seq(106, 737, 105)` + 840 |
| GSM8K | `seq(110, 1200, 109)` + 1307 |
| HellaSwag | `seq(113, 5501, 112)` + 5608 |
| TruthfulQA | 105, 209, 313, 418, 523, 628 |
| WinoGrande | 105, 209, 313, 417, 521, 626, 731, 836, 941, 1046 |

**第二步 — WLE 全量打分**（链接各 chunk 参数，估计全量 θ）：

```bash
Rscript scripts/02_wle_scoring.r --benchmark=truthfulqa
```

**第三步 — 自适应测试**：

```bash
Rscript scripts/03_atlas_cat.r --benchmark_name=truthfulqa \
        --se_theta_stop=0.1 --n_cores=8
```

`--se_theta_stop`：停止阈值，0.1 为高精度，0.3 为快速模式。

**第四步 — p-IRT 准确率估计**：

```bash
Rscript scripts/04_pirt_accuracy.r --benchmark=truthfulqa --se_theta_stop=0.1
```

**第五步 — 实际准确率**（ground truth，用于评估对比）：

```bash
Rscript scripts/05_compute_actual_acc.r --benchmark=truthfulqa
```

**第六步 — 对比分析与汇总**：

```bash
# 单基准对比（每个 SE 阈值分别执行）
Rscript scripts/analysis/compare_pirt_actual.r \
        --benchmark=truthfulqa --se_theta_stop=0.1

# 全局 MAE 汇总（所有基准完成后执行一次）
python3 scripts/analysis/summarize_pirt_for_all_method.py
python3 scripts/analysis/summarize_theta_for_all_method.py
```

---

## 输出文件说明

### 各基准目录（`{benchmark}/`）

| 文件 | 内容 |
|------|------|
| `irt_item_parameters_combined.csv` | 链接后的 3PL 题目参数（a1, d, g, u） |
| `irt_person_scores_WLE_SE_test.csv` | 全量 WLE 能力估计 θ（ground truth） |
| `pirt_accuracy_se_<SE>.csv` | p-IRT 准确率估计（各模型） |
| `actual_accuracy.csv` | 全量题目真实准确率 |
| `pirt_vs_actual_se_<SE>.csv` | 合并对比结果，含 error、abs_error、squared_error |
| `pirt_vs_actual_se_<SE>.png` | p-IRT 与实际准确率散点图 |
| `atlas_*/irt_person_scores_ATLAS_<SE>.csv` | 自适应 θ 估计及使用题目数 |
| `atlas_*/item_selection_frequency_<SE>.csv` | 各题目被选中频率统计 |
| `atlas_*/selected_items.zip` | 各模型逐题选题记录（含全部 SE 阈值） |

### 全局汇总（repo 根目录）

| 文件 | 内容 |
|------|------|
| `rmse_cat_summary.csv` | 各基准 × SE 阈值的 RMSE 汇总 |
| `summary_pirt_mae_sd_se.csv` | 各基准 × SE 阈值的 p-IRT MAE/SD/SE |
| `summary_theta_mae_sd_se.csv` | 各基准 × 方法的 θ MAE（对比 WLE ground truth） |

---

## 可复现性说明

第三步自适应测试使用每个模型固定的随机种子，结果是确定性的。仓库中已提供预计算的选题记录（`atlas_*/selected_items.zip`），可直接用于复现论文数值，无需重跑 ATLAS。

解压并使用预计算选题记录：

```bash
# 解压某基准的选题记录
cd arc/atlas_arc_random && unzip selected_items.zip

# 然后直接运行第四至六步
Rscript scripts/04_pirt_accuracy.r --benchmark=arc --se_theta_stop=0.1
```

从头完整复现（重新运行第三步）：

```bash
Rscript scripts/03_atlas_cat.r --benchmark_name=arc --se_theta_stop=0.1 --n_cores=8
```

---

## IRT 模型说明

**三参数逻辑斯蒂（3PL）模型：**

$$P(X_{ij}=1 \mid \theta_j) = g_i + (1-g_i)\cdot\frac{1}{1+\exp(-(a_i\theta_j + d_i))}$$

- **$a_i$**：区分度参数
- **$d_i$**：难度相关参数（$b_i = -d_i/a_i$）
- **$g_i$**：猜测参数（下渐近线）

**均值-标准差链接**（将第 k 块链接到参照块）：

$$A = \frac{\text{sd}(\theta_\text{ref})}{\text{sd}(\theta_k)}, \quad B = \bar\theta_\text{ref} - A\bar\theta_k$$

$$a_i^* = a_i/A, \quad d_i^* = A\cdot d_i + B\cdot a_i$$

---

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@inproceedings{
li2026adaptive,
title={Adaptive Testing for {LLM} Evaluation: A Psychometric Alternative to Static Benchmarks},
author={Peiyu Li and Xiuxiu Tang and Si Chen and Ying Cheng and Ronald Metoyer and Ting Hua and Nitesh V Chawla},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```
