# ATLAS: Adaptive Testing for LLM Ability Scoring

ATLAS is a framework for **adaptive testing of large language models (LLMs)** using **Item Response Theory (IRT)** and **Computerized Adaptive Testing (CAT)** to provide precise, efficient, and fair measurement of LLM ability.

---

## 📖 Overview

### Motivation

Traditional static benchmarks require administering thousands of items to each model, which is:
- **Computationally expensive**: Running models on entire test banks is resource-intensive
- **Inefficient**: Many items provide little information about model ability
- **Security concerns**: Exposing all items increases risk of contamination

ATLAS provides an **adaptive alternative** that:
- Estimates ability (θ) with **far fewer items** (typically 30-100 vs. 500-1000+)
- Selects the **most informative items** for each model dynamically
- Monitors **test security** through exposure and overlap metrics

### Key Features

**1. Partition-Based Common-Person Calibration**
- Partitions large item pools into manageable subsets (≥100 items each)
- Calibrates each partition independently using 3-Parameter Logistic (3PL) IRT models
- Links partitions to a common scale via **mean–sigma transformations**
- Reduces computational complexity from O(|I|³) to O(K · max|Iₖ|³)

**2. Adaptive Item Selection**
- **Maximum Fisher Information (MFI)** criterion for optimal precision
- **Randomesque selection**: Randomly choose from top 5 most informative items
- **Expected A Posteriori (EAP)** estimation for ability updates
- Flexible stopping rules based on Standard Error (SE) thresholds

**3. Test Security Monitoring**
- Item exposure rate analysis (fraction of models seeing each item)
- Test overlap rate calculation (similarity between model-specific tests)
- Position analysis for item selection patterns
- Comprehensive statistical reporting

**4. Supported Benchmarks**

| Benchmark | Models | Items | Avg. M2 RMSEA |
|-----------|--------|-------|---------------|
| **WinoGrande** | 4,680 | 1,046 | 0.0565 |
| **TruthfulQA** | 4,635 | 628 | 0.0690 |
| **HellaSwag** | 3,467 | 5,608 | 0.0482 |
| **GSM8K** | 4,195 | 1,306 | 0.0436 |
| **ARC** | 4,162 | 842 | 0.0588 |

**Note**: RMSEA values reflect average fit from M2 statistics under the 3PL IRT model. Items were filtered based on variability, ceiling effects, and point-biserial discrimination. Calibration and ability estimation performed using the `mirt` package in R with WLE estimator.

---

## ⚙️ Setup

### 1. Data Requirements

* Benchmark response matrices (binary: correct 1/incorrect 0) in CSV format
* Each CSV have models as rows and items as columns
* Data files are placed in `data/` directory:
  * `clean_response_matrix_arc.csv`
  * `clean_response_matrix_gsm8k.csv`
  * `clean_response_matrix_hellaswag.csv`
  * `clean_response_matrix_truthfulqa.csv`
  * `clean_response_matrix_winogrande.csv`

### 2. R Dependencies

```r
install.packages(c("mirt", "catR"))
```

- **R** (≥ 4.2 recommended)
- **mirt**: IRT model estimation and parameter extraction
- **catR**: Adaptive testing algorithms and item selection

### 3. Python Dependencies

```bash
pip install pandas numpy
```

- **Python** (≥ 3.7 recommended)
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computations for statistical analysis

### 4. Directory Structure

```
ATLAS/
├── arc/
│   ├── irt_arc.r                        # Partition-based IRT calibration
│   ├── get_theta_whole_WLE.r            # Partition linking & baseline ability
│   ├── atlas_random.r                   # Adaptive testing (ATLAS)
│   ├── irt_item_parameters_combined.csv # Linked item parameters
│   ├── irt_person_scores_WLE_SE.csv     # Baseline WLE ability estimates
│   └── atlas_arc_random/                # ATLAS outputs (created by script)
│       ├── selected_items_<SE>/         # Per-model item selections (gitignored - too large)
│       ├── irt_person_scores_ATLAS_<SE>.csv
│       ├── item_selection_frequency_<SE>.csv # Item usage statistics (gitignored - too large)
│       └── item_position_analysis_<SE>.csv
├── gsm8k/          # Same structure as arc/
├── hellaswag/      # Same structure as arc/
├── truthfulqa/     # Same structure as arc/
├── winogrande/     # Same structure as arc/
├── data/           # Response matrices (unzipped from data.zip)
│   ├── clean_response_matrix_arc.csv
│   ├── clean_response_matrix_gsm8k.csv
│   ├── clean_response_matrix_hellaswag.csv
│   ├── clean_response_matrix_truthfulqa.csv
│   └── clean_response_matrix_winogrande.csv
└── calculate_item_overlap.py  # Test overlap & exposure analysis
```

**Note on File Storage**: 
- **Kept in repository**: Item parameters, person scores, frequency/position analyses (small CSV files)
- **Gitignored**: Raw response matrices in `data/` and large ATLAS output files (too large for git)

---

## 🚀 Usage

### Step 1: Partition-Based IRT Calibration

Calibrate item parameters using 3PL IRT models on partitioned subsets. Each partition is calibrated independently:

**ARC (8 partitions):**
```bash
cd arc/
for i in $(seq 106 105 737) 842; do
   Rscript irt_arc.r $i
done
```

**GSM8K (13 partitions):**
```bash
cd gsm8k/
for i in $(seq 106 105 1319); do
   Rscript irt_gsm8k.r $i
done
```

**TruthfulQA (6 partitions):**
```bash
cd truthfulqa/
for i in 105 209 313 418 523 628; do
   Rscript irt_truthfulqa.r $i
done
```

> 💡 **Note**: Partition boundaries are predefined in each script. The scripts use the `mirt` package with EM algorithm and 3PL item type. Each partition produces:
> - `irt_item_parameters_<index>.csv`: Item parameters (a1, d, g, u)
> - `irt_person_scores_<index>.csv`: Person ability estimates for that partition
> - `m2_<index>.csv`: Model fit statistics

---

### Step 2: Link Partitions and Estimate Baseline Ability

After calibrating all partitions, link them to a common scale using mean–sigma transformations and compute baseline ability estimates:

```bash
cd <benchmark>/
Rscript get_theta_whole_WLE.r
```

**This script:**
1. Reads all partition-specific item parameters and person scores
2. Links partitions to a common scale using the first partition as reference
3. Applies mean–sigma linking: A = sd(ref)/sd(partition), B = mean(ref) - A×mean(partition)
4. Transforms item parameters: a₁* = a₁/A, d* = A×d + B×a₁
5. Fits a fixed-parameter IRT model to the full item set
6. Estimates ability (θ) for all models using **Weighted Likelihood Estimation (WLE)**
7. Outputs `irt_item_parameters_combined.csv` and `irt_person_scores_WLE_SE.csv`

---

### Step 3: Run Adaptive Testing (ATLAS)

Execute the adaptive testing algorithm with customizable stopping rules:

```bash
cd <benchmark>/
Rscript atlas_random.r --se_theta_stop=0.1 --test_length=100
```

**Parameters:**
- `--se_theta_stop`: Standard error threshold for stopping (default: 0.2)
  - Lower values = more precision but more items needed
  - Typical values: 0.1 (high precision), 0.2 (moderate), 0.3 (fast)
- `--test_length`: Number of models to test (default: all models in dataset)
  - Use a smaller value for testing/development

**Algorithm Details:**
1. **Initialization**: Start each model at θ = 0
2. **First Item**: Randomly select from items with difficulty near θ = 0 (within ±0.5)
3. **Subsequent Items**: 
   - **Criterion**: Maximum Fisher Information (MFI) at current θ estimate
   - **Selection**: Randomesque (randomly pick from top 5 most informative items)
   - **Estimation**: EAP (Expected A Posteriori) for θ updates
4. **Stopping Rule**: Stop when (items ≥ 30) AND (SE(θ) ≤ threshold)
5. **Maximum Items**: 500 items per model (safety limit)

**Outputs** (saved to `atlas_<benchmark>_random/`):
- `irt_person_scores_ATLAS_<SE>.csv`: Final ability estimates, item counts, timing
- `selected_items_<SE>/<model>_items.csv`: Item sequence for each model
- `item_selection_frequency_<SE>.csv`: Item usage statistics across all models
- `item_position_analysis_<SE>.csv`: Position statistics for each item

---

### Step 4: Calculate Test Overlap and Exposure Metrics

After running adaptive testing, analyze test security using the Python script:

```bash
python calculate_item_overlap.py --data_path=<benchmark>/atlas_<benchmark>_random --se_theta_stop=<SE> --benchmark=<benchmark>
```

**Parameters:**
- `--data_path`: Path to directory containing ATLAS outputs (default: `winogrande/atlas_winogrande_random`)
- `--se_theta_stop`: SE threshold used in ATLAS run (default: `0.3`)
- `--benchmark`: Benchmark name for output file prefix (default: `winogrande`)

**Example:**
```bash
python calculate_item_overlap.py --data_path=arc/atlas_arc_random --se_theta_stop=0.1 --benchmark=arc
```

**What it does:**
1. Reads individual item selection files from `selected_items_<SE>/` directory
2. Reads pre-computed frequency file `item_selection_frequency_<SE>.csv`
3. Calculates **test overlap rate**: T̄ = (N × Σ P(Aⱼ)²) / (L̄ × (N-1)) - 1/(N-1)
4. Computes **item exposure rates**: P(Aⱼ) = hⱼ/N
5. Validates results using both calculation methods (should match)
6. Generates summary statistics for exposure rates

**Outputs:**
- `<benchmark>_<SE>_test_overlap_results.csv`: Test overlap rate and exposure statistics
- `<benchmark>_<SE>_item_exposure_rates.csv`: Exposure rate for each item

---

## 📈 Analysis & Metrics

ATLAS automatically computes comprehensive metrics after adaptive testing:

### 1. Item Exposure Rate

**Definition**: Fraction of models (examinees) that were administered a given item

**Formula** (implemented in `calculate_item_overlap.py`):
```
P(Aⱼ) = hⱼ / N
```
Where:
- hⱼ = number of times item j was administered
- N = total number of examinees (models)

**Interpretation:**
- High exposure (>0.5): Item is overused, potential security risk
- Low exposure (<0.05): Item is rarely informative
- Target range: 0.10-0.40 for balanced test security

### 2. Test Overlap Rate

**Definition**: Average overlap across all test administrations, measuring how similar tests are

**Formula** (implemented in `calculate_item_overlap.py`):
```
T̄ = (N × Σ P(Aⱼ)²) / (L̄ × (N-1)) - 1/(N-1)
```
Where:
- N = total number of examinees (models)
- P(Aⱼ) = exposure rate of item j
- L̄ = mean test length across all examinees

**Interpretation:**
- 0 = completely different item sets (no overlap)
- 1 = identical item sets (complete overlap)
- Target range: 0.15-0.30 balances adaptivity and comparability

**Implementation**: Both the R script's frequency file and the Python script calculate this metric. The Python script provides two independent calculation methods that validate each other.

### 3. Position Analysis

Tracks at which position each item typically appears in the test sequence:
- **Mean position**: Average administration order
- **Position variance**: Consistency of selection timing
- **First-item frequency**: How often an item appears first

### 4. Ability Estimation Quality

**Metrics compared:**
- **θ_WLE**: Baseline estimate using all items
- **θ_ATLAS**: Adaptive estimate using selected items
- **RMSE**: Precision of adaptive estimates
- **Efficiency**: # items used in adaptive vs. full test

---

## 📊 Output Files

### Per-Benchmark Directory Structure

**Calibration Outputs:**
- `irt_item_parameters_<index>.csv`: Raw partition-specific parameters
- `irt_item_parameters_combined.csv`: Linked parameters on common scale
- `irt_person_scores_WLE_SE.csv`: Baseline ability estimates
  - Columns: `Model_Name`, `Theta_WLE`, `SE`

**Adaptive Testing Outputs (from R scripts):**
- `irt_person_scores_ATLAS_<SE>.csv`: Adaptive ability estimates
  - Columns: `Model_Name`, `Theta_ATLAS`, `Theta_WLE`, `SE`, `Num_Items`, `Time_Taken_Sec`
- `selected_items_<SE>/<model>_items.csv`: Per-model item selection sequence
  - Columns: `item_id`, `order`, `score`, `theta`, `se`
- `item_selection_frequency_<SE>.csv`: Item usage statistics
  - Columns: `item_id`, `frequency`, `percentage`
- `item_position_analysis_<SE>.csv`: Position statistics for each item
  - Columns: `item_id`, `frequency`, `mean_position`, `min_position`, `max_position`, `sd_position`

**Test Security Analysis Outputs (from Python script):**
- `<benchmark>_<SE>_test_overlap_results.csv`: Overall test overlap rate and summary
  - Columns: `N`, `overlap_rate_method1`, `overlap_rate_method2`, `mean_exposure_rate`, `std_exposure_rate`
- `<benchmark>_<SE>_item_exposure_rates.csv`: Item-level exposure rates
  - Columns: `item_id`, `exposure_rate`

---

## 🔬 Technical Details

### IRT Model Specification

**3-Parameter Logistic (3PL) Model:**

```
P(Xᵢⱼ = 1 | θⱼ, aᵢ, dᵢ, gᵢ) = gᵢ + (uᵢ - gᵢ) × [1 + exp(-(aᵢθⱼ + dᵢ))]⁻¹
```

Where:
- **aᵢ** (a1): Discrimination parameter
- **dᵢ** (d): Easiness parameter (related to difficulty: bᵢ = -dᵢ/aᵢ)
- **gᵢ** (g): Lower asymptote (guessing parameter)
- **uᵢ** (u): Upper asymptote (typically 1.0)

### Mean-Sigma Linking Transformation

**Linking Constants:**
```
A = sd(θ_reference) / sd(θ_partition)
B = mean(θ_reference) - A × mean(θ_partition)
```

**Parameter Transformation:**
```
a₁* = a₁ / A
d* = A × d + B × a₁
g* = g  (unchanged)
u* = u  (unchanged)
```

### Ability Estimation Methods

1. **WLE (Weighted Likelihood Estimation)**: 
   - Used for baseline full-test estimates
   - More robust to extreme response patterns
   - Provides finite estimates for perfect scores

2. **EAP (Expected A Posteriori)**:
   - Used during adaptive testing
   - Incorporates prior distribution (N(0,1))
   - More stable with few items
   - Computed with 61 quadrature points

### Item Selection Algorithm

**Criterion**: Maximum Fisher Information (MFI)
```
I(θ | item) = [P'(θ)]² / [P(θ)(1 - P(θ))]
```

**Randomesque Selection**:
- Identify 5 items with highest information at current θ estimate
- Randomly select one from this set
- Reduces item overexposure while maintaining efficiency

---

## 📝 Example Workflow

**Complete workflow for ARC benchmark with SE ≤ 0.1:**

```bash
# 1. Navigate to benchmark directory
cd arc/

# 2. Run partition-based calibration (all 8 partitions)
for i in $(seq 106 105 737) 842; do
   Rscript irt_arc.r $i
done

# 3. Link partitions and compute baseline ability estimates
Rscript get_theta_whole_WLE.r

# 4. Run adaptive testing with SE ≤ 0.1 stopping rule (all models)
Rscript atlas_random.r --se_theta_stop=0.1

# 5. Calculate test overlap and exposure rates
cd ..
python calculate_item_overlap.py --data_path=arc/atlas_arc_random --se_theta_stop=0.1 --benchmark=arc

# 6. Examine outputs
head arc/atlas_arc_random/irt_person_scores_ATLAS_0.1.csv
head arc_0.1_test_overlap_results.csv
head arc_0.1_item_exposure_rates.csv
```

**Quick test run (first 10 models only):**

```bash
cd winogrande/
Rscript atlas_random.r --se_theta_stop=0.3 --test_length=10
cd ..
python calculate_item_overlap.py --data_path=winogrande/atlas_winogrande_random --se_theta_stop=0.3 --benchmark=winogrande
```