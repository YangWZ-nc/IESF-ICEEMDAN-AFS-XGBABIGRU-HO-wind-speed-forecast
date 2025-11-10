# Time Series Forecasting Framework with Signal Decomposition and Meta-heuristic Optimization

A comprehensive Python framework for time series forecasting that combines advanced signal decomposition techniques (ICEEMDAN, IESF) with machine learning models (XGBoost, BiGRU) optimized by meta-heuristic algorithms (Hippopotamus Optimization).

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Algorithm Details](#algorithm-details)
- [Configuration](#configuration)
- [Results and Outputs](#results-and-outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This framework implements a state-of-the-art approach to time series forecasting by:

1. **Two-Stage Signal Decomposition**: 
   - **Stage 1 (IESF)**: Integrated Exponential Smoothing to extract long-term trends
   - **Stage 2 (ICEEMDAN)**: Improved CEEMDAN to decompose detrended residuals into oscillatory components
2. **Feature Engineering**: Creating predictive features from decomposed components
3. **Model Training**: 
   - **XGBoost**: Trained on trend components
   - **BiGRU**: Trained on ICEEMDAN components for multi-dimensional temporal learning
4. **Hyperparameter Optimization**: Using Hippopotamus Optimization (HO) algorithm for intelligent hyperparameter tuning
5. **Multi-step Forecasting**: Predicting multiple future time steps with weighted evaluation

### Target Applications
- Wind speed forecasting
- Energy demand prediction
- Financial time series
- Any univariate time series with complex patterns

## ✨ Key Features

### Signal Processing
- **Two-Stage Decomposition Strategy**:
  - **IESF (Stage 1)**: Extracts long-term trend using optimized exponential smoothing
  - **ICEEMDAN (Stage 2)**: Decomposes detrended residuals into IMF components
- **Reproducible CEEMDAN**: Deterministic decomposition with controlled random seeding
- **Laplacian Noise**: Superior to Gaussian noise for time series stability
- **Bayesian Optimization**: Adaptive tuning of both smoothing (alpha) and decomposition (epsilon) parameters
- **Zero-Phase Trend Extraction**: Forward-backward exponential smoothing eliminates time lag

### Machine Learning Models
- **XGBoost Regressor**: Gradient boosting with GPU acceleration support
- **BiGRU Neural Network**: Bidirectional GRU for capturing temporal dependencies
- **Multi-output Support**: Native or independent model training for multi-step forecasting

### Optimization
- **Hippopotamus Optimization (HO)**: Novel meta-heuristic algorithm for hyperparameter search
- **Levy Flight**: Enhanced exploration using Lévy flight random walks
- **GPU Acceleration**: CUDA support via CuPy for faster optimization

### Evaluation
- **Custom F24 Metric**: Combines MSE, MAE, and MAPE for robust evaluation
- **Weighted Multi-step**: Different importance weights for different forecast horizons
- **Comprehensive Metrics**: RMSE, MAE, R², MAPE for each prediction step

## 📁 Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── Signal Decomposition
│   ├── ICEEMDAN.py             # Improved CEEMDAN with Bayesian optimization
│   └── IESF.py                 # Integrated Exponential Smoothing & CEEMDAN
│
├── Forecasting Models
│   ├── XGBoost.py              # XGBoost-based forecasting with HO optimization
│   └── BIGRU.py                # BiGRU neural network with HO optimization
│
├── Optimization Algorithms
│   ├── ho.py                   # Hippopotamus Optimization algorithm
│   ├── levy.py                 # Lévy flight implementation (CPU & GPU)
│   └── fun_info.py             # Benchmark functions and F24 evaluation metric
│
└── Input Data
    └── *.csv                    # Your time series CSV files
```

## 🔄 Framework Architecture

### Two Forecasting Pipelines

```
Pipeline 1: IESF → XGBoost (For Trend)
═══════════════════════════════════════════════════════════════

Input Time Series (Wind speed)
         ↓
    ┌────────────────────────────────────┐
    │  IESF.py (Two-Stage Decomposition) │
    └────────────────────────────────────┘
         ↓
    Stage 1: Exponential Smoothing
         ├─→ Long-term Trend
         └─→ Detrended Residuals
         ↓
    Stage 2: ICEEMDAN on Residuals
         ├─→ IMF_1, IMF_2, ..., IMF_N
         └─→ Final Residual
         ↓
    Output: predictive_optimized_ceemdan_decomposition_*.csv
         ├─→ Extracted_Trend
         ├─→ Residuals_After_Smoothing
         ├─→ IMF_1, IMF_2, ..., IMF_N
         └─→ CEEMDAN_Residual
         ↓
    ┌────────────────────────────────────┐
    │  XGBoost.py (Gradient Boosting)    │
    └────────────────────────────────────┘
         ├─→ Feature Engineering
         ├─→ HO Hyperparameter Optimization
         └─→ Multi-step Forecasting
         ↓
    Final Predictions with R², F24, RMSE, MAE


Pipeline 2: ICEEMDAN → BiGRU
═══════════════════════════════════════════════════════════════

Input Time Series (speed m/s)
         ↓
    ┌────────────────────────────────────┐
    │  ICEEMDAN.py (Direct Decomposition)│
    └────────────────────────────────────┘
         ↓
    Bayesian-Optimized CEEMDAN
         ├─→ IMF_1 (High frequency)
         ├─→ IMF_2
         ├─→ IMF_3
         ├─→ ...
         ├─→ IMF_N (Low frequency)
         └─→ Residual (Trend-like)
         ↓
    Output: *_CEEMDAN_Laplace_optimized_decomposition.csv
         ├─→ Original_Signal
         ├─→ IMF_1, IMF_2, ..., IMF_N
         └─→ Residual
         ↓
    ┌────────────────────────────────────┐
    │  BIGRU.py (Bidirectional GRU)      │
    └────────────────────────────────────┘
         ├─→ Multi-dimensional Input (all IMFs + Residual)
         ├─→ HO Hyperparameter Optimization
         └─→ Temporal Pattern Learning
         ↓
    Final Predictions with Weighted RMSE
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for acceleration

### Step 1: Clone or Download
```bash
# Download all project files to your working directory
```

### Step 2: Install Dependencies

#### Basic Installation (CPU only)
```bash
pip install -r requirements.txt
```

#### With GPU Support
```bash
# Check your CUDA version first
nvcc --version

# For CUDA 11.x
pip install -r requirements.txt
pip install cupy-cuda11x

# For CUDA 12.x
pip install -r requirements.txt
pip install cupy-cuda12x

# Install PyTorch with CUDA support (visit pytorch.org for exact command)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation
```python
python -c "import numpy, pandas, xgboost, torch, PyEMD; print('All packages installed successfully!')"
```

## 🚀 Quick Start

### Example 1: Two-Stage Signal Decomposition

```python
# Run IESF to extract long-term trend and perform initial decomposition
python IESF.py

# Expected output:
# - Extracted trend component
# - Detrended residuals
# - CEEMDAN decomposition of residuals
# - Complete decomposition with trend + IMFs + final residual
```

**Input Requirements:**
- CSV file with a column named `Wind speed` (for IESF.py)
- Minimum 50+ data points recommended

**What happens in IESF:**
1. Optimizes alpha parameter for exponential smoothing
2. Extracts zero-phase long-term trend
3. Computes residuals after trend removal
4. Applies ICEEMDAN to decompose residuals
5. Outputs: Trend + IMF components + final residual

### Example 2: XGBoost Forecasting (Uses IESF Output)

```python
# 1. First, run IESF decomposition
python IESF.py

# 2. Update the file path in XGBoost.py (line 1482)
# file_path = "predictive_optimized_ceemdan_decomposition_XXXXXX.csv"

# 3. Run XGBoost forecasting
python XGBoost.py

# The model will:
# - Load IESF decomposed components (trend + IMFs + residual)
# - Optimize hyperparameters using HO algorithm
# - Train XGBoost models on decomposed features
# - Generate multi-step forecasts
# - Save results to timestamped folder
```

### Example 3: BiGRU Forecasting (Uses ICEEMDAN Output)

```python
# 1. Run ICEEMDAN for direct signal decomposition
python ICEEMDAN.py

# 2. Update the CSV path in BIGRU.py (line 29)
# CSV_PATH = r"your_iceemdan_decomposition.csv"

# 3. Configure prediction parameters (lines 34-52)
# LOOK_BACK = 30          # Historical window size
# PREDICTION_STEPS = 3    # Number of steps to forecast
# USER_SEED = 12345       # Random seed for reproducibility

# 4. Run BiGRU forecasting
python BIGRU.py

# The model will:
# - Load ICEEMDAN IMF components and residuals
# - Optimize BiGRU hyperparameters using HO
# - Train bidirectional GRU networks on multi-dimensional input
# - Generate weighted multi-step forecasts
# - Save predictions to CSV
```

## 📖 Detailed Usage

### Pipeline 1: IESF → XGBoost

**Decomposition Strategy:**
- IESF performs two-stage decomposition:
  1. Extracts long-term trend using optimized exponential smoothing
  2. Applies ICEEMDAN to detrended residuals
- Output: Trend + IMF components + Final residual
- Best for: Capturing both trend and cyclical patterns

**Step 1: Prepare Your Data**
```csv
# Example: wind_speed_data.csv
Wind speed
5.2
5.8
6.1
...
```

**Step 2: Run IESF Decomposition**
```bash
python IESF.py
```

This generates:
- `predictive_optimized_ceemdan_decomposition_XXXXXX.csv` containing:
  - `Original_Series`: Original time series
  - `Extracted_Trend`: Long-term trend (from exponential smoothing)
  - `Residuals_After_Smoothing`: Signal after trend removal
  - `IMF_1, IMF_2, ..., IMF_N`: Oscillatory components (from CEEMDAN on residuals)
  - `CEEMDAN_Residual`: Final residual after all decomposition

**Step 3: Configure XGBoost**

Edit `XGBoost.py`:
```python
# Line 1482: Set your IESF decomposed file path
file_path = "predictive_alpha_optimized_results_yourfile/predictive_optimized_ceemdan_decomposition_20241028_120000.csv"

# Line 1486: Set random seed
RANDOM_SEED = 42

# Lines 1503-1512: Configure forecasting
results = main(
    file_path,
    steps=3,                    # Forecast 3 steps ahead
    weights=[5, 3, 2],         # Weight importance: step1 > step2 > step3
    seed=RANDOM_SEED,
    use_transform=True,        # Apply differencing for stationarity
    max_iterations=100,        # HO algorithm iterations
    search_agents=32           # Number of search agents in HO
)
```

**Step 4: Run XGBoost**
```bash
python XGBoost.py
```

**Expected Output:**
```
Starting HO optimization...
Iteration 1: Best value = 0.0234
Iteration 2: Best value = 0.0198
...
Iteration 100: Best value = 0.0089

Training final models with best parameters...
Training Set - Weighted F24: 0.0076, R²: 0.9812
Validation Set - Weighted F24: 0.0089, R²: 0.9745
Test Set - Weighted F24: 0.0095, R²: 0.9698

Results saved to: xgboost_forecasting_results_seed_42_20241028_143022/
Finish!
```

### Pipeline 2: ICEEMDAN → BiGRU

**Decomposition Strategy:**
- ICEEMDAN performs direct signal decomposition into IMFs
- No explicit trend extraction
- Output: Multiple IMF components + Residual
- Best for: Multi-dimensional temporal pattern learning

**Step 1: Run ICEEMDAN Decomposition**
```bash
python ICEEMDAN.py
```

This creates:
- `*_CEEMDAN_Laplace_optimized_decomposition.csv` containing:
  - `Original_Signal`: Original time series
  - `IMF_1, IMF_2, ..., IMF_N`: Intrinsic Mode Functions (oscillatory components)
  - `Residual`: Final residual (trend-like component)

**Step 2: Configure BiGRU**

Edit `BIGRU.py`:
```python
# Line 29: Set input CSV path (from ICEEMDAN)
CSV_PATH = r"CEEMDAN_Laplace_Optimized_Results_XXXXXX/wind_speed_data_CEEMDAN_Laplace_optimized_decomposition.csv"

# Line 30-31: Specify column names
IMF_PREFIX = 'IMF_'           # Prefix for IMF columns
RESIDUAL_COLUMN = 'Residual'  # Residual column name

# Lines 34-46: Configure model parameters
LOOK_BACK = 30                # Use 30 historical points
PREDICTION_STEPS = 3          # Predict 3 steps ahead
EPOCHS = 30                   # Training epochs
HO_SEARCH_AGENTS = 16        # HO population size
HO_MAX_ITERATIONS = 30       # HO iterations
STEP_WEIGHTS = [0.5, 0.3, 0.2]  # Step importance weights

# Line 52: Set your random seed
USER_SEED = 12345
```

**Step 3: Run BiGRU**
```bash
python BIGRU.py
```

**Expected Output:**
```
Loading data from CSV...
Found 6 IMF components and 1 residual
Total features: 7

Running HO optimization for hyperparameters...
HO Iteration 1/30: Best score = 2.456
HO Iteration 30/30: Best score = 1.234

Best hyperparameters found:
- hidden_size: 96
- num_layers: 2
- learning_rate: 0.0045
- dropout: 0.23
- batch_size: 32

Training final model with best parameters...
Epoch 1/30: Loss = 0.1234
Epoch 30/30: Loss = 0.0234

Final weighted RMSE: 1.234
Results saved to: BiGRU_HO_MultiDim_Sum_F24_WeightedRMSE_Results_20241028_150000/
Finish!
```

## 🧮 Algorithm Details

### 1. ICEEMDAN (Improved Complete EEMD with Adaptive Noise)

**Innovation:**
- Laplacian white noise (instead of Gaussian) for better stability
- Bayesian optimization of epsilon parameter
- Predictive quality scoring function optimized for forecasting

**Workflow:**
```
Input Signal → Reference Testing → Adaptive Search Space → 
Bayesian Optimization → CEEMDAN Decomposition → Quality Evaluation
```

**Key Parameters:**
- `trials`: Number of ensemble realizations (default: 100)
- `epsilon`: Noise amplitude (optimized automatically, typically 0.2-0.6)
- `seed`: Random seed for reproducibility

### 2. IESF (Integrated Exponential Smoothing & CEEMDAN)

**Two-Stage Decomposition Strategy:**

**Stage 1: Trend Extraction**
- Uses zero-phase exponential smoothing to extract long-term trend
- Optimizes alpha parameter via Bayesian optimization
- Forward-backward smoothing eliminates phase shift

**Stage 2: Residual Decomposition**
- Applies ICEEMDAN to detrended residuals
- Decomposes oscillatory patterns into IMF components
- Optimizes epsilon parameter for CEEMDAN

**Workflow:**
```
Input Signal → 
  ↓
Stage 1: Optimize Alpha → Extract Trend (Exponential Smoothing) →
  ↓
Compute Residuals = Original - Trend →
  ↓
Stage 2: Optimize Epsilon → ICEEMDAN Decomposition of Residuals →
  ↓
Output: Trend + IMF₁ + IMF₂ + ... + IMFₙ + Final Residual
```

**Key Features:**
- Zero-phase trend extraction (no time lag)
- Joint optimization of smoothing and decomposition parameters
- Predictive error proxy for quality assessment
- Separates long-term trend from short-term oscillations

**Why Two Stages?**
- **Better Separation**: Trend and oscillations separated explicitly
- **Improved Stationarity**: CEEMDAN works better on detrended data
- **Interpretability**: Clear distinction between trend and cyclical components
- **Forecasting Performance**: Each component type can be modeled appropriately

### 3. Hippopotamus Optimization (HO)

**Inspiration:** Mimics hippopotamus social behavior and predator escape strategies

**Three Main Operators:**
1. **Position update near dominant hippo** (exploitation)
2. **Defense mechanism** (exploration using Lévy flight)
3. **Territorial adjustment** (local search)

**Advantages:**
- Fast convergence for continuous optimization
- Good balance between exploration and exploitation
- GPU-accelerated via CuPy

**Hyperparameters Optimized:**
- **XGBoost**: `learning_rate`, `n_estimators`
- **BiGRU**: `hidden_size`, `num_layers`, `learning_rate`, `dropout`, `batch_size`

### 4. F24 Evaluation Metric

Custom metric combining three error types:

```python
F24 = (1/3N) * Σ(y_true - y_pred)² +      # MSE component
      (1/3N) * Σ|y_true - y_pred| +       # MAE component
      (1/3N) * Σ|y_true - y_pred|/|y_true| # MAPE component
```

**Weighted Multi-step:**
```python
Final_Score = w₁·F24₁ + w₂·F24₂ + w₃·F24₃
# Default weights: [5, 3, 2] (closer predictions weighted higher)
```

## ⚙️ Configuration (Only example, the configurations are not same for different seasons)

### XGBoost Configuration

**Fixed Hyperparameters** (lines 30-42 in XGBoost.py):
```python
FIXED_XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'max_depth': 8,
    'subsample': 0.60,
    'colsample_bytree': 0.99,
    'reg_alpha': 2e-07,
    'reg_lambda': 0.69,
    'min_child_weight': 17,
    'gamma': 1e-07
}
```

**Optimized by HO** (lines 46-49):
```python
HO_PARAM_BOUNDS = {
    'learning_rate': (0.01, 1),
    'n_estimators': (10, 2000)
}
```

### BiGRU Configuration

**Global Settings** (Config class, lines 26-52 in BIGRU.py):
```python
class Config:
    LOOK_BACK = 30              # Historical window size
    PREDICTION_STEPS = 3        # Forecast horizon
    EPOCHS = 30                 # Training epochs
    HO_SEARCH_AGENTS = 16      # Population size
    HO_MAX_ITERATIONS = 30     # Optimization iterations
    STEP_WEIGHTS = [0.5, 0.3, 0.2]  # Multi-step weights
    USER_SEED = 12345          # Random seed
```

**HO Search Space** (lines 469-470):
```python
lb = np.array([32, 1, 0.0001, 0.0, 16])      # Lower bounds
ub = np.array([128, 3, 0.01, 0.5, 64])       # Upper bounds
# [hidden_size, num_layers, lr, dropout, batch_size]
```

## 📊 Results and Outputs

### ICEEMDAN Outputs

```
CEEMDAN_Laplace_Optimized_Results_YYYYMMDD_HHMMSS/
├── *_CEEMDAN_Laplace_optimized_decomposition.csv
│   ├── Original_Signal
│   ├── IMF_1, IMF_2, ..., IMF_N
│   └── Residual
├── *_energy_distribution_optimized.csv
│   └── Energy percentage for each component
├── *_quality_metrics_optimized.csv
│   ├── Number of IMFs
│   ├── Optimized Epsilon
│   ├── Reconstruction Error (RRMSE)
│   ├── Orthogonality Index (OI)
│   └── Predictive Quality Score
└── processing_summary_optimized.csv
    └── Summary for all processed files
```

### IESF Outputs

```
predictive_alpha_optimized_results_[filename]/
└── predictive_optimized_ceemdan_decomposition_YYYYMMDD_HHMMSS.csv
    ├── Original_Series
    ├── Extracted_Trend              # Key column for BiGRU
    ├── Residuals_After_Smoothing
    ├── IMF_1, IMF_2, ..., IMF_N
    └── CEEMDAN_Residual
```

### XGBoost Outputs

```
xgboost_forecasting_results_seed_42_YYYYMMDD_HHMMSS/
├── logs/
│   └── training_log_42.log           # Detailed training log
├── models/
│   ├── joint_model.pkl               # Or model_step_1.pkl, model_step_2.pkl, ...
│   └── ...
├── results/
│   ├── results_summary.json          # Best params, metrics summary
│   ├── detailed_performance.json     # Train/Val/Test metrics
│   ├── complete_test_predictions.csv # Full predictions with errors
│   └── feature_importance.csv        # Feature importance ranking
├── optimization/
│   └── ho_convergence.csv            # HO algorithm convergence
└── plots/ (if generated)
    └── *.png                         # Visualization plots
```

### BiGRU Outputs

```
BiGRU_HO_MultiDim_Sum_F24_WeightedRMSE_Results_YYYYMMDD_HHMMSS/
├── complete_log.txt                           # Full execution log
├── ho_optimization_log.json                   # HO optimization results
├── stability_test_summary.json                # Final run statistics
├── deterministic_checkpoints.json             # Reproducibility checkpoints
└── final_sum_predictions_multidim_weighted_best_seed_XXXXX.csv
    ├── sum_prediction_step_1, step_2, step_3  # Predictions
    ├── sum_true_step_1, step_2, step_3        # Actual values
    ├── sum_error_step_1, step_2, step_3       # Absolute errors
    └── Metadata (seed, weights, RMSE, etc.)
```

## 🐛 Troubleshooting

### Common Issues

**1. ImportError: No module named 'PyEMD'**
```bash
pip install PyEMD
```

**2. CuPy installation fails**
```bash
# Try CPU-only mode - comment out cupy in requirements.txt
# Or ensure CUDA toolkit is installed:
# Download from: https://developer.nvidia.com/cuda-downloads
```

**3. Out of Memory (OOM) on GPU**
```python
# Reduce batch size in BIGRU.py
Config.HO_SEARCH_AGENTS = 8  # Reduce from 16
# Or in XGBoost.py
search_agents=16  # Reduce from 32
```

**4. scikit-optimize not found**
```bash
pip install scikit-optimize
# If still fails:
pip install scikit-optimize[plots]
```

**5. XGBoost doesn't use GPU**
```python
# Check XGBoost version
import xgboost as xgb
print(xgb.__version__)  # Should be >= 1.5.0

# Verify GPU support
print(xgb.get_config()['use_rmm'])
```

**6. "No CSV files found" error**
- Ensure CSV files are in the same directory as the Python scripts
- Check column names match expected format:
  - IESF.py expects: `Wind speed`
  - ICEEMDAN.py expects: `speed(m/s)`
  - XGBoost.py expects: IESF output (`predictive_optimized_ceemdan_decomposition_*.csv`)
  - BiGRU.py expects: ICEEMDAN output (`*_CEEMDAN_Laplace_optimized_decomposition.csv`)

**7. Results are not reproducible**
- Ensure you set the same random seed across runs
- Use deterministic mode (already enabled in BiGRU)
- Note: Some minor variation may occur due to floating-point arithmetic

### Performance Tips

1. **Use GPU acceleration** for faster training:
   ```python
   # XGBoost automatically detects GPU
   # For PyTorch (BiGRU):
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

2. **Adjust HO parameters** for speed vs. accuracy trade-off:
   ```python
   # Faster (less accurate)
   max_iterations=50, search_agents=16
   
   # Slower (more accurate)
   max_iterations=200, search_agents=64
   ```

3. **Use multi-core for data processing**:
   ```python
   # XGBoost
   'n_jobs': -1  # Use all CPU cores
   
   # BiGRU
   torch.set_num_threads(8)  # Adjust based on your CPU
   ```

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{time_series_forecasting_framework,
  title={Time Series Forecasting Framework with Signal Decomposition and Meta-heuristic Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/your-repo}
}
```

**Key Algorithms:**

1. Hippopotamus Optimization:
```bibtex
@article{amiri2024hippopotamus,
  title={Hippopotamus optimization algorithm: A novel nature-inspired optimization algorithm},
  author={Amiri, Mohammad H and Mehrabi Hashjin, Nastaran and Montazeri, Mohsen and Mirjalili, Seyedali and Khodadadi, Nima},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={5032},
  year={2024}
}
```

2. CEEMDAN:
```bibtex
@article{torres2011complete,
  title={A complete ensemble empirical mode decomposition with adaptive noise},
  author={Torres, Maria E and Colominas, Marcelo A and Schlotthauer, Gaston and Flandrin, Patrick},
  journal={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={4144--4147},
  year={2011}
}
```

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing issues on GitHub
3. Contact: [zhixinyangnc@yahoo.com]

---

**Note:** This framework is designed for research purposes. For production use, additional validation and testing are recommended.

**Last Updated:** October 2025
