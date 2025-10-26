import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from packaging import version
from statsmodels.tsa.stattools import adfuller
import warnings
import random
import os
import json
import pickle
import logging
from datetime import datetime
import shutil

warnings.filterwarnings('ignore')

# Import F24 evaluation function
from fun_info import F24

# Import HO (Hippopotamus Optimization) algorithm
from ho import HO

# Global configuration: Fixed XGBoost hyperparameters
# These parameters are kept constant during optimization
FIXED_XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'max_depth': 8,
    'subsample': 0.6018511082680795,
    'colsample_bytree': 0.9991332660884722,
    'colsample_bylevel': 0.9899263233933308,
    'reg_alpha': 2.0091241453173424e-07,
    'reg_lambda': 0.6986580166614497,
    'min_child_weight': 17,
    'gamma': 1.0420513361631241e-07,
    'verbosity': 0
}

# Parameter bounds for HO algorithm optimization
# Only learning_rate and n_estimators are optimized
HO_PARAM_BOUNDS = {
    'learning_rate': (0.01, 1),
    'n_estimators': (10, 2000)
}


def setup_logging_and_output(seed=42, base_dir="xgboost_forecasting_results"):
    """
    Set up comprehensive logging system and output folder structure.
    
    Args:
        seed: Random seed for reproducibility
        base_dir: Base directory name for output files
        
    Returns:
        output_dir: Path to output directory
        subdirs: Dictionary containing subdirectory paths
        logger: Configured logger object
    """
    # Create timestamped result folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_seed_{seed}_{timestamp}"

    # Define subdirectory structure for organizing outputs
    subdirs = {
        'logs': os.path.join(output_dir, 'logs'),
        'models': os.path.join(output_dir, 'models'),
        'results': os.path.join(output_dir, 'results'),
        'data': os.path.join(output_dir, 'data'),
        'plots': os.path.join(output_dir, 'plots'),
        'optimization': os.path.join(output_dir, 'optimization')
    }

    # Create all subdirectories
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)

    # Set up log file path
    log_file = os.path.join(subdirs['logs'], f'training_log_{seed}.log')

    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  
        ]
    )

    logger = logging.getLogger('xgboost_forecasting')

    return output_dir, subdirs, logger


def save_comprehensive_results(results, output_dir, subdirs, logger, file_path, X_test, y_test):
    """
    Save all results to files, including complete data comparison.
    
    Args:
        results: Dictionary containing all training results
        output_dir: Output directory path
        subdirs: Dictionary of subdirectory paths
        logger: Logger object
        file_path: Original data file path
        X_test: Test features
        y_test: Test targets
    """

    # 1. Save trained models
    models_dir = subdirs['models']

    if len(results['models']) == 1:
        # Joint model (single model for all outputs)
        model_path = os.path.join(models_dir, 'joint_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(results['models'][0], f)
    else:
        # Independent models (separate model for each step)
        for i, model in enumerate(results['models']):
            model_path = os.path.join(models_dir, f'model_step_{i + 1}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

    # 2. Save detailed results
    results_dir = subdirs['results']

    # Save basic results summary
    summary_path = os.path.join(results_dir, 'results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results['results_summary'], f, indent=2, ensure_ascii=False)

    # Save detailed performance metrics
    performance_data = {}
    for dataset_name in ['train_results', 'val_results', 'test_results']:
        if dataset_name in results:
            dataset_results = results[dataset_name]
            performance_data[dataset_name] = {}

            for key, value in dataset_results.items():
                if isinstance(value, (int, float, str, bool, list)):
                    performance_data[dataset_name][key] = value
                elif isinstance(value, np.ndarray):
                    performance_data[dataset_name][key] = value.tolist()

    performance_path = os.path.join(results_dir, 'detailed_performance.json')
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)

    # 3. Save complete prediction results CSV (including original data)
    if 'test_results' in results:
        # Build complete test set dataframe
        complete_test_df = pd.DataFrame()

        # Add sample indices
        complete_test_df['sample_id'] = range(len(X_test))

        # Add key features (first few most important features)
        feature_cols = X_test.columns.tolist()
        important_features = feature_cols[:min(10, len(feature_cols))]  

        for feature in important_features:
            complete_test_df[f'feature_{feature}'] = X_test[feature].values

        # Add true values
        for step in range(results['steps']):
            complete_test_df[f'true_step_{step + 1}'] = y_test.iloc[:, step].values

        # Add predictions (transformed domain)
        test_predictions = []
        for step in range(results['steps']):
            step_key = f"step_{step + 1}"
            if step_key in results['test_results']:
                predictions = results['test_results'][step_key]['predictions']
                complete_test_df[f'pred_transformed_step_{step + 1}'] = predictions
                test_predictions.append(predictions)

        # Add predictions (original domain, if available)
        if 'y_pred_original' in results['test_results']:
            y_pred_original = results['test_results']['y_pred_original']
            y_true_original = results['test_results']['y_true_original']

            for step in range(results['steps']):
                complete_test_df[f'true_original_step_{step + 1}'] = y_true_original[:, step]
                complete_test_df[f'pred_original_step_{step + 1}'] = y_pred_original[:, step]

        # Calculate errors
        for step in range(results['steps']):
            if f'true_step_{step + 1}' in complete_test_df and f'pred_transformed_step_{step + 1}' in complete_test_df:
                true_vals = complete_test_df[f'true_step_{step + 1}']
                pred_vals = complete_test_df[f'pred_transformed_step_{step + 1}']
                complete_test_df[f'error_transformed_step_{step + 1}'] = pred_vals - true_vals
                complete_test_df[f'abs_error_transformed_step_{step + 1}'] = np.abs(pred_vals - true_vals)
                complete_test_df[f'pct_error_transformed_step_{step + 1}'] = ((pred_vals - true_vals) / (
                            true_vals + 1e-8)) * 100

            # Original domain errors (if available)
            if f'true_original_step_{step + 1}' in complete_test_df and f'pred_original_step_{step + 1}' in complete_test_df:
                true_orig = complete_test_df[f'true_original_step_{step + 1}']
                pred_orig = complete_test_df[f'pred_original_step_{step + 1}']
                complete_test_df[f'error_original_step_{step + 1}'] = pred_orig - true_orig
                complete_test_df[f'abs_error_original_step_{step + 1}'] = np.abs(pred_orig - true_orig)
                complete_test_df[f'pct_error_original_step_{step + 1}'] = ((pred_orig - true_orig) / (
                            true_orig + 1e-8)) * 100

        # Save complete test results
        complete_path = os.path.join(results_dir, 'complete_test_results.csv')
        complete_test_df.to_csv(complete_path, index=False)

        # Save simplified prediction results
        if test_predictions:
            simple_pred_df = pd.DataFrame(
                np.column_stack(test_predictions),
                columns=[f'pred_step_{i + 1}' for i in range(len(test_predictions))]
            )
            simple_pred_df.index.name = 'sample_id'

            simple_path = os.path.join(results_dir, 'test_predictions_simple.csv')
            simple_pred_df.to_csv(simple_path)

        # Create performance summary table
        performance_summary = pd.DataFrame()
        performance_rows = []

        for step in range(results['steps']):
            step_key = f"step_{step + 1}"
            if step_key in results['test_results']:
                step_results = results['test_results'][step_key]
                row = {
                    'step': step + 1,
                    'weight': results['weights'][step],
                    'rmse': step_results['rmse'],
                    'mae': step_results['mae'],
                    'r2': step_results['r2'],
                    'mape': step_results['mape'],
                    'f24': step_results['f24'],
                    'domain': step_results['domain']
                }
                performance_rows.append(row)

                # Add original domain results if available
                step_key_orig = f"step_{step + 1}_original"
                if step_key_orig in results['test_results']:
                    orig_results = results['test_results'][step_key_orig]
                    row_orig = {
                        'step': step + 1,
                        'weight': results['weights'][step],
                        'rmse': orig_results['rmse'],
                        'mae': orig_results['mae'],
                        'r2': orig_results['r2'],
                        'mape': orig_results['mape'],
                        'f24': orig_results['f24'],
                        'domain': 'original'
                    }
                    performance_rows.append(row_orig)

        if performance_rows:
            performance_summary = pd.DataFrame(performance_rows)
            perf_summary_path = os.path.join(results_dir, 'performance_summary.csv')
            performance_summary.to_csv(perf_summary_path, index=False)

    if 'feature_importance' in results and not results['feature_importance'].empty:
        fi_path = os.path.join(results_dir, 'feature_importance.csv')
        results['feature_importance'].to_csv(fi_path, index=False)

    if 'ho_convergence' in results:
        optim_dir = subdirs['optimization']

        convergence_df = pd.DataFrame({
            'iteration': range(1, len(results['ho_convergence']) + 1),
            'best_f24_score': results['ho_convergence']
        })
        convergence_path = os.path.join(optim_dir, 'ho_convergence.csv')
        convergence_df.to_csv(convergence_path, index=False)

        best_params_path = os.path.join(optim_dir, 'ho_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(results['best_params'], f, indent=2)


    data_dir = subdirs['data']

    data_info = {
        'source_file': file_path,
        'transform_operations': results.get('transform_ops', []),
        'data_shapes': {
            'train_samples': len(results.get('X_train', [])),
            'val_samples': len(results.get('X_val', [])),
            'test_samples': len(results.get('X_test', [])),
            'features': len(results.get('final_features', [])),
            'prediction_steps': results['steps']
        },
        'selected_lag_features': results['results_summary'].get('selected_lag_features', []),
        'total_features_used': results['results_summary'].get('total_features_used', 0)
    }

    data_info_path = os.path.join(data_dir, 'data_info.json')
    with open(data_info_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)


    report_lines = [
        "# XGBoost Multi-Step Time Series Forecasting - Run Report",
        f"",
        f"## Basic Information",
        f"- Run Time: {datetime.now()}",
        f"- Random Seed: {results['seed']}",
        f"- XGBoost Version: {results['results_summary']['xgboost_version']}",
        f"- Prediction Steps: {results['steps']}",
        f"- Step Weights: {results['weights']}",
        f"- Optimization Algorithm: HO (Hippopotamus Optimization)",
        f"",
        f"## Data Information",
        f"- Source File: {file_path}",
        f"- Training Samples: {data_info['data_shapes']['train_samples']}",
        f"- Validation Samples: {data_info['data_shapes']['val_samples']}",
        f"- Test Samples: {data_info['data_shapes']['test_samples']}",
        f"- Features Used: {data_info['total_features_used']}",
        f"- Selected Lag Features: {len(data_info['selected_lag_features'])} ",
        f"",
        f"## Model Configuration",
        f"- Training Mode: {'Joint Model' if len(results['models']) == 1 else 'Independent Models'}",
        f"- Differencing Transform: {'Enabled' if results['use_transform'] else 'Disabled'}",
        f"- Transform Operations: {results.get('transform_ops', 'None')}",
        f"",
        f"## Performance Results",
    ]

    if 'test_results' in results:
        main_score = results['test_results']['weighted_avg_f24']
        report_lines.append(f"- Test Set Weighted F24 Score: {main_score:.4f}")

        for step in range(results['steps']):
            step_key = f"step_{step + 1}"
            if step_key in results['test_results']:
                step_results = results['test_results'][step_key]
                report_lines.append(f"- Step {step + 1}: F24={step_results['f24']:.4f}, R²={step_results['r2']:.4f}")

    report_lines.extend([
        f"",
        f"## Output Files Description",
        f"### Main Result Files",
        f"- `complete_test_results.csv`: Complete Test Set results including features, true values, predictions, and errors",
        f"- `performance_summary.csv`: Performance metrics summary for each step",
        f"- `test_predictions_simple.csv`: Simplified prediction results (predictions only)",
        f"- `ho_convergence.csv`: HO algorithm convergence history",
        f"",
        f"### Complete Test Results File Column Descriptions",
        f"- `sample_id`: Sample ID",
        f"- `feature_*`: Key input features",
        f"- `true_step_*`: Transformed DomainTrue Values",
        f"- `pred_transformed_step_*`: Transformed DomainPredicted Values",
        f"- `true_original_step_*`: Original DomainTrue Values（if availableDifferencing Transform）",
        f"- `pred_original_step_*`: Original DomainPredicted Values（if availableDifferencing Transform）",
        f"- `error_*_step_*`: Prediction error (predicted - true)",
        f"- `abs_error_*_step_*`: Absolute error",
        f"- `pct_error_*_step_*`: Percentage error (%)",
        f"",
        f"### Other Files",
        f"- Model files: models/",
        f"- Feature importance: results/feature_importance.csv",
        f"- Optimization history: optimization/",
        f"- Log files: logs/",
        f"",
        f"## Reproducibility",
        f"Using the same seed {results['seed']} can fully reproduce these results。",
        f"",
        f"Generated at: {datetime.now()}"
    ])

    report_path = os.path.join(output_dir, 'README.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


    seed_value = results['seed']
    log_file_path = os.path.join(subdirs['logs'], f'training_log_{seed_value}.log')

    return output_dir


def check_f24_function():
    """Check if F24 evaluation function is working properly"""

    test_y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    try:
        if hasattr(F24, '__code__'):
            arg_count = F24.__code__.co_argcount
            arg_names = F24.__code__.co_varnames[:arg_count]

        try:
            result = F24(None, test_y_true, test_y_pred)
        except Exception as e:

    except Exception as e:



def set_random_seeds(seed=42):
    """
    Set random seeds for all relevant libraries to ensure full reproducibility.
    
    Args:
        seed (int): Random seed value
        
    Returns:
        int: The seed value that was set
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set Python hash seed for reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set XGBoost random state
    os.environ['XGB_RANDOM_STATE'] = str(seed)

    # Set CuPy random seed if available (for GPU operations)
    try:
        import cupy
        cupy.random.seed(seed)
        os.environ['CUPY_SEED'] = str(seed)
    except ImportError:
        pass


    return seed


def pick_tree_method(use_multioutput=False):
    """
    Select appropriate tree method based on model configuration.
    
    Args:
        use_multioutput (bool): Whether using multi-output regression
        
    Returns:
        str: Tree method name ('hist' or 'gpu_hist')
    """
    # Multi-output mode only supports 'hist' tree method
    if use_multioutput:
        return 'hist'

    # Try to use GPU if available, otherwise fallback to CPU
    try:
        import cupy  
        return 'gpu_hist'
    except Exception:
        return 'hist'


def make_multioutput_model(seed=42, use_multioutput=True, learning_rate=0.03, n_estimators=500):
    version_info = get_xgb_version_info()

    tree_method = pick_tree_method(use_multioutput and version_info['supports_multi_strategy'])

    params = FIXED_XGBOOST_PARAMS.copy()
    params.update({
        'tree_method': tree_method,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'random_state': seed
    })

    if use_multioutput and version_info['supports_multi_strategy'] and tree_method == 'hist':
        try:
            import cupy
        except ImportError:
            pass

    if use_multioutput and version_info['supports_multi_strategy']:
        params['multi_strategy'] = 'multi_output_tree'  
        base = xgb.XGBRegressor(**params)
        wrapper = base  
        native_multioutput = True
    else:
        base = xgb.XGBRegressor(**params)
        wrapper = MultiOutputRegressor(base)
        native_multioutput = False

    return wrapper, native_multioutput


def create_all_lag_features(df, target_col='Extracted_Trend', max_lag=8):
    """
    Create all possible lag features, let Optimization Algorithm select optimal combination
    """

    df = df.copy()
    lag_columns = []

    for lag in range(1, max_lag + 1):
        lag_col = f'{target_col}_lag_{lag}'
        df[lag_col] = df[target_col].shift(lag)
        lag_columns.append(lag_col)


    return df, lag_columns


def get_lag_features_from_ho_params(ho_params, lag_columns, max_selected=6):
    """
    Extract lag feature selection from HO algorithm optimized parameters
    ho_params: HO algorithm optimized parameter vector
    lag_columns: List of all available lag features
    """
    selected_features = [col for col in lag_columns if '_lag_1' in col]

    other_lag_features = [col for col in lag_columns if '_lag_1' not in col]

    lag_selection_params = ho_params[2:]

    for i, col in enumerate(other_lag_features):
        if i < len(lag_selection_params):
            if lag_selection_params[i] > 0.5:
                selected_features.append(col)

    if len(selected_features) > max_selected:
        lag_1_features = [col for col in selected_features if '_lag_1' in col]
        other_selected = [col for col in selected_features if '_lag_1' not in col]

        other_selected_sorted = sorted(other_selected,
                                       key=lambda x: int(x.split('_lag_')[1]) if '_lag_' in x else 999)

        n_others_to_keep = max_selected - len(lag_1_features)
        selected_features = lag_1_features + other_selected_sorted[:n_others_to_keep]

    return selected_features


def add_daily_cycle_features(df, ts_col=None, freq_per_day=96, target_col='Extracted_Trend', max_lag=8):
    """Construct daily cycle features and all possible lag features at 15min granularity"""
    df = df.copy()

    if ts_col and ts_col in df.columns:
        t = pd.to_datetime(df[ts_col])
        slot = (t.view('int64') // (15 * 60 * 1_000_000_000)) % freq_per_day  
    else:
        slot = np.arange(len(df)) % freq_per_day

    df['slot_in_day'] = slot
    df['sin_d'] = np.sin(2 * np.pi * slot / freq_per_day)
    df['cos_d'] = np.cos(2 * np.pi * slot / freq_per_day)

    week_p = freq_per_day * 7
    df['sin_w'] = np.sin(2 * np.pi * np.arange(len(df)) / week_p)
    df['cos_w'] = np.cos(2 * np.pi * np.arange(len(df)) / week_p)

    df, lag_columns = create_all_lag_features(df, target_col, max_lag)

    for w in [3, 6, 12]:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].rolling(w).mean()
        df[f'{target_col}_roll_std_{w}'] = df[target_col].rolling(w).std()

    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_2'] = df[target_col].diff(2)

    df[f'{target_col}_ewm_short'] = df[target_col].ewm(span=3).mean()
    df[f'{target_col}_ewm_long'] = df[target_col].ewm(span=7).mean()

    df.attrs['lag_columns'] = lag_columns

    return df


def need_diff_adf(y, p_th=0.05):
    try:
        p = adfuller(np.asarray(y).astype(float), autolag='AIC')[1]
        return p > p_th  
    except Exception:
        y = np.asarray(y).astype(float)
        trend = np.polyfit(np.arange(len(y)), y, 1)[0]
        return abs(trend) > (y.std() + 1e-6) * 0.02


def apply_transform(y, season=96):
    """Returns y_tr, inverse_fn, flag describing the applied transformation"""
    y = pd.Series(y).astype(float)
    use_diff1 = need_diff_adf(y)
    use_seas = (season is not None) and (season > 1) and (y.autocorr(lag=season) or 0) > 0.3

    y_tr = y.copy()
    ops = []
    if use_seas:
        y_tr = y_tr - y_tr.shift(season)
        ops.append(('seas', season))
    if use_diff1:
        y_tr = y_tr.diff()
        ops.append(('diff', 1))

    y_tr = y_tr.dropna()

    def invert(pred_steps, last_vals):
        """
        pred_steps: Predicted "Transformed Domain" values, length=H
        last_vals: Last "Original Domain" values needed for inverse transform (list/ndarray)
        """
        pred_steps = list(map(float, pred_steps))
        hist = list(map(float, last_vals))
        out = []
        for i, p in enumerate(pred_steps):
            val = p
            cur_idx = len(hist) + i
            cur = p

            if ('diff', 1) in ops:
                base = hist[-1] if (len(hist) > 0) else 0.0
                cur = base + cur
                hist.append(cur)

            if ('seas', season) in ops:
                idx = cur_idx - season
                base = hist[idx] if idx >= 0 else hist[-season]
                cur = base + (cur if ('diff', 1) in ops else p)
                if ('diff', 1) in ops:
                    hist[-1] = cur
                else:
                    hist.append(cur)

            if ('diff', 1) not in ops and ('seas', season) not in ops:
                hist.append(cur)

            out.append(hist[-1])
        return np.array(out)

    return y_tr.values, invert, ops


def load_data(file_path):
    """Load CSV data"""
    df = pd.read_csv(file_path)


    return df


def create_multistep_targets(df, target_col='Extracted_Trend', steps=3):
    """Create target variables for multi-step prediction"""
    df = df.copy()

    for i in range(1, steps + 1):
        df[f'{target_col}_step_{i}'] = df[target_col].shift(-i)

    df = df.dropna()

    return df


def invert_direct_block(y_pred_tr, y_orig, start_idx, steps, ops, season=96):
    """
    y_pred_tr: (m, steps) Transformed Domain predictions (corresponding to y_tr[t+1..t+steps]）
    y_orig   : Original complete sequence
    start_idx: Test Set starting position in original sequence (= len(train)+len(val)）
    Returns: Original Domain predictions (m, steps)
    """
    m = y_pred_tr.shape[0]
    out = np.zeros_like(y_pred_tr, dtype=float)

    has_seas = ('seas', season) in ops
    has_diff = ('diff', 1) in ops

    for i in range(m):
        t = start_idx + i  
        if has_seas:
            if t - season >= 0:
                prev_seas = y_orig[t] - y_orig[t - season]
            else:
                prev_seas = y_orig[t]  
        else:
            prev_seas = y_orig[t]

        for h in range(steps):
            val_tr = y_pred_tr[i, h]  
            if has_diff:
                y_seas_cur = prev_seas + val_tr
            else:
                y_seas_cur = val_tr
            if has_seas:
                if t + h + 1 - season >= 0:
                    base = y_orig[t + h + 1 - season]
                    y_cur = base + y_seas_cur
                else:
                    y_cur = y_seas_cur  
            else:
                y_cur = y_seas_cur

            out[i, h] = y_cur
            prev_seas = y_seas_cur  
    return out


def split_data(df, target_col='Extracted_Trend', steps=3, train_ratio=0.6, val_ratio=0.2, use_transform=False):
    """Split data in temporal order, supporting multi-step prediction and optional Differencing Transform"""

    df_original = df.copy()  

    df_features = add_daily_cycle_features(df, target_col=target_col, max_lag=8)

    if use_transform:
        y_original = df_features[target_col].values
        y_transformed, inverse_fn, ops = apply_transform(y_original, season=96)

        valid_idx = len(y_original) - len(y_transformed)
        df_features = df_features.iloc[valid_idx:].copy()
        df_features[target_col] = y_transformed

        df_original = df_original.iloc[valid_idx:].copy()
    else:
        inverse_fn = None
        ops = []

    df_final = create_multistep_targets(df_features, target_col, steps)

    final_valid_rows = len(df_final)
    df_original = df_original.iloc[:final_valid_rows].copy()

    lag_columns = df_features.attrs.get('lag_columns', [])

    n = len(df_final)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    target_cols = [f'{target_col}_step_{i}' for i in range(1, steps + 1)]
    feature_cols = [col for col in df_final.columns if col not in target_cols]

    X = df_final[feature_cols]
    y = df_final[target_cols]  

    X.attrs['lag_columns'] = lag_columns

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    for dataset in [X_train, X_val, X_test]:
        dataset.attrs['lag_columns'] = lag_columns

    df_orig_train = df_original.iloc[:train_end]
    df_orig_val = df_original.iloc[train_end:val_end]
    df_orig_test = df_original.iloc[val_end:]


    return (X_train, X_val, X_test, y_train, y_val, y_test,
            df_original, ops, train_end, val_end)


def get_xgb_version_info():
    """Get XGBoost Version information and return compatibility flags"""
    version_str = xgb.__version__
    version_tuple = tuple(map(int, version_str.split('.')[:2]))

    supports_callbacks = version_tuple >= (1, 6)
    supports_multi_strategy = version_tuple >= (2, 0)

    return {
        'version': version_str,
        'supports_callbacks': supports_callbacks,
        'supports_multi_strategy': supports_multi_strategy
    }


def fit_xgb_model_compatible(model, X_train, y_train, X_val, y_val, early_stopping_rounds=50, verbose=False):
    """Training function compatible with different XGBoost Versions"""
    version_info = get_xgb_version_info()

    try:
        if version_info['supports_callbacks']:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
                verbose=verbose
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
    except Exception as e:
        model.fit(X_train, y_train, verbose=verbose)

    return model


from sklearn.base import clone


def fit_multioutput_regressor_with_es(wrapper, Xtr, Ytr, Xva, Yva, esr=200):
    """Add early stopping for each sub-model in MultiOutputRegressor"""
    ests = []
    base = wrapper.estimator
    for j in range(Ytr.shape[1]):
        est = clone(base)
        fit_xgb_model_compatible(est, Xtr, Ytr.iloc[:, j], Xva, Yva.iloc[:, j],
                                 early_stopping_rounds=esr, verbose=False)
        ests.append(est)
    wrapper.estimators_ = ests
    return wrapper


def fit_multioutput_with_cv(X, Y, steps=3, seed=42, gap=8, n_splits=5, esr=200, use_multioutput=True):
    """
    X: Features DataFrame
    Y: Target of shape (n, steps) (processed according to whether differencing is applied)
    Returns: Fitted model (single model)
    """
    model, native = make_multioutput_model(seed=seed, use_multioutput=use_multioutput)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    tr_idx, va_idx = list(tscv.split(X))[-1]
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    Ytr, Yva = Y.iloc[tr_idx], Y.iloc[va_idx]

    if native and isinstance(model, xgb.XGBRegressor):
        model = fit_xgb_model_compatible(model, Xtr, Ytr, Xva, Yva, esr, verbose=False)
    else:
        model = fit_multioutput_regressor_with_es(model, Xtr, Ytr, Xva, Yva, esr=esr)

    return model


def ho_objective_function(params, X_train, X_val, y_train, y_val, lag_columns, weights=[5, 3, 2], seed=42,
                          use_native_multioutput=True):
    """
    HO algorithm objective function - only optimize learning_rate, n_estimators and lag feature selection

    params: [learning_rate, n_estimators, lag_feature_1, lag_feature_2, ...]
    where:
    - learning_rate: 0.01 ~ 0.3
    - n_estimators: 100 ~ 2000
    - lag_feature_i: 0 ~ 1 (>0.5 means select this lag feature)
    """

    try:
        version_info = get_xgb_version_info()

        learning_rate = params[0]
        n_estimators = int(params[1])

        selected_lag_features = get_lag_features_from_ho_params(params, lag_columns, max_selected=6)

        base_features = [col for col in X_train.columns if not any(lag in col for lag in lag_columns)]
        final_features = base_features + selected_lag_features

        X_train_subset = X_train[final_features]
        X_val_subset = X_val[final_features]

        tree_method = pick_tree_method(use_native_multioutput and version_info['supports_multi_strategy'])

        params_dict = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'tree_method': tree_method,
            'verbosity': 0,
            'seed': seed,
            'random_state': seed,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': 8,
            'subsample': 0.6018511082680795,
            'colsample_bytree': 0.9991332660884722,
            'colsample_bylevel': 0.9899263233933308,
            'reg_alpha': 2.0091241453173424e-07,
            'reg_lambda': 0.6986580166614497,
            'min_child_weight': 17,
            'gamma': 1.0420513361631241e-07,
        }

        if use_native_multioutput and version_info['supports_multi_strategy']:
            params_dict['multi_strategy'] = 'multi_output_tree'
            model = xgb.XGBRegressor(**params_dict)

            try:
                model.fit(X_train_subset, y_train, verbose=False)
                y_pred_all = model.predict(X_val_subset)

                total_weighted_score = 0
                total_weight = sum(weights)

                for step in range(len(weights)):
                    y_pred = y_pred_all[:, step]
                    y_true = y_val.iloc[:, step].values

                    try:
                        f24_score = F24(None, y_true, y_pred)
                    except Exception as e:
                        f24_score = np.sqrt(mean_squared_error(y_true, y_pred))

                    total_weighted_score += f24_score * weights[step]

                return total_weighted_score / total_weight

            except Exception as e:
                if "Only the hist tree method is supported" in str(e):
                else:
                use_native_multioutput = False

        if not use_native_multioutput or not version_info['supports_multi_strategy']:
            total_weighted_score = 0
            total_weight = sum(weights)

            tree_method_independent = pick_tree_method(use_multioutput=False)
            params_dict['tree_method'] = tree_method_independent

            if 'multi_strategy' in params_dict:
                del params_dict['multi_strategy']

            for step in range(len(weights)):
                np.random.seed(seed + step)

                model = xgb.XGBRegressor(**params_dict)

                try:
                    model.fit(X_train_subset, y_train.iloc[:, step], verbose=False)
                except Exception as e:
                    simple_params = {
                        'objective': 'reg:squarederror',
                        'tree_method': 'hist',
                        'random_state': seed,
                        'n_estimators': 100,
                        'learning_rate': 0.1
                    }
                    model = xgb.XGBRegressor(**simple_params)
                    model.fit(X_train_subset, y_train.iloc[:, step], verbose=False)

                y_pred = model.predict(X_val_subset)
                y_true = y_val.iloc[:, step].values

                try:
                    f24_score = F24(None, y_true, y_pred)
                except Exception as e:
                    f24_score = np.sqrt(mean_squared_error(y_true, y_pred))

                total_weighted_score += f24_score * weights[step]

            return total_weighted_score / total_weight

    except Exception as e:
        return 1e6


def optimize_with_ho(X_train, X_val, y_train, y_val, lag_columns, weights=[5, 3, 2], seed=42,
                     use_native_multioutput=True, max_iterations=50, search_agents=20):
    """
    Use HO algorithm to optimize XGBoost hyperparameters and lag feature selection
    """


    dimension = 2 + len(lag_columns)

    lr_min, lr_max = HO_PARAM_BOUNDS['learning_rate']
    ne_min, ne_max = HO_PARAM_BOUNDS['n_estimators']

    lowerbound = np.array([lr_min, ne_min] + [0.0] * len(lag_columns))

    upperbound = np.array([lr_max, ne_max] + [1.0] * len(lag_columns))


    def wrapped_objective(params):
        return ho_objective_function(params, X_train, X_val, y_train, y_val, lag_columns, weights, seed,
                                     use_native_multioutput)

    try:
        best_fitness, best_position, convergence_curve = HO(
            SearchAgents=search_agents,
            Max_iterations=max_iterations,
            lowerbound=lowerbound,
            upperbound=upperbound,
            dimension=dimension,
            fitness=wrapped_objective
        )


        best_learning_rate = best_position[0]
        best_n_estimators = int(best_position[1])

        best_lag_features = get_lag_features_from_ho_params(best_position, lag_columns)


        best_params = {
            'learning_rate': best_learning_rate,
            'n_estimators': best_n_estimators,
            'selected_lag_features': best_lag_features,
            'ho_raw_position': best_position.tolist()
        }

        return best_params, best_fitness, convergence_curve

    except Exception as e:
        default_params = {
            'learning_rate': 0.1,
            'n_estimators': 500,
            'selected_lag_features': [col for col in lag_columns if '_lag_1' in col][:3],  
            'ho_raw_position': None
        }
        return default_params, 1e6, []


def evaluate_model(models, X_test, y_test, dataset_name="Test Set", weights=[5, 3, 2],
                   df_original=None, ops=None, test_start_idx=None, steps=3,
                   target_col='Extracted_Trend', use_transform=False):
    """
    Evaluate multi-step prediction model performance.
    Supports evaluation in both Original Domain and Transformed Domain.
    
    Args:
        models (list): List of trained models
        X_test: Test features
        y_test: Test targets
        dataset_name (str): Name of dataset
        weights (list): Weights for each step
        df_original: Original data before transformation
        ops (list): Transformation operations
        test_start_idx (int): Starting index in original sequence
        steps (int): Number of prediction steps
        target_col (str): Target column name
        use_transform (bool): Whether transformation was used
        
    Returns:
        dict: Evaluation results with metrics for each step
    """
    results = {}
    total_weighted_f24 = 0
    total_weight = sum(weights)


    if len(models) == 1 and hasattr(models[0], 'predict'):
        model = models[0]
        y_pred_all = model.predict(X_test)

        if len(y_pred_all.shape) == 1:
            y_pred_all = y_pred_all.reshape(-1, 1)
    else:
        y_pred_all = np.column_stack([model.predict(X_test) for model in models])


    for step in range(min(len(weights), y_pred_all.shape[1])):
        y_pred = y_pred_all[:, step]
        y_true = y_test.iloc[:, step].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        try:
            f24_score = F24(None, y_true, y_pred)
        except Exception as e:
            f24_score = rmse

        total_weighted_f24 += f24_score * weights[step]


        step_name = f"step_{step + 1}"
        results[step_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'f24': f24_score,
            'predictions': y_pred,
            'weight': weights[step],
            'domain': 'transformed'
        }

    weighted_avg_f24_trans = total_weighted_f24 / total_weight
    results['weighted_avg_f24_transformed'] = weighted_avg_f24_trans

    if use_transform and df_original is not None and ops is not None and test_start_idx is not None:

        y_orig_full = df_original[target_col].values
        y_pred_original = invert_direct_block(y_pred_all, y_orig_full, test_start_idx, steps, ops, season=96)

        y_true_original = np.zeros((len(X_test), steps))
        for h in range(steps):
            for i in range(len(X_test)):
                if test_start_idx + i + h + 1 < len(y_orig_full):
                    y_true_original[i, h] = y_orig_full[test_start_idx + i + h + 1]
                else:
                    y_true_original[i, h] = y_orig_full[-1]  

        total_weighted_f24_orig = 0

        for step in range(min(len(weights), steps)):
            y_pred_orig = y_pred_original[:, step]
            y_true_orig = y_true_original[:, step]

            rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
            r2_orig = r2_score(y_true_orig, y_pred_orig)
            mape_orig = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100

            try:
                f24_score_orig = F24(None, y_true_orig, y_pred_orig)
            except Exception as e:
                f24_score_orig = rmse_orig

            total_weighted_f24_orig += f24_score_orig * weights[step]


            step_name_orig = f"step_{step + 1}_original"
            results[step_name_orig] = {
                'rmse': rmse_orig,
                'mae': mae_orig,
                'r2': r2_orig,
                'mape': mape_orig,
                'f24': f24_score_orig,
                'predictions': y_pred_orig,
                'weight': weights[step],
                'domain': 'original'
            }

        weighted_avg_f24_orig = total_weighted_f24_orig / total_weight
        results['weighted_avg_f24_original'] = weighted_avg_f24_orig
        results['weighted_avg_f24'] = weighted_avg_f24_orig  

        results['y_true_original'] = y_true_original
        results['y_pred_original'] = y_pred_original

    else:
        results['weighted_avg_f24'] = weighted_avg_f24_trans

    return results


def analyze_feature_importance(models, feature_names, top_n=20):
    """Analyze Feature importance of multi-step prediction model, handling multi-target tree limitations"""

    if len(models) == 1:
        model = models[0]
        try:
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            top_features = feature_importance_df.head(top_n)
            for idx, row in top_features.iterrows():

            return feature_importance_df

        except Exception as e:
            if "multi-target tree" in str(e) or "not yet implemented" in str(e):

                return pd.DataFrame({'feature': feature_names, 'importance': [0.0] * len(feature_names)})
            else:
                return pd.DataFrame({'feature': feature_names, 'importance': [0.0] * len(feature_names)})

    else:
        all_importance_df = []

        for step, model in enumerate(models):
            try:
                importance = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance,
                    'step': f'Step {step + 1}'
                }).sort_values('importance', ascending=False)

                all_importance_df.append(feature_importance_df)

                top_features = feature_importance_df.head(top_n)
                for idx, row in top_features.iterrows():

            except Exception as e:
                empty_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': [0.0] * len(feature_names),
                    'step': f'Step {step + 1}'
                })
                all_importance_df.append(empty_df)

        if all_importance_df:
            combined_df = pd.concat(all_importance_df, ignore_index=True)

            avg_importance = combined_df.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False)

            if avg_importance['importance'].sum() > 0:
                top_avg_features = avg_importance.head(top_n)
                for idx, row in top_avg_features.iterrows():

            return combined_df
        else:
            return pd.DataFrame({'feature': feature_names, 'importance': [0.0] * len(feature_names), 'step': 'unknown'})


def main(file_path, steps=3, weights=[5, 3, 2], seed=42, use_transform=False, use_native_multioutput=True,
         max_iterations=50, search_agents=20):
    """
    Main execution function - Enhanced version with comprehensive logging and file output.
    
    Args:
        file_path (str): Path to CSV data file
        steps (int): Number of prediction steps (default: 3)
        weights (list): Weights for each step in F24 calculation (default: [5, 3, 2])
        seed (int): Random seed for reproducibility (default: 42)
        use_transform (bool): Whether to apply differencing transform (default: False)
        use_native_multioutput (bool): Use XGBoost 2.0+ native multi-output (default: True)
        max_iterations (int): HO algorithm max iterations (default: 50)
        search_agents (int): Number of search agents in HO algorithm (default: 20)
        
    Returns:
        dict: Comprehensive results including models, metrics, and analysis
    """
    # Set up output directories and logging
    output_dir, subdirs, logger = setup_logging_and_output(seed)

    try:
        # Initialize random seeds for reproducibility
        set_random_seeds(seed)

        # Check XGBoost version and capabilities
        version_info = get_xgb_version_info()


        # Disable native multi-output if not supported by current XGBoost version
        if use_native_multioutput and not version_info['supports_multi_strategy']:
            use_native_multioutput = False

        # Verify F24 evaluation function is available
        check_f24_function()

        try:
            import cupy
        except ImportError:

        # Load time series data from CSV file
        df = load_data(file_path)

        # Split data into train/validation/test sets
        # Apply differencing transform if use_transform=True
        (X_train, X_val, X_test, y_train, y_val, y_test,
         df_original, ops, train_end, val_end) = split_data(
            df, steps=steps, use_transform=use_transform
        )

        # Extract lag feature column names for feature selection
        lag_columns = X_train.attrs.get('lag_columns', [])
        if not lag_columns:
            lag_columns = [col for col in X_train.columns if '_lag_' in col]

        # Store lag column info in all datasets for consistency
        for dataset in [X_train, X_val, X_test]:
            dataset.attrs['lag_columns'] = lag_columns

        # Record starting indices for inverse transform later
        train_start_idx = 0
        val_start_idx = train_end
        test_start_idx = val_end


        # Optimize hyperparameters using HO (Hippopotamus Optimization) algorithm
        # This will find optimal learning_rate, n_estimators, and lag feature subset
        best_params, best_score, convergence_curve = optimize_with_ho(
            X_train, X_val, y_train, y_val,
            lag_columns, weights, seed, use_native_multioutput,
            max_iterations, search_agents
        )

        # Log best parameters found by HO algorithm
        for key, value in best_params.items():
            if key not in ['selected_lag_features', 'ho_raw_position']:

        # Extract selected lag features from optimization result
        selected_lag_features = best_params['selected_lag_features']
        for feature in selected_lag_features:



        # Combine base features with selected lag features
        base_features = [col for col in X_train.columns if not any(lag in col for lag in lag_columns)]
        final_features = base_features + selected_lag_features

        # Create final feature sets with selected features only
        X_train_final = X_train[final_features]
        X_val_final = X_val[final_features]
        X_test_final = X_test[final_features]

        # Select appropriate tree method based on model type
        final_tree_method = pick_tree_method(use_native_multioutput and version_info['supports_multi_strategy'])

        # Prepare final XGBoost parameters
        # Combine fixed params with optimized params
        final_xgb_params = FIXED_XGBOOST_PARAMS.copy()
        final_xgb_params.update({
            'eval_metric': 'rmse',
            'tree_method': final_tree_method,
            'seed': seed,
            'random_state': seed,
            'learning_rate': best_params['learning_rate'],
            'n_estimators': best_params['n_estimators']
        })


        # Train model(s) using best parameters
        # Two strategies: joint model (XGBoost 2.0+) or independent models (older versions)
        if use_native_multioutput and version_info['supports_multi_strategy']:
            # Use native multi-output regression (XGBoost 2.0+)
            final_xgb_params['multi_strategy'] = 'multi_output_tree'
            model = xgb.XGBRegressor(**final_xgb_params)

            try:
                model = fit_xgb_model_compatible(model, X_train_final, y_train, X_val_final, y_val,
                                                 early_stopping_rounds=50, verbose=True)
                final_models = [model]  # Single model for all outputs
            except Exception as e:
                if "Only the hist tree method is supported" in str(e):
                    final_xgb_params['tree_method'] = 'hist'
                    try:
                        model = xgb.XGBRegressor(**final_xgb_params)
                        model = fit_xgb_model_compatible(model, X_train_final, y_train, X_val_final, y_val,
                                                         early_stopping_rounds=50, verbose=True)
                        final_models = [model]
                    except Exception as e2:
                        use_native_multioutput = False
                else:
                    use_native_multioutput = False

        # Train independent models if native multi-output is not available
        if not use_native_multioutput or not version_info['supports_multi_strategy']:
            final_models = []

            # Use GPU-compatible tree method for independent models
            independent_tree_method = pick_tree_method(use_multioutput=False)
            final_xgb_params['tree_method'] = independent_tree_method

            # Remove multi_strategy parameter for independent training
            if 'multi_strategy' in final_xgb_params:
                del final_xgb_params['multi_strategy']


            # Train a separate model for each prediction step
            for step in range(steps):

                # Set different random seed for each step for diversity
                np.random.seed(seed + step)

                model = xgb.XGBRegressor(**final_xgb_params)

                try:
                    # Train with early stopping based on validation set
                    model = fit_xgb_model_compatible(
                        model, X_train_final, y_train.iloc[:, step],
                        X_val_final, y_val.iloc[:, step],
                        early_stopping_rounds=50, verbose=True
                    )
                    final_models.append(model)
                except Exception as e:
                    # Fallback to simple parameters if training fails
                    simple_params = {
                        'objective': 'reg:squarederror',
                        'tree_method': 'hist',
                        'random_state': seed + step,
                        'n_estimators': 100,
                        'learning_rate': 0.1
                    }
                    model = xgb.XGBRegressor(**simple_params)
                    model.fit(X_train_final, y_train.iloc[:, step])
                    final_models.append(model)


        # Prepare evaluation parameters
        eval_params = {
            'weights': weights,
            'df_original': df_original,
            'ops': ops,
            'steps': steps,
            'target_col': 'Extracted_Trend',
            'use_transform': use_transform
        }

        # Evaluate model on training set
        train_results = evaluate_model(
            final_models, X_train_final, y_train, "Training Set",
            test_start_idx=train_start_idx, **eval_params
        )

        # Evaluate model on validation set
        val_results = evaluate_model(
            final_models, X_val_final, y_val, "Validation Set",
            test_start_idx=val_start_idx, **eval_params
        )

        # Evaluate model on test set
        test_results = evaluate_model(
            final_models, X_test_final, y_test, "Test Set",
            test_start_idx=test_start_idx, **eval_params
        )


        # Analyze feature importance from trained models
        feature_importance_analysis = analyze_feature_importance(final_models, final_features)

        # Organize all results into a comprehensive dictionary
        results = {
            'models': final_models,
            'best_params': best_params,
            'train_results': train_results,
            'val_results': val_results,
            'test_results': test_results,
            'feature_importance': feature_importance_analysis,
            'ho_convergence': convergence_curve,
            'steps': steps,
            'weights': weights,
            'seed': seed,
            'df_original': df_original,
            'transform_ops': ops,
            'use_transform': use_transform,
            'final_features': final_features,
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'results_summary': {
                'seed': seed,
                'best_params': {k: v for k, v in best_params.items() if
                                k not in ['selected_lag_features', 'ho_raw_position']},
                'selected_lag_features': selected_lag_features,
                'total_features_used': len(final_features),
                'best_value': best_score,
                'test_weighted_f24': test_results['weighted_avg_f24'],
                'steps': steps,
                'weights': weights,
                'use_transform': use_transform,
                'use_native_multioutput': use_native_multioutput,
                'transform_ops': ops,
                'xgboost_version': xgb.__version__,
                'use_original_domain_for_main_score': use_transform and len(ops) > 0,
                'optimization_algorithm': 'HO',
                'ho_iterations': max_iterations,
                'ho_search_agents': search_agents,
                'output_directory': output_dir
            }
        }

        save_comprehensive_results(results, output_dir, subdirs, logger, file_path, X_test_final, y_test)


        if results['use_transform'] and 'weighted_avg_f24_original' in results['test_results']:
            main_domain = "Original Domain"
        else:
            main_domain = "Transformed Domain" if results['use_transform'] else "Original Domain"

        for step in range(results['steps']):
            step_key = f"step_{step + 1}"
            if results['use_transform'] and f"{step_key}_original" in results['test_results']:
                orig_results = results['test_results'][f"{step_key}_original"]
                    f"Step {step + 1} (Weight{results['weights'][step]}) [Original Domain] - F24: {orig_results['f24']:.4f}, R²: {orig_results['r2']:.4f}")
            else:
                step_results = results['test_results'][step_key]
                    f"Step {step + 1} (Weight{results['weights'][step]}) [{main_domain}] - F24: {step_results['f24']:.4f}, R²: {step_results['r2']:.4f}")



        return results

    except Exception as e:
        raise

    finally:
        for handler in logging.getLogger('xgboost_forecasting').handlers:
            handler.close()


# Main program execution
if __name__ == "__main__":
    # Configuration: Specify input CSV file path
    file_path = "predictive_optimized_ceemdan_decomposition_20250928_051436.csv"

    # Set fixed random seed for reproducibility
    RANDOM_SEED = 42


    # Check XGBoost version and features
    version_info = get_xgb_version_info()

    if version_info['supports_multi_strategy']:
    else:

    # Determine whether to use native multi-output based on version
    use_multioutput = version_info['supports_multi_strategy']

    # Run main training pipeline
    # - steps=3: Predict 3 future time steps
    # - weights=[5, 3, 2]: Weighted importance for each step in F24 metric
    # - use_transform=True: Apply differencing transform for stationarity
    # - max_iterations=100: HO algorithm iterations
    # - search_agents=32: Number of search agents in HO algorithm
    results = main(
        file_path,
        steps=3,
        weights=[5, 3, 2],
        seed=RANDOM_SEED,
        use_transform=True,  
        use_native_multioutput=use_multioutput,  
        max_iterations=100,  
        search_agents=32  
    )


    # Determine which domain to report based on transform settings
    if results['use_transform'] and 'weighted_avg_f24_original' in results['test_results']:
        main_domain = "Original Domain"
    else:
        main_domain = "Transformed Domain" if results['use_transform'] else "Original Domain"

    # Display results for each prediction step
    for step in range(results['steps']):
        step_key = f"step_{step + 1}"
        if results['use_transform'] and f"{step_key}_original" in results['test_results']:
            orig_results = results['test_results'][f"{step_key}_original"]
                f"Step {step + 1} (Weight{results['weights'][step]}) [Original Domain] - F24: {orig_results['f24']:.4f}, R²: {orig_results['r2']:.4f}")
        else:
            step_results = results['test_results'][step_key]
                f"Step {step + 1} (Weight{results['weights'][step]}) [{main_domain}] - F24: {step_results['f24']:.4f}, R²: {step_results['r2']:.4f}")



    # Additional checks or operations can be added here
    if results['use_transform']:
    else:




print("Finish!")
