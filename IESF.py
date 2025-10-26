import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy import stats
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
import warnings
import gc
import sys

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from PyEMD import CEEMDAN
    CEEMDAN_AVAILABLE = True
except ImportError:
    CEEMDAN_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False


# Reproducible CEEMDAN decomposition class with controlled noise generation
class ReproducibleCEEMDAN:
    def __init__(self, trials=100, epsilon=0.02, seed=42):
        self.trials = trials
        self.epsilon = epsilon
        self.seed = seed

    # Generate Laplace noise for more stable decomposition
    def _laplace_noise_generator(self, *args, **kwargs):
        if args:
            shape = args[0] if isinstance(args[0], (tuple, list)) else args
        else:
            shape = (1,)
        scale = 1.0 / np.sqrt(2.0)
        return np.random.laplace(loc=0.0, scale=scale, size=shape)

    # Perform CEEMDAN decomposition with reproducible noise
    def decompose(self, signal):
        self._reset_random_state()
        original_randn = np.random.randn
        original_normal = np.random.normal

        try:
            # Replace random number generators with Laplace noise
            np.random.randn = self._laplace_noise_generator
            np.random.normal = lambda loc=0.0, scale=1.0, size=None: (
                np.random.laplace(loc=loc, scale=scale / np.sqrt(2.0), size=size)
            )
            ceemdan = CEEMDAN(trials=self.trials, epsilon=self.epsilon)
            ceemdan.noise_seed(self.seed)
            imfs = ceemdan(signal)
            residual = signal - np.sum(imfs, axis=0)
        finally:
            # Restore original random number generators
            np.random.randn = original_randn
            np.random.normal = original_normal

        return imfs, residual

    def _reset_random_state(self):
        np.random.seed(self.seed)
        gc.collect()


# Calculate orthogonality index to measure IMF independence
def calculate_orthogonality_index(imfs):
    n_imfs = imfs.shape[0]
    cross_terms = 0
    # Sum all cross-correlation terms
    for i in range(n_imfs):
        for j in range(i + 1, n_imfs):
            cross_terms += np.sum(imfs[i] * imfs[j])
    total_energy = np.sum(imfs ** 2)
    if total_energy > 0:
        oi = 2 * abs(cross_terms) / total_energy
    else:
        oi = 0
    return oi


# Estimate the main period of a signal using FFT
def estimate_main_period(signal):
    if len(signal) < 4:
        return len(signal)
    # Compute FFT and power spectrum
    fft_vals = fft(signal - np.mean(signal))
    freqs = fftfreq(len(signal))
    power_spectrum = np.abs(fft_vals) ** 2
    positive_freqs = freqs[1:len(freqs)//2]
    positive_power = power_spectrum[1:len(power_spectrum)//2]
    if len(positive_power) == 0:
        return len(signal)
    # Find dominant frequency
    max_power_idx = np.argmax(positive_power)
    dominant_freq = positive_freqs[max_power_idx]
    if dominant_freq > 0:
        main_period = 1.0 / dominant_freq
    else:
        main_period = len(signal)
    return main_period


# Create lagged feature matrix for autoregressive modeling
def create_lag_matrix(signal, p):
    signal = np.array(signal)
    n = len(signal)
    if n <= p:
        return np.array([]).reshape(0, p), np.array([])
    X = np.zeros((n - p, p))
    # Create p lagged features
    for i in range(p):
        X[:, i] = signal[p-1-i:n-1-i]
    y = signal[p:]
    return X, y


# Calculate Mean Absolute Scaled Error for forecast evaluation
def calculate_mase(y_true, y_pred, y_train):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    if len(y_true) == 0 or len(y_train) <= 1:
        return 1.0
    # Forecast error
    forecast_error = np.mean(np.abs(y_true - y_pred))
    # Naive forecast error (using first difference)
    if len(y_train) > 1:
        naive_error = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_error = 1.0
    if naive_error == 0:
        naive_error = 1e-10
    mase = forecast_error / naive_error
    return mase


# Perform rolling window AR prediction to assess forecastability
def rolling_ar_prediction(signal, p=8, H=1, min_train=128, step=8):
    signal = np.array(signal)
    n = len(signal)
    if n < min_train + p + H:
        return 1.0
    predictions = []
    actuals = []
    # Rolling window forecasting
    for start_idx in range(min_train, n - H, step):
        if start_idx + H >= n:
            break
        train_signal = signal[:start_idx]
        X_train, y_train = create_lag_matrix(train_signal, p)
        if len(X_train) == 0:
            continue
        try:
            # Train AR model and make multi-step forecast
            model = LinearRegression()
            model.fit(X_train, y_train)
            current_signal = train_signal.copy()
            pred_values = []
            for h in range(H):
                if len(current_signal) < p:
                    break
                X_pred = current_signal[-p:][::-1].reshape(1, -1)
                next_pred = model.predict(X_pred)[0]
                pred_values.append(next_pred)
                current_signal = np.append(current_signal, next_pred)
            if len(pred_values) == H:
                predictions.extend(pred_values)
                actuals.extend(signal[start_idx:start_idx + H].tolist())
        except Exception:
            continue
    if len(predictions) == 0:
        return 1.0
    mase = calculate_mase(actuals, predictions, signal)
    return mase


# Assign frequency band weight based on period and forecast horizon
def frequency_band_weight(period, H):
    if period < 1.5 * H:
        return 0.25  # High frequency components
    elif 1.5 * H <= period <= 8 * H:
        return 1.00  # Mid frequency components (most relevant)
    else:
        return 0.50  # Low frequency components


# Calculate predictive error proxy using weighted component forecasts
def calculate_pred_err_proxy(imfs, residual, H_set=[1, 4], p=8, min_train=128, step=8):
    # Combine IMFs and residual into components
    components = []
    for i in range(imfs.shape[0]):
        components.append(imfs[i])
    components.append(np.array(residual))
    # Calculate energy and period for each component
    energies = []
    periods = []
    for component in components:
        energy = np.sum(component ** 2)
        energies.append(energy)
        period = estimate_main_period(component)
        periods.append(period)
    total_energy = sum(energies)
    if total_energy == 0:
        return 1.0
    energy_ratios = [e / total_energy for e in energies]
    # Evaluate forecast error for different horizons
    E_values = []
    for H in H_set:
        component_mases = []
        weights = []
        for i, component in enumerate(components):
            mase_h = rolling_ar_prediction(component, p=p, H=H, 
                                         min_train=min_train, step=step)
            component_mases.append(mase_h)
            # Weight by energy ratio and frequency band relevance
            g_weight = frequency_band_weight(periods[i], H)
            weights.append(energy_ratios[i] * g_weight)
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        E_H = sum(w * mase for w, mase in zip(weights, component_mases))
        E_values.append(E_H)
    pred_err_proxy = np.mean(E_values)
    return pred_err_proxy


# Calculate comprehensive quality score for decomposition
def calculate_new_quality_score(imfs, residual, signal):
    # Predictive error proxy (most important)
    pred_err_proxy = calculate_pred_err_proxy(imfs, residual, H_set=[1, 4])
    # Reconstruction error
    reconstructed = np.sum(imfs, axis=0) + residual
    mse = np.mean((signal - reconstructed) ** 2)
    signal_power = np.mean(signal ** 2)
    rrmse = np.sqrt(mse / signal_power) if signal_power > 0 else 0
    # Orthogonality index
    oi = calculate_orthogonality_index(imfs)
    # Complexity penalty based on number of IMFs
    K = imfs.shape[0]
    K_target = 6
    cpx = abs(K - K_target) / K_target + 0.1 * max(0, K - 10)
    # Energy conservation check
    imf_energy = np.sum(imfs ** 2)
    residual_energy = np.sum(residual ** 2)
    signal_energy = np.sum(signal ** 2)
    energy_gap = abs(imf_energy + residual_energy - signal_energy) / signal_energy if signal_energy > 0 else 0
    # Weighted combination of all metrics
    final_score = (0.55 * pred_err_proxy + 0.15 * rrmse + 
                   0.10 * oi + 0.10 * cpx + 0.10 * energy_gap)
    return {
        'pred_err_proxy': pred_err_proxy,
        'rrmse': rrmse,
        'oi': oi,
        'cpx': cpx,
        'energy_gap': energy_gap,
        'final_score': final_score
    }


# Normalize scores across different parameter settings for fair comparison
def normalize_and_combine_scores(all_scores_list):
    if not all_scores_list:
        return []
    metrics = ['pred_err_proxy', 'rrmse', 'oi', 'cpx', 'energy_gap']
    # Collect all metric values
    metric_values = {metric: [] for metric in metrics}
    for scores in all_scores_list:
        for metric in metrics:
            metric_values[metric].append(scores[metric])
    # Min-max normalization for each metric
    normalized_values = {metric: [] for metric in metrics}
    for metric in metrics:
        values = np.array(metric_values[metric])
        min_val = np.min(values)
        max_val = np.max(values)
        eps = 1e-10
        if max_val - min_val < eps:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - min_val) / (max_val - min_val + eps)
        normalized_values[metric] = normalized.tolist()
    # Apply weights to normalized metrics
    weights = {
        'pred_err_proxy': 0.55,
        'rrmse': 0.15,
        'oi': 0.10,
        'cpx': 0.10,
        'energy_gap': 0.10
    }
    final_scores = []
    for i in range(len(all_scores_list)):
        final_score = sum(weights[metric] * normalized_values[metric][i] 
                         for metric in metrics)
        final_scores.append(final_score)
    return final_scores


# Optimize epsilon parameter using Bayesian optimization
def optimize_epsilon_bayesian(signal, trials=100, seed=42, n_calls=25, strategy='predictive'):
    if not SKOPT_AVAILABLE:
        return 0.4

    # Evaluate reference epsilon values to establish baseline
    reference_epsilons = [0.2, 0.3, 0.4, 0.5, 0.6]
    reference_scores = []
    for eps in reference_epsilons:
        decomposer = ReproducibleCEEMDAN(trials=trials, epsilon=eps, seed=seed)
        imfs, residual = decomposer.decompose(signal)

        if strategy == 'predictive':
            score_dict = calculate_new_quality_score(imfs, residual, signal)
            score = score_dict['final_score']
        elif strategy == 'comprehensive':
            score_dict = calculate_new_quality_score(imfs, residual, signal)
            score = score_dict['final_score']
        elif strategy == 'balanced':
            oi = calculate_orthogonality_index(imfs)
            reconstructed = np.sum(imfs, axis=0) + residual
            mse = np.mean((signal - reconstructed) ** 2)
            signal_power = np.mean(signal ** 2)
            rrmse = np.sqrt(mse / signal_power) if signal_power > 0 else 0
            n_imfs = imfs.shape[0]
            target_imfs = 6
            imf_penalty = abs(n_imfs - target_imfs) / target_imfs
            score = 0.2 * oi + 0.5 * rrmse * 100 + 0.3 * imf_penalty
        else:
            score = calculate_orthogonality_index(imfs)

        reference_scores.append((eps, score))

    best_ref_eps, best_ref_score = min(reference_scores, key=lambda x: x[1])

    # Objective function with robustness check (average over multiple runs)
    def robust_objective_function(epsilon_list):
        epsilon_val = epsilon_list[0]
        try:
            scores = []
            for run in range(2):
                run_seed = seed + run * 1000
                decomposer = ReproducibleCEEMDAN(trials=trials, epsilon=epsilon_val, seed=run_seed)
                imfs, residual = decomposer.decompose(signal)

                if strategy == 'predictive':
                    score_dict = calculate_new_quality_score(imfs, residual, signal)
                    score = score_dict['final_score']
                elif strategy == 'comprehensive':
                    score_dict = calculate_new_quality_score(imfs, residual, signal)
                    score = score_dict['final_score']
                elif strategy == 'balanced':
                    oi = calculate_orthogonality_index(imfs)
                    reconstructed = np.sum(imfs, axis=0) + residual
                    mse = np.mean((signal - reconstructed) ** 2)
                    signal_power = np.mean(signal ** 2)
                    rrmse = np.sqrt(mse / signal_power) if signal_power > 0 else 0
                    n_imfs = imfs.shape[0]
                    target_imfs = 6
                    imf_penalty = abs(n_imfs - target_imfs) / target_imfs
                    score = 0.2 * oi + 0.5 * rrmse * 100 + 0.3 * imf_penalty
                else:
                    score = calculate_orthogonality_index(imfs)

                scores.append(score)
            avg_score = np.mean(scores)
            return avg_score
        except Exception as e:
            return 1e6

    # Define search space around best reference epsilon
    if strategy in ['predictive', 'comprehensive', 'balanced']:
        center = best_ref_eps
        radius = 0.15
        lower_bound = max(0.1, center - radius)
        upper_bound = min(0.8, center + radius)
    else:
        lower_bound, upper_bound = 0.1, 0.8

    space = [Real(lower_bound, upper_bound, name='epsilon')]

    try:
        # Run Bayesian optimization
        result = gp_minimize(
            robust_objective_function,
            space,
            n_calls=n_calls,
            random_state=seed,
            acq_func='EI',
            n_initial_points=6,
            n_jobs=1,
            xi=0.05,
            kappa=2.0
        )
        optimal_epsilon = result.x[0]
        optimal_score = result.fun
        improvement = (best_ref_score - optimal_score) / best_ref_score * 100

        # Return optimal or reference epsilon based on improvement
        if improvement > 0:
            return optimal_epsilon
        else:
            return best_ref_eps
    except Exception as e:
        return best_ref_eps


# Main class for alpha-optimized CEEMDAN decomposition
class AlphaOptimizedCEEMDAN:
    def __init__(self, h_min=30, ceemdan_trials=100, seed=42):
        self.h_min = h_min
        self.ceemdan_trials = ceemdan_trials
        self.seed = seed
        # Theoretical constraint on alpha based on minimum half-life
        self.alpha_max_constraint = 1 - 2 ** (-1 / h_min)
        self.alpha_min = 0.01
        self.optimal_alpha = None
        self.best_ceemdan_results = None
        self.optimization_history = []
        self.series = None
        self.original_data = None

        if not CEEMDAN_AVAILABLE:
            raise ImportError("PyEMD library not installed")

    # Load time series data from DataFrame or array
    def load_data(self, data, column_name=None):
        if isinstance(data, pd.DataFrame):
            self.original_data = data.copy()
            if column_name is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found")
                column_name = numeric_cols[0]
            self.series = data[column_name].dropna().values
        else:
            self.series = np.array(data)
            self.original_data = pd.DataFrame({'data': self.series})
        return self.series

    # Single exponential smoothing
    def single_smoothing(self, series, alpha):
        n = len(series)
        result = np.zeros(n)
        result[0] = series[0]
        for t in range(1, n):
            result[t] = alpha * series[t] + (1 - alpha) * result[t - 1]
        return result

    # Double exponential smoothing for trend extraction
    def double_smoothing(self, series, alpha):
        single_smooth = self.single_smoothing(series, alpha)
        n = len(series)
        double_smooth = np.zeros(n)
        double_smooth[0] = single_smooth[0]
        for t in range(1, n):
            double_smooth[t] = alpha * single_smooth[t] + (1 - alpha) * double_smooth[t - 1]
        return single_smooth, double_smooth

    # Zero-phase trend extraction using forward-backward smoothing
    def forward_backward_smoothing(self, series, alpha):
        _, forward_trend = self.double_smoothing(series, alpha)
        reversed_series = series[::-1]
        _, backward_trend = self.double_smoothing(reversed_series, alpha)
        backward_trend = backward_trend[::-1]
        # Average forward and backward trends to eliminate phase shift
        zero_phase_trend = (forward_trend + backward_trend) / 2
        return zero_phase_trend

    # Objective function for alpha optimization
    def objective_function(self, alpha):
        try:
            alpha_val = alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha
            # Check alpha constraints
            if alpha_val > self.alpha_max_constraint or alpha_val < self.alpha_min:
                return 10.0
            # Extract trend and compute residuals
            trend = self.forward_backward_smoothing(self.series, alpha_val)
            residuals = self.series - trend
            # Optimize epsilon for CEEMDAN on residuals
            optimal_epsilon = optimize_epsilon_bayesian(
                residuals,
                trials=self.ceemdan_trials,
                seed=self.seed,
                n_calls=25,
                strategy='predictive'
            )
            # Decompose residuals with CEEMDAN
            decomposer = ReproducibleCEEMDAN(trials=self.ceemdan_trials, epsilon=optimal_epsilon, seed=self.seed)
            imfs, ceemdan_residual = decomposer.decompose(residuals)
            # Evaluate decomposition quality
            score_dict = calculate_new_quality_score(imfs, ceemdan_residual, residuals)
            quality_score = score_dict['final_score']
            return quality_score
        except Exception as e:
            return 10.0

    # Optimize alpha parameter using Bayesian or grid search
    def optimize_alpha(self, n_calls=15, method='bayesian'):
        if self.series is None:
            raise ValueError("Please load data first")

        evaluation_results = []

        if method == 'bayesian' and SKOPT_AVAILABLE:
            space = [Real(self.alpha_min, self.alpha_max_constraint, name='alpha')]

            @use_named_args(space)
            def objective_wrapper(**params):
                alpha_val = params['alpha']
                objective_val = self.objective_function(alpha_val)
                evaluation_results.append({
                    'alpha': alpha_val,
                    'quality_score': objective_val
                })
                return objective_val

            result = gp_minimize(objective_wrapper, space, n_calls=n_calls,
                                 random_state=self.seed, acq_func='EI')
            self.optimal_alpha = result.x[0]
            best_score = result.fun
        else:
            # Grid search fallback
            alpha_values = np.linspace(self.alpha_min, self.alpha_max_constraint, n_calls)
            best_alpha = None
            best_score = float('inf')

            for i, alpha_val in enumerate(alpha_values):
                objective_val = self.objective_function(alpha_val)
                evaluation_results.append({
                    'alpha': alpha_val,
                    'quality_score': objective_val
                })
                if objective_val < best_score:
                    best_score = objective_val
                    best_alpha = alpha_val
            self.optimal_alpha = best_alpha

        self.optimization_history = evaluation_results
        return self.optimal_alpha

    # Get final decomposition using optimized parameters
    def get_final_decomposition(self):
        if self.optimal_alpha is None:
            raise ValueError("Please run optimization first")

        # Extract trend with optimal alpha
        trend = self.forward_backward_smoothing(self.series, self.optimal_alpha)
        residuals_after_smoothing = self.series - trend
        # Optimize epsilon for final decomposition
        optimal_epsilon = optimize_epsilon_bayesian(
            residuals_after_smoothing,
            trials=self.ceemdan_trials,
            seed=self.seed,
            n_calls=25,
            strategy='predictive'
        )
        # Final CEEMDAN decomposition
        decomposer = ReproducibleCEEMDAN(trials=self.ceemdan_trials, epsilon=optimal_epsilon, seed=self.seed)
        imfs, ceemdan_residual = decomposer.decompose(residuals_after_smoothing)
        score_dict = calculate_new_quality_score(imfs, ceemdan_residual, residuals_after_smoothing)
        # Verify reconstruction accuracy
        reconstructed_residuals = np.sum(imfs, axis=0) + ceemdan_residual
        reconstruction_error = np.mean((residuals_after_smoothing - reconstructed_residuals) ** 2)

        results = {
            'optimal_alpha': self.optimal_alpha,
            'optimal_epsilon': optimal_epsilon,
            'half_life': -np.log(2) / np.log(1 - self.optimal_alpha) if self.optimal_alpha < 0.999 else np.inf,
            'original_series': self.series,
            'extracted_trend': trend,
            'residuals_after_smoothing': residuals_after_smoothing,
            'imfs': imfs,
            'ceemdan_residual': ceemdan_residual,
            'num_imfs': imfs.shape[0],
            'score_dict': score_dict,
            'reconstruction_error': reconstruction_error,
            'optimization_history': self.optimization_history
        }

        self.best_ceemdan_results = results
        return results

    # Save decomposition results to CSV files
    def save_results(self, results, save_path=None):
        if save_path is None:
            save_path = 'predictive_alpha_optimized_ceemdan_results'

        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare decomposition data
        decomp_data = {
            'Original_Series': results['original_series'],
            'Extracted_Trend': results['extracted_trend'],
            'Residuals_After_Smoothing': results['residuals_after_smoothing'],
            'CEEMDAN_Residual': results['ceemdan_residual']
        }

        for i, imf in enumerate(results['imfs']):
            decomp_data[f'IMF_{i + 1}'] = imf

        decomp_df = pd.DataFrame(decomp_data)
        decomp_path = os.path.join(save_path, f'predictive_optimized_ceemdan_decomposition_{timestamp}.csv')
        decomp_df.to_csv(decomp_path, index=False, encoding='utf-8-sig')

        return save_path


# Main execution function
def main():
    if not CEEMDAN_AVAILABLE:
        print("Error: PyEMD library required")
        return

    import glob
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found in current directory")
        return

    # Process all CSV files with Wind speed column
    for csv_file in csv_files:
        try:
            data = pd.read_csv(csv_file)
            if 'Wind speed' not in data.columns:
                continue

            wind_speed_data = data['Wind speed'].dropna()
            if len(wind_speed_data) < 50:
                continue

            # Initialize optimizer with parameters
            optimizer = AlphaOptimizedCEEMDAN(
                h_min=1,
                ceemdan_trials=100,
                seed=42
            )

            # Run optimization and decomposition
            optimizer.load_data(data, 'Wind speed')
            optimal_alpha = optimizer.optimize_alpha(
                n_calls=20,
                method='bayesian' if SKOPT_AVAILABLE else 'grid'
            )

            results = optimizer.get_final_decomposition()
            file_base_name = os.path.splitext(csv_file)[0]
            save_path = f'predictive_alpha_optimized_results_{file_base_name}'
            optimizer.save_results(results, save_path=save_path)

        except Exception as e:
            continue


if __name__ == "__main__":
    main()