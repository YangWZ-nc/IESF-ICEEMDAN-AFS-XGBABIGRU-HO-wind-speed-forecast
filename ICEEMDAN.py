# Improved CEEMDAN with Laplacian Noise & Bayesian Optimization
# Key Features: Predictive-oriented decomposition, adaptive epsilon tuning, enhanced stability

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from datetime import datetime
import os
import glob
import warnings
from scipy.signal import hilbert
import gc
import sys

try:
    from skopt import gp_minimize
    from skopt.space import Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

warnings.filterwarnings('ignore')


class ReproducibleCEEMDAN:
    # Reproducible CEEMDAN with Laplacian white noise (superior to Gaussian for time series)

    def __init__(self, trials=100, epsilon=0.02, seed=42):
        self.trials = trials
        self.epsilon = epsilon
        self.seed = seed

    def _laplace_noise_generator(self, *args, **kwargs):
        # Laplace noise generator: scale=1/√2 ensures std=1, matching standard Gaussian
        if args:
            shape = args[0] if isinstance(args[0], (tuple, list)) else args
        else:
            shape = (1,)

        scale = 1.0 / np.sqrt(2.0)
        return np.random.laplace(loc=0.0, scale=scale, size=shape)

    def decompose(self, signal):
        self._reset_random_state()

        original_randn = np.random.randn
        original_normal = np.random.normal

        try:
            np.random.randn = self._laplace_noise_generator
            np.random.normal = lambda loc=0.0, scale=1.0, size=None: (
                np.random.laplace(loc=loc, scale=scale / np.sqrt(2.0), size=size)
            )

            ceemdan = CEEMDAN(trials=self.trials, epsilon=self.epsilon)
            ceemdan.noise_seed(self.seed)

            imfs = ceemdan(signal)

            residual = signal - np.sum(imfs, axis=0)

        finally:
            np.random.randn = original_randn
            np.random.normal = original_normal

        return imfs, residual

    def _reset_random_state(self):
        np.random.seed(self.seed)
        gc.collect()


def calculate_orthogonality_index(imfs):
    n_imfs = imfs.shape[0]
    cross_terms = 0

    for i in range(n_imfs):
        for j in range(i + 1, n_imfs):
            cross_terms += np.sum(imfs[i] * imfs[j])

    total_energy = np.sum(imfs ** 2)
    if total_energy > 0:
        oi = 2 * abs(cross_terms) / total_energy
    else:
        oi = 0
    return oi


def calculate_predictive_quality_score(imfs, residual, signal):
    # Predictive-oriented score: prioritizes reconstruction quality, stationarity, 
    # autocorrelation, and optimal IMF count (~6) for time series forecasting
    n_imfs = imfs.shape[0]

    reconstructed = np.sum(imfs, axis=0) + residual
    reconstruction_error = np.mean((signal - reconstructed) ** 2)
    signal_power = np.var(signal) if np.var(signal) > 0 else 1e-8
    normalized_error = reconstruction_error / signal_power

    imf_quality = 0
    for i, imf in enumerate(imfs):
        imf_variance = np.var(imf) if np.var(imf) > 0 else 1e-8
        windowed_vars = []
        window_size = min(50, len(imf) // 10)
        for j in range(0, len(imf) - window_size, window_size):
            windowed_vars.append(np.var(imf[j:j + window_size]))

        if len(windowed_vars) > 1:
            variance_stability = np.std(windowed_vars) / np.mean(windowed_vars) if np.mean(windowed_vars) > 0 else 1
        else:
            variance_stability = 0

        if len(imf) > 10:
            autocorr_1 = np.corrcoef(imf[:-1], imf[1:])[0, 1] if not np.isnan(
                np.corrcoef(imf[:-1], imf[1:])[0, 1]) else 0
            autocorr_quality = abs(autocorr_1)
        else:
            autocorr_quality = 0

        imf_usefulness = variance_stability + (1 - autocorr_quality)
        imf_quality += imf_usefulness

    imf_quality = imf_quality / n_imfs if n_imfs > 0 else 0

    target_imfs = 6
    complexity_penalty = abs(n_imfs - target_imfs) / target_imfs

    if n_imfs > 10:
        complexity_penalty += (n_imfs - 10) * 0.1

    total_signal_energy = np.sum(signal ** 2)
    imf_energies = [np.sum(imf ** 2) for imf in imfs]
    residual_energy = np.sum(residual ** 2)
    total_decomp_energy = sum(imf_energies) + residual_energy

    energy_preservation = abs(
        total_signal_energy - total_decomp_energy) / total_signal_energy if total_signal_energy > 0 else 0

    # Weighted score optimized for forecasting models (BiGRU, LSTM, etc.)
    # Weights: 40% reconstruction + 30% IMF usefulness + 20% complexity + 10% energy preservation
    predictive_score = (
            0.4 * normalized_error +
            0.3 * imf_quality +
            0.2 * complexity_penalty +
            0.1 * energy_preservation
    )

    return predictive_score


def calculate_comprehensive_score(imfs, residual, signal):
    return calculate_predictive_quality_score(imfs, residual, signal)


def bayesian_optimize_epsilon_enhanced(signal, trials=100, seed=42, optimization_strategy='predictive'):
    # Enhanced Bayesian optimization with smart features:
    # 1. Reference value pre-testing to find good starting point
    # 2. Adaptive search space centered on best reference
    # 3. Multiple runs per epsilon for stability (reduces random fluctuation)
    # 4. Intelligent fallback if improvement < 5%
    if not SKOPT_AVAILABLE:
        return None, 'disabled'

    actual_strategy = optimization_strategy

    # Step 1: Test reference values to identify promising region
    reference_epsilons = [0.2, 0.3, 0.4, 0.5, 0.6]
    reference_results = {}

    for eps in reference_epsilons:
        decomposer = ReproducibleCEEMDAN(trials=trials, epsilon=eps, seed=seed)
        scores = []
        for _ in range(2):
            imfs, residual = decomposer.decompose(signal)
            score = calculate_predictive_quality_score(imfs, residual, signal)
            scores.append(score)
        reference_results[eps] = np.mean(scores)

    best_reference_epsilon = min(reference_results, key=reference_results.get)
    best_reference_score = reference_results[best_reference_epsilon]

    # Step 2: Adaptive search space centered on best reference (±0.15 range)
    search_lower = max(0.05, best_reference_epsilon - 0.15)
    search_upper = min(1.0, best_reference_epsilon + 0.15)

    def objective_predictive(epsilon):
        # Run twice per epsilon and average to reduce random noise impact
        epsilon = max(0.05, min(1.0, epsilon[0]))
        decomposer = ReproducibleCEEMDAN(trials=trials, epsilon=epsilon, seed=seed)

        scores = []
        for _ in range(2):
            imfs, residual = decomposer.decompose(signal)
            score = calculate_predictive_quality_score(imfs, residual, signal)
            scores.append(score)

        return np.mean(scores)

    space = [Real(search_lower, search_upper, name='epsilon')]

    try:
        # Step 3: Bayesian optimization with 25 calls for thorough search
        result = gp_minimize(
            objective_predictive,
            space,
            n_calls=25,
            n_initial_points=6,
            random_state=seed,
            acq_func='EI',
            n_jobs=1,
            verbose=False
        )

        optimal_epsilon = result.x[0]
        optimal_score = result.fun

        # Step 4: Intelligent fallback - use reference value if improvement < 5%
        improvement = (best_reference_score - optimal_score) / best_reference_score
        if improvement < 0.05:
            optimal_epsilon = best_reference_epsilon

    except Exception as e:
        optimal_epsilon = best_reference_epsilon

    return optimal_epsilon, actual_strategy


def calculate_quality_metrics(imfs, residual, signal):
    n_imfs = imfs.shape[0]

    reconstructed = np.sum(imfs, axis=0) + residual
    reconstruction_error = np.sqrt(np.mean((signal - reconstructed) ** 2))
    signal_std = np.std(signal) if np.std(signal) > 0 else 1e-8
    rrmse = reconstruction_error / signal_std

    oi = calculate_orthogonality_index(imfs)

    signal_energy = np.sum(signal ** 2)

    imf_energies = []
    for i, imf in enumerate(imfs):
        energy = np.sum(imf ** 2)
        imf_energies.append(energy)

    residual_energy = np.sum(residual ** 2)

    return {
        'num_imfs': n_imfs,
        'rrmse': rrmse,
        'orthogonality_index': oi,
        'signal_energy': signal_energy,
        'imf_energies': imf_energies,
        'residual_energy': residual_energy
    }


def save_results(imfs, residual, signal, csv_file, output_dir, metrics, optimal_epsilon, optimization_strategy):
    file_base_name = os.path.splitext(os.path.basename(csv_file))[0]

    result_data = {'Original_Signal': signal}

    for i, imf in enumerate(imfs):
        result_data[f'IMF_{i + 1}'] = imf

    result_data['Residuals_After_Smoothing'] = residual

    result_df = pd.DataFrame(result_data)
    result_output_path = os.path.join(output_dir, f"{file_base_name}_CEEMDAN_Laplace_optimized_decomposition.csv")
    result_df.to_csv(result_output_path, index=False, encoding='utf-8-sig')

    energy_data = {
        'Component': [f'IMF_{i + 1}' for i in range(len(imfs))] + ['Residuals_After_Smoothing', 'Total'],
        'Energy': metrics['imf_energies'] + [metrics['residual_energy'], sum(metrics['imf_energies']) + metrics['residual_energy']],
        'Energy_Percentage': [(e / metrics['signal_energy'] * 100) for e in metrics['imf_energies']] +
                             [metrics['residual_energy'] / metrics['signal_energy'] * 100,
                              (sum(metrics['imf_energies']) + metrics['residual_energy']) / metrics['signal_energy'] * 100]
    }

    energy_df = pd.DataFrame(energy_data)
    energy_output_path = os.path.join(output_dir, f"{file_base_name}_energy_distribution_optimized.csv")
    energy_df.to_csv(energy_output_path, index=False, encoding='utf-8-sig')


def process_single_csv(csv_file, output_dir, trials=100, seed=42, enable_optimization=True,
                       optimization_strategy='predictive'):
    # Main processing pipeline: optimize epsilon → decompose → evaluate → save results
    try:
        df = pd.read_csv(csv_file)

        if 'speed(m/s)' not in df.columns:
            return None

        wind_speed = df['speed(m/s)'].dropna().values

        if len(wind_speed) < 100:
            return None

        if enable_optimization and SKOPT_AVAILABLE:
            optimal_epsilon, optimization_strategy = bayesian_optimize_epsilon_enhanced(
                wind_speed, trials=trials, seed=seed,
                optimization_strategy=optimization_strategy
            )

            if optimal_epsilon is None:
                optimal_epsilon = 0.4
                optimization_strategy = 'default'
        else:
            optimal_epsilon = 0.4
            optimization_strategy = 'default'

        decomposer = ReproducibleCEEMDAN(trials=trials, epsilon=optimal_epsilon, seed=seed)
        imfs, residual = decomposer.decompose(wind_speed)

        metrics = calculate_quality_metrics(imfs, residual, wind_speed)

        predictive_score = calculate_predictive_quality_score(imfs, residual, wind_speed)

        save_results(imfs, residual, wind_speed, csv_file, output_dir, metrics, optimal_epsilon, optimization_strategy)

        quality_data = {
            'Metric': [
                'Number of IMFs',
                'Optimized Epsilon',
                'Optimization Strategy',
                'Reconstruction Error (RRMSE)',
                'Orthogonality Index (OI)',
                'Predictive Quality Score',
                'Signal Energy',
                'Total Decomposition Energy',
                'Energy Preservation Ratio',
                'Noise Type',
                'Bayesian Optimization Status'
            ],
            'Value': [
                metrics['num_imfs'],
                f"{optimal_epsilon:.6f}",
                optimization_strategy,
                f"{metrics['rrmse']:.6e}",
                f"{metrics['orthogonality_index']:.6f}",
                f"{predictive_score:.6f}",
                f"{metrics['signal_energy']:.6e}",
                f"{(sum(metrics['imf_energies']) + metrics['residual_energy']):.6e}",
                f"{((sum(metrics['imf_energies']) + metrics['residual_energy']) / metrics['signal_energy']):.6f}",
                'Laplace_White_Noise',
                'Enabled' if (enable_optimization and SKOPT_AVAILABLE) else 'Disabled'
            ]
        }

        quality_df = pd.DataFrame(quality_data)
        quality_output_path = os.path.join(output_dir, f"{file_base_name}_quality_metrics_optimized.csv")
        quality_df.to_csv(quality_output_path, index=False, encoding='utf-8-sig')

        return {
            'file_name': os.path.basename(csv_file),
            'data_length': len(wind_speed),
            'num_imfs': metrics['num_imfs'],
            'optimized_epsilon': optimal_epsilon,
            'optimization_strategy': optimization_strategy,
            'orthogonality_index': metrics['orthogonality_index'],
            'rrmse': metrics['rrmse'],
            'predictive_quality_score': predictive_score,
            'signal_energy': metrics['signal_energy'],
            'noise_type': 'Laplace_White_Noise',
            'bayesian_optimization': 'Enabled' if (enable_optimization and SKOPT_AVAILABLE) else 'Disabled',
            'processing_status': 'Success'
        }

    except Exception as e:
        return {
            'file_name': os.path.basename(csv_file),
            'processing_status': 'Error',
            'error_message': str(e),
            'bayesian_optimization': 'N/A',
            'optimization_strategy': 'N/A'
        }


def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, f"CEEMDAN_Laplace_Optimized_Results_{timestamp}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def main():
    # Batch process all CSV files with optimized CEEMDAN decomposition
    start_time = datetime.now()

    output_dir = create_output_directory()

    current_dir = os.getcwd()
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))

    if not csv_files:
        return

    if SKOPT_AVAILABLE:
        optimization_strategy = 'predictive'
    else:
        optimization_strategy = 'disabled'

    trials = 100
    seed = 42
    enable_optimization = True

    results = []
    successful_files = 0

    for csv_file in csv_files:
        result = process_single_csv(csv_file, output_dir, trials=trials, seed=seed,
                                    enable_optimization=enable_optimization,
                                    optimization_strategy=optimization_strategy)
        if result:
            results.append(result)
            if result['processing_status'] == 'Success':
                successful_files += 1

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(output_dir, "processing_summary_optimized.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    readme_path = os.path.join(output_dir, "00_README_分析说明.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("CEEMDAN风速分解工具 - 拉普拉斯白噪声版本（超强贝叶斯优化）\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CEEMDAN trials: {trials}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"贝叶斯优化: {'启用' if SKOPT_AVAILABLE else '未启用'}\n")
        f.write(f"优化策略: {optimization_strategy}\n")
        f.write(f"噪声类型: 拉普拉斯白噪声\n\n")

        f.write("【超强贝叶斯优化】核心特性:\n")
        f.write("-" * 50 + "\n")
        if optimization_strategy == 'predictive':
            f.write("预测导向策略 (专为时序预测优化):\n")
            f.write("• 目标: 获得最适合预测任务的IMF分解\n")
            f.write("• 重构质量权重: 40% (确保信息无损)\n")
            f.write("• IMF预测有用性: 30% (平稳性+自相关性)\n")
            f.write("• 复杂度控制: 20% (目标6个IMF，避免过度分解)\n")
            f.write("• 信息保留度: 10% (能量守恒检验)\n\n")

            f.write("智能优化流程:\n")
            f.write("1. 参考值预测试: 测试[0.2,0.3,0.4,0.5,0.6]找最佳\n")
            f.write("2. 自适应搜索: 以最佳参考值为中心±0.15范围搜索\n")
            f.write("3. 稳定性保证: 每个epsilon运行2次取平均\n")
            f.write("4. 智能回退: 改进<5%时采用参考最佳值\n")
            f.write("5. 充分搜索: 25次贝叶斯调用，6个初始点\n")

        elif optimization_strategy == 'comprehensive':
            f.write("综合评分策略:\n")
            f.write("• 降低OI权重，避免过度追求正交性\n")
            f.write("• 重视重构误差和IMF数量合理性\n")
        elif optimization_strategy == 'balanced':
            f.write("平衡策略: OI(20%) + RRMSE(50%) + IMF惩罚(30%)\n")
        elif optimization_strategy == 'oi_only':
            f.write("仅OI优化策略 (可能导致过度分解)\n")
        else:
            f.write("未启用贝叶斯优化，使用默认epsilon=0.4\n")

        f.write(f"\n处理结果:\n")
        f.write("-" * 20 + "\n")
        f.write(f"总文件数: {len(csv_files)}\n")
        f.write(f"成功处理: {successful_files}\n")
        f.write(f"处理失败: {len(csv_files) - successful_files}\n")

        f.write(f"\n输出文件说明:\n")
        f.write("-" * 20 + "\n")
        f.write("*_CEEMDAN_Laplace_optimized_decomposition.csv - 详细的IMF分解结果\n")
        f.write("*_energy_distribution_optimized.csv - 能量分布分析\n")
        f.write("*_quality_metrics_optimized.csv - 质量指标评估(含预测质量评分)\n")
        f.write("processing_summary_optimized.csv - 所有文件的处理摘要\n")
        f.write("00_README_分析说明.txt - 本说明文件\n")

    end_time = datetime.now()
    processing_time = end_time - start_time

    print("Finish!")


if __name__ == "__main__":
    main()
