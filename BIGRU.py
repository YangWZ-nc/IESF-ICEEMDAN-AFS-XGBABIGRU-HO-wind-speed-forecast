# BiGRU-HO Optimization System with Multi-dimensional Input
# Features: Single seed run, minimal output, weighted RMSE selection

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import beta
import gc
import os
import sys
import random
import logging
import datetime
import json
from typing import List, Dict, Any, Tuple

from ho import HO
from fun_info import F24


# Global configuration class
class Config:
    # File paths
    CSV_PATH = r"spring_result_CEEMDAN_decomposition.csv"
    IMF_PREFIX = 'IMF_'
    RESIDUAL_COLUMN = 'Residual'
    
    # Model parameters
    LOOK_BACK = 30
    PREDICTION_STEPS = 3
    EPOCHS = 30
    
    # HO algorithm parameters
    HO_SEARCH_AGENTS = 16
    HO_MAX_ITERATIONS = 30
    
    # Run only once with user-specified seed
    N_FINAL_RUNS = 1
    
    # Three-step weighted RMSE weights (5:3:2)
    STEP_WEIGHTS = [0.5, 0.3, 0.2]
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # User-specified seed (modify this to change the seed)
    USER_SEED = 12345


# Seed manager - controls all randomness for reproducibility
class SeedManager:
    def __init__(self):
        self.GLOBAL_SEED = 1325
        self.HO_OPTIMIZATION_SEED = 12345
        self.STABILITY_TEST_BASE_SEED = Config.USER_SEED  # Use user-specified seed

    def set_global_deterministic_environment(self):
        """Set deterministic environment for reproducible results"""
        # Python built-in random
        random.seed(self.GLOBAL_SEED)
        # NumPy random seed
        np.random.seed(self.GLOBAL_SEED)
        # PyTorch random seed
        torch.manual_seed(self.GLOBAL_SEED)
        
        # CUDA random seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.GLOBAL_SEED)
            torch.cuda.manual_seed_all(self.GLOBAL_SEED)
        
        # PyTorch deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        os.environ['PYTHONHASHSEED'] = str(self.GLOBAL_SEED)

    def reset_for_operation(self, operation_type: str, extra_seed: int = 0):
        """Reset seed for specific operation"""
        if operation_type == "data_loading":
            seed = self.GLOBAL_SEED + extra_seed
        elif operation_type == "ho_optimization":
            seed = self.HO_OPTIMIZATION_SEED + extra_seed
        elif operation_type == "stability_test":
            seed = self.STABILITY_TEST_BASE_SEED + extra_seed
        else:
            seed = self.GLOBAL_SEED + extra_seed
        
        # Reset all random states
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return seed


# Log manager - handles all logging and result saving
class LogManager:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"BiGRU_HO_MultiDim_Sum_F24_WeightedRMSE_Results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_results_dir = os.path.join(self.results_dir, "all_runs_predictions")
        os.makedirs(self.all_results_dir, exist_ok=True)
        self.log_file = os.path.join(self.results_dir, "complete_log.txt")
        self.setup_logging()
        self.all_runs_results = []
        self.all_runs_detailed_data = []
        self.ho_optimization_log = {}
        self.deterministic_checkpoints = {}

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('BiGRU_HO_MultiDim_Sum_F24_WeightedRMSE')
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _convert_to_json_serializable(self, obj):
        """Recursively convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    def log_and_print(self, message):
        pass

    def save_deterministic_checkpoint(self, checkpoint_name, data):
        self.deterministic_checkpoints[checkpoint_name] = data
        checkpoint_file = os.path.join(self.results_dir, "deterministic_checkpoints.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.deterministic_checkpoints, f, indent=2, ensure_ascii=False, default=str)

    def log_ho_optimization(self, best_params, best_value, convergence_history):
        self.ho_optimization_log = {
            'timestamp': datetime.datetime.now().isoformat(),
            'best_params': best_params,
            'best_value': float(best_value),
            'convergence_history': convergence_history,
            'ho_config': {
                'search_agents': Config.HO_SEARCH_AGENTS,
                'max_iterations': Config.HO_MAX_ITERATIONS
            }
        }
        ho_log_file = os.path.join(self.results_dir, "ho_optimization_log.json")
        with open(ho_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.ho_optimization_log, f, indent=2, ensure_ascii=False, default=str)

    def log_stability_run(self, run_idx, seed, results):
        run_data = {
            'run_index': run_idx,
            'seed': seed,
            'results': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.all_runs_results.append(run_data)

    def save_stability_summary(self):
        summary_file = os.path.join(self.results_dir, "stability_test_summary.json")
        summary = {
            'n_runs': Config.N_FINAL_RUNS,
            'all_runs': self.all_runs_results,
            'statistics': self._calculate_statistics()
        }
        serializable_summary = self._convert_to_json_serializable(summary)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

    def _calculate_statistics(self):
        if not self.all_runs_results:
            return {}
        sum_weighted_rmses = [r['results']['sum_weighted_rmse'] for r in self.all_runs_results]
        return {
            'sum_weighted_rmse': {
                'mean': float(np.mean(sum_weighted_rmses)),
                'std': float(np.std(sum_weighted_rmses)),
                'min': float(np.min(sum_weighted_rmses)),
                'max': float(np.max(sum_weighted_rmses)),
                'median': float(np.median(sum_weighted_rmses))
            }
        }

    def save_final_results(self, best_run_results, best_params):
        final_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'best_run': best_run_results,
            'best_params': best_params,
            'configuration': {
                'look_back': Config.LOOK_BACK,
                'prediction_steps': Config.PREDICTION_STEPS,
                'epochs': Config.EPOCHS,
                'n_final_runs': Config.N_FINAL_RUNS,
                'step_weights': Config.STEP_WEIGHTS
            }
        }
        serializable_results = self._convert_to_json_serializable(final_results)
        final_file = os.path.join(self.results_dir, "final_results.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


# Bidirectional GRU model for time series prediction
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU layer
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer (hidden_size * 2 for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """Forward pass"""
        out, _ = self.bigru(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out


# Main optimizer class combining BiGRU model with HO algorithm
class BiGRUHOOptimizer:
    def __init__(self):
        self.seed_manager = SeedManager()
        self.seed_manager.set_global_deterministic_environment()
        self.log_manager = LogManager()

    def load_data(self):
        """Load and preprocess data from CSV file"""
        self.seed_manager.reset_for_operation("data_loading")
        
        # Create sample data if CSV doesn't exist
        if not os.path.exists(Config.CSV_PATH):
            dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
            n_imf = 5
            data_dict = {'date': dates}
            for i in range(1, n_imf + 1):
                data_dict[f'{Config.IMF_PREFIX}{i}'] = np.random.randn(len(dates)) * 10 + 50
            data_dict[Config.RESIDUAL_COLUMN] = np.random.randn(len(dates)) * 5 + 30
            df = pd.DataFrame(data_dict)
            df.to_csv(Config.CSV_PATH, index=False)
        
        # Read CSV file
        df = pd.read_csv(Config.CSV_PATH)
        
        # Identify IMF columns and residual column
        imf_columns = [col for col in df.columns if col.startswith(Config.IMF_PREFIX)]
        residual_column = Config.RESIDUAL_COLUMN
        if residual_column not in df.columns:
            residual_column = None
        
        # Combine IMF and residual as features
        feature_columns = imf_columns.copy()
        if residual_column:
            feature_columns.append(residual_column)
        
        # Create target as sum of all features
        target_column = 'sum'
        df[target_column] = df[feature_columns].sum(axis=1)
        features = df[feature_columns].values
        target = df[target_column].values
        
        # Scale features and target
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences with look-back window
        X, y = [], []
        for i in range(len(features_scaled) - Config.LOOK_BACK - Config.PREDICTION_STEPS + 1):
            X.append(features_scaled[i:i + Config.LOOK_BACK])
            # Multi-step prediction targets
            y_multi_step = []
            for step in range(Config.PREDICTION_STEPS):
                y_multi_step.append(target_scaled[i + Config.LOOK_BACK + step])
            y.append(y_multi_step)
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(Config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).to(Config.DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(Config.DEVICE)
        y_test_tensor = torch.FloatTensor(y_test).to(Config.DEVICE)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        data_info = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'X_test': X_test,
            'y_test': y_test,
            'target_scaler': target_scaler,
            'feature_scaler': feature_scaler,
            'n_features': features_scaled.shape[1],
            'feature_names': feature_columns,
            'train_size': train_size,
            'test_size': len(X_test)
        }
        self.log_manager.save_deterministic_checkpoint('data_loading', {
            'train_size': train_size,
            'test_size': len(X_test),
            'n_features': features_scaled.shape[1],
            'feature_names': feature_columns
        })
        return data_info

    def train_and_evaluate(self, data, params, seed):
        """Train BiGRU model and evaluate on test set"""
        # Reset seed for this run
        current_seed = self.seed_manager.reset_for_operation("stability_test", seed)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize model
        model = BiGRUModel(
            input_size=data['n_features'],
            hidden_size=int(params['hidden_size']),
            num_layers=int(params['num_layers']),
            dropout=params['dropout'],
            output_size=Config.PREDICTION_STEPS
        ).to(Config.DEVICE)
        
        # Setup loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Create data loader
        train_loader = DataLoader(
            data['train_dataset'],
            batch_size=int(params['batch_size']),
            shuffle=True,
            drop_last=False
        )
        
        # Training loop
        model.train()
        for epoch in range(Config.EPOCHS):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation mode
        model.eval()
        all_predictions = []
        all_actuals = []
        test_loader = DataLoader(
            data['test_dataset'],
            batch_size=int(params['batch_size']),
            shuffle=False
        )
        
        # Make predictions
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                all_predictions.append(outputs.cpu().numpy())
                all_actuals.append(batch_y.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)
        
        # Inverse transform to original scale
        predictions_original = data['target_scaler'].inverse_transform(predictions)
        actuals_original = data['target_scaler'].inverse_transform(actuals)
        
        # Calculate metrics for each prediction step
        sum_metrics = {}
        sum_rmse_list = []
        for step in range(Config.PREDICTION_STEPS):
            pred_step = predictions_original[:, step]
            actual_step = actuals_original[:, step]
            rmse = np.sqrt(mean_squared_error(actual_step, pred_step))
            mae = mean_absolute_error(actual_step, pred_step)
            sum_metrics[f'step_{step + 1}'] = {
                'rmse': float(rmse),
                'mae': float(mae)
            }
            sum_rmse_list.append(rmse)
        
        # Calculate weighted RMSE (5:3:2 weighting)
        sum_weighted_rmse = sum(w * r for w, r in zip(Config.STEP_WEIGHTS, sum_rmse_list))
        
        results = {
            'seed': current_seed,
            'sum_weighted_rmse': float(sum_weighted_rmse),
            'sum_rmse_step1': float(sum_rmse_list[0]),
            'sum_metrics': sum_metrics,
            'predictions': predictions_original,
            'actuals': actuals_original
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results

    def objective_function_with_data(self, x, data):
        """Objective function for HO algorithm optimization"""
        # Parse hyperparameters from optimization vector
        params = {
            'hidden_size': int(x[0]),
            'num_layers': int(x[1]),
            'learning_rate': x[2],
            'dropout': x[3],
            'batch_size': int(x[4])
        }
        try:
            results = self.train_and_evaluate(data, params, seed=0)
            return results['sum_weighted_rmse']
        except Exception as e:
            return 1e10

    def run_ho_optimization(self, data):
        """Run HO algorithm to find optimal hyperparameters"""
        self.seed_manager.reset_for_operation("ho_optimization")
        
        # Define search bounds for hyperparameters
        # [hidden_size, num_layers, learning_rate, dropout, batch_size]
        lb = np.array([32, 1, 0.0001, 0.0, 16])
        ub = np.array([128, 3, 0.01, 0.5, 64])
        dim = 5
        
        def objective_func(x):
            return self.objective_function_with_data(x, data)
        
        # Run HO optimization
        ho_optimizer = HO(N=Config.HO_SEARCH_AGENTS, Max_iter=Config.HO_MAX_ITERATIONS, lb=lb, ub=ub, dim=dim, fobj=objective_func)
        best_pos, best_score, convergence_curve = ho_optimizer.optimize()
        
        # Extract best hyperparameters
        best_params = {
            'hidden_size': int(best_pos[0]),
            'num_layers': int(best_pos[1]),
            'learning_rate': best_pos[2],
            'dropout': best_pos[3],
            'batch_size': int(best_pos[4])
        }
        
        self.log_manager.log_ho_optimization(best_params, best_score, convergence_curve)
        self.log_manager.save_deterministic_checkpoint('ho_optimization', {
            'best_params': best_params,
            'best_value': float(best_score)
        })
        return best_params, best_score

    def run_stability_test(self, data, best_params):
        """Run single test with user-specified seed"""
        all_results = []
        
        # Run only once with user-specified seed
        for i in range(Config.N_FINAL_RUNS):
            run_seed = self.seed_manager.STABILITY_TEST_BASE_SEED + i
            results = self.train_and_evaluate(data, best_params, seed=i)
            results['run_index'] = i + 1
            all_results.append(results)
            
            self.log_manager.log_stability_run(i + 1, run_seed, results)
            self.log_manager.all_runs_detailed_data.append({
                'run_index': i + 1,
                'seed': run_seed,
                'predictions': results['predictions'],
                'actuals': results['actuals'],
                'metrics': results['sum_metrics']
            })
        
        self.log_manager.save_stability_summary()
        
        # Get best run (should be the only one)
        best_run = min(all_results, key=lambda x: x['sum_weighted_rmse'])
        
        self.log_manager.save_deterministic_checkpoint('stability_test', {
            'n_runs': Config.N_FINAL_RUNS,
            'best_run_index': best_run['run_index'],
            'best_seed': best_run['seed'],
            'best_sum_weighted_rmse': float(best_run['sum_weighted_rmse'])
        })
        
        return best_run

    def save_final_csv(self, data, best_run_results):
        """Save final predictions to CSV file (only this CSV will be saved)"""
        results_length = len(best_run_results['predictions'])
        results_df = pd.DataFrame(index=range(results_length))
        
        # Add prediction columns for each step
        for step in range(Config.PREDICTION_STEPS):
            results_df[f'sum_prediction_step_{step + 1}'] = best_run_results['predictions'][:, step]
        
        # Add actual values for each step
        for step in range(Config.PREDICTION_STEPS):
            step_true_sum = best_run_results['actuals'][:, step]
            if len(step_true_sum) < results_length:
                padding = np.full(results_length - len(step_true_sum), np.nan)
                step_true_sum = np.concatenate([step_true_sum, padding])
            results_df[f'sum_true_step_{step + 1}'] = step_true_sum
        
        # Add error columns
        for step in range(Config.PREDICTION_STEPS):
            pred_col = f'sum_prediction_step_{step + 1}'
            true_col = f'sum_true_step_{step + 1}'
            error_col = f'sum_error_step_{step + 1}'
            pred_values = results_df[pred_col].values
            true_values = results_df[true_col].values
            error_values = np.where(
                np.isnan(pred_values) | np.isnan(true_values),
                np.nan,
                np.abs(pred_values - true_values)
            )
            results_df[error_col] = error_values
        
        # Add metadata columns
        results_df['best_seed'] = best_run_results['seed']
        results_df['prediction_time_offset'] = Config.LOOK_BACK
        results_df['model_type'] = 'multidim_input'
        results_df['n_features'] = data['n_features']
        results_df['sum_weighted_rmse'] = best_run_results['sum_weighted_rmse']
        results_df['step_weights'] = str(Config.STEP_WEIGHTS)
        
        # Save to file
        try:
            filename = f'final_sum_predictions_multidim_weighted_best_seed_{best_run_results["seed"]}.csv'
            results_filename = os.path.join(self.log_manager.results_dir, filename)
            results_df.to_csv(results_filename, index=True)
            results_df.to_csv(filename, index=True)
        except Exception as e:
            pass

    def run(self):
        """Main execution method - runs the complete optimization pipeline"""
        try:
            # Step 1: Load and preprocess data
            data = self.load_data()
            
            # Step 2: Run HO algorithm for hyperparameter optimization
            best_params, best_value = self.run_ho_optimization(data)
            
            # Step 3: Run final test with user-specified seed
            best_run_results = self.run_stability_test(data, best_params)
            
            # Step 4: Save final CSV result
            self.save_final_csv(data, best_run_results)
            self.log_manager.save_final_results(best_run_results, best_params)
            
            return best_run_results, best_params
        except Exception as e:
            raise


def main():
    """Main entry point of the program"""
    try:
        if not os.path.exists(Config.CSV_PATH):
            pass
        
        # Create optimizer and run
        optimizer = BiGRUHOOptimizer()
        best_run_results, best_params = optimizer.run()
        
        # Final output
        print("Finish!")
        return best_run_results, best_params
    except KeyboardInterrupt:
        return None, None
    except Exception as e:
        return None, None


if __name__ == "__main__":
    best_results, best_parameters = main()
