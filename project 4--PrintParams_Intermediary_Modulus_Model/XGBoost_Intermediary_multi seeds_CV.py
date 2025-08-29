import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import openpyxl
import joblib  # For saving models
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error


class Logger:
    """Redirect output to both console and file"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        # Specify file encoding as utf-8
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_unique_filename(base_name):
    """Generate a unique filename, adding sequential suffix if it exists"""
    if not os.path.exists(base_name):
        return base_name

    directory, full_name = os.path.split(base_name)
    name, ext = os.path.splitext(full_name)

    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def run_mediation_experiment(seed, data, param_grid):
    """Run mediation effect experiment and return results"""
    np.random.seed(seed)

    # Define variables
    # Independent variables: printing parameters (3)
    predictors_names = ['printing_temperature', 'feed_rate', 'printing_speed']  # T (temperature), V (speed), F (feed rate)
    # Mediator variables: width and height
    mediator_names = ['Width', 'Height']
    # Dependent variable: mechanical modulus
    target_name = 'Experiment_mean(MPa)'

    # Dataset splitting: stratified sampling based on mechanical modulus
    # Training set (70%), validation set (15%), test set (15%)
    # Dynamically adjust number of bins to ensure at least 4 samples per bin (for two stratified samplings)
    min_samples_per_bin = 4  # Ensure at least 4 samples per bin
    max_bins = 10

    # Calculate maximum possible number of bins
    max_possible_bins = len(data) // min_samples_per_bin
    num_bins = min(max_bins, max_possible_bins)

    # Ensure at least 2 bins
    num_bins = max(2, num_bins)

    # Create bins
    data['target_bin'] = pd.cut(data[target_name], bins=num_bins, labels=False)

    # Check sample count per bin, merge adjacent bins if any has insufficient samples
    bin_counts = data['target_bin'].value_counts().sort_index()
    while (bin_counts < min_samples_per_bin).any():
        # Find bin with fewest samples
        min_bin = bin_counts.idxmin()
        # Merge with adjacent bin
        if min_bin == 0:
            data['target_bin'] = data['target_bin'].replace(1, 0)
        elif min_bin == len(bin_counts) - 1:
            data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
        else:
            # Merge with neighboring bin that has more samples
            left_count = bin_counts[min_bin - 1]
            right_count = bin_counts[min_bin + 1]
            if left_count >= right_count:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
            else:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin + 1)
        # Recalculate bin counts
        bin_counts = data['target_bin'].value_counts().sort_index()
        # Rename bin labels to ensure continuity
        data['target_bin'] = pd.Categorical(data['target_bin']).codes
        bin_counts = data['target_bin'].value_counts().sort_index()

        # If only one bin remains, can't merge further, break loop
        if len(bin_counts) == 1:
            break

    # Output bin information to verify each bin has at least 4 samples
    print(f"Stratified sampling verification:")
    print(f"- Number of bins: {len(bin_counts)}")
    for bin_idx in bin_counts.index:
        print(f"  Bin {bin_idx}: {bin_counts[bin_idx]} samples")
    assert (bin_counts >= min_samples_per_bin).all() or len(bin_counts) == 1, \
        "Some bins have fewer than 4 samples, please check data or adjust binning strategy"

    # If all data is in one bin, use random sampling instead of stratified sampling
    stratify_param = data['target_bin'] if len(bin_counts) > 1 else None

    # First stratified sampling: split into training set and temporary set
    train_data, temp_data = train_test_split(
        data,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_param  # Stratify based on target variable
    )

    # Prepare stratification parameter for second sampling
    if stratify_param is not None:
        stratify_param_temp = temp_data['target_bin']
        # Check sample count per bin in temporary set
        temp_bin_counts = stratify_param_temp.value_counts()
        # If any bin has fewer than 2 samples, use random sampling
        if (temp_bin_counts < 2).any():
            stratify_param_temp = None
    else:
        stratify_param_temp = None

    # Second stratified sampling: split temporary set into validation and test sets
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_param_temp  # Stratify based on target variable
    )

    # Remove auxiliary column
    train_data = train_data.drop('target_bin', axis=1)
    val_data = val_data.drop('target_bin', axis=1)
    test_data = test_data.drop('target_bin', axis=1)

    # --------------------------
    # 1. Mediation model (nested):
    #    First layer: 3 printing parameters → width and height
    #    Second layer: 3 printing parameters + predicted width and height → mechanical modulus (5 features total)
    # --------------------------

    # 1.1 Step 1: Predict width and height using 3 printing parameters (first layer)
    # Verify first layer has 3 input features
    assert len(predictors_names) == 3, f"Mediation model first layer input feature count error: should be 3, actual {len(predictors_names)}"

    # Predict width
    xgb_width = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_width = GridSearchCV(xgb_width, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_width.fit(train_data[predictors_names], train_data['Width'])
    best_width = grid_width.best_estimator_

    # Predict height
    xgb_height = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_height = GridSearchCV(xgb_height, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_height.fit(train_data[predictors_names], train_data['Height'])
    best_height = grid_height.best_estimator_

    # Generate predicted width and height, combine with original printing parameters as second layer features (5 features total)
    # Training set features
    train_data['predicted_Width'] = best_width.predict(train_data[predictors_names])
    train_data['predicted_Height'] = best_height.predict(train_data[predictors_names])
    mediation_train_features = train_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # Verify feature count is 5
    assert mediation_train_features.shape[1] == 5, \
        f"Mediation model second layer training feature count error: should be 5, actual {mediation_train_features.shape[1]}"

    # Validation set features
    val_data['predicted_Width'] = best_width.predict(val_data[predictors_names])
    val_data['predicted_Height'] = best_height.predict(val_data[predictors_names])
    mediation_val_features = val_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # Verify feature count is 5
    assert mediation_val_features.shape[1] == 5, \
        f"Mediation model second layer validation feature count error: should be 5, actual {mediation_val_features.shape[1]}"

    # Test set features
    test_data['predicted_Width'] = best_width.predict(test_data[predictors_names])
    test_data['predicted_Height'] = best_height.predict(test_data[predictors_names])
    mediation_test_features = test_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # Verify feature count is 5
    assert mediation_test_features.shape[1] == 5, \
        f"Mediation model second layer test feature count error: should be 5, actual {mediation_test_features.shape[1]}"

    # 1.2 Step 2: Predict mechanical modulus using 5 features (second layer)
    xgb_mediation = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_mediation = GridSearchCV(xgb_mediation, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_mediation.fit(mediation_train_features, train_data[target_name])
    best_mediation = grid_mediation.best_estimator_

    # Predict on each dataset
    y_pred_val_mediation = best_mediation.predict(mediation_val_features)
    y_pred_test_mediation = best_mediation.predict(mediation_test_features)

    # --------------------------
    # 2. Direct model: 3 printing parameters directly predict mechanical modulus
    # --------------------------
    # Verify input feature count is 3
    assert len(predictors_names) == 3, f"Direct model input feature count error: should be 3, actual {len(predictors_names)}"

    xgb_direct = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_direct = GridSearchCV(xgb_direct, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_direct.fit(train_data[predictors_names], train_data[target_name])
    best_direct = grid_direct.best_estimator_

    # Predict on each dataset
    y_pred_val_direct = best_direct.predict(val_data[predictors_names])
    y_pred_test_direct = best_direct.predict(test_data[predictors_names])

    # --------------------------
    # 3. Hybrid model: 3 printing parameters + 2 actual width/height → mechanical modulus (5 features total)
    # --------------------------
    hybrid_features = predictors_names + mediator_names  # T, V, F, W_true, H_true

    # Verify input feature count is 5
    assert len(hybrid_features) == 5, f"Hybrid model input feature count error: should be 5, actual {len(hybrid_features)}"

    xgb_hybrid = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_hybrid = GridSearchCV(xgb_hybrid, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_hybrid.fit(train_data[hybrid_features], train_data[target_name])
    best_hybrid = grid_hybrid.best_estimator_

    # Predict on each dataset
    y_pred_val_hybrid = best_hybrid.predict(val_data[hybrid_features])
    y_pred_test_hybrid = best_hybrid.predict(test_data[hybrid_features])

    # --------------------------
    # Calculate evaluation metrics
    # --------------------------
    results = {
        'mediation': {  # Mediation model: 3 printing parameters→5 features→mechanical modulus
            'val': {
                'MSE': mean_squared_error(val_data[target_name], y_pred_val_mediation),
                'R2': r2_score(val_data[target_name], y_pred_val_mediation)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target_name], y_pred_test_mediation),
                'R2': r2_score(test_data[target_name], y_pred_test_mediation),
                'MAE': mean_absolute_error(test_data[target_name], y_pred_test_mediation),
                'MedAE': median_absolute_error(test_data[target_name], y_pred_test_mediation)
            }
        },
        'direct': {  # Direct model: 3 printing parameters directly→mechanical modulus
            'val': {
                'MSE': mean_squared_error(val_data[target_name], y_pred_val_direct),
                'R2': r2_score(val_data[target_name], y_pred_val_direct)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target_name], y_pred_test_direct),
                'R2': r2_score(test_data[target_name], y_pred_test_direct),
                'MAE': mean_absolute_error(test_data[target_name], y_pred_test_direct),
                'MedAE': median_absolute_error(test_data[target_name], y_pred_test_direct)
            }
        },
        'hybrid': {  # Hybrid model: 5 features (3+2)→mechanical modulus
            'val': {
                'MSE': mean_squared_error(val_data[target_name], y_pred_val_hybrid),
                'R2': r2_score(val_data[target_name], y_pred_val_hybrid)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target_name], y_pred_test_hybrid),
                'R2': r2_score(test_data[target_name], y_pred_test_hybrid),
                'MAE': mean_absolute_error(test_data[target_name], y_pred_test_hybrid),
                'MedAE': median_absolute_error(test_data[target_name], y_pred_test_hybrid)
            }
        }
    }

    # Save prediction results
    predictions = {
        'mediation': {
            'val': {'y_true': val_data[target_name], 'y_pred': y_pred_val_mediation},
            'test': {'y_true': test_data[target_name], 'y_pred': y_pred_test_mediation}
        },
        'direct': {
            'val': {'y_true': val_data[target_name], 'y_pred': y_pred_val_direct},
            'test': {'y_true': test_data[target_name], 'y_pred': y_pred_test_direct}
        },
        'hybrid': {
            'val': {'y_true': val_data[target_name], 'y_pred': y_pred_val_hybrid},
            'test': {'y_true': test_data[target_name], 'y_pred': y_pred_test_hybrid}
        }
    }

    # Save models and best parameters
    models = {
        'width': best_width,
        'height': best_height,
        'mediation': best_mediation,
        'direct': best_direct,
        'hybrid': best_hybrid,
        'best_params': {
            'width': grid_width.best_params_,
            'height': grid_height.best_params_,
            'mediation': grid_mediation.best_params_,
            'direct': grid_direct.best_params_,
            'hybrid': grid_hybrid.best_params_
        },
        'features': {
            'predictors': predictors_names,
            'mediators': mediator_names,
            'hybrid': hybrid_features,
            'mediation_second_layer': mediation_train_features.columns.tolist()
        }
    }

    return results, predictions, models, test_data


# Create output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)
os.makedirs("XGB_Regression_Comparison", exist_ok=True)
os.makedirs("models", exist_ok=True)  # Model saving directory

# Start timing
start_time = time.time()

# Generate unique log filename
base_log_file = "XGB_Intermediary(multi seeds)_CV_effect_analysis_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# Redirect output stream
sys.stdout = Logger(log_file)

print(f"Starting XGBoost mediation effect analysis experiment, logs will be saved to {log_file}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# Load data - do not use aspect_ratio, keep original width and height
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Width', 'Height', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("Data preparation complete:")
print(f"- Included features: {list(data.columns)}")
print(f"- Printing parameters (independent variables): {['printing_temperature', 'feed_rate', 'printing_speed']}")
print(f"- Mediator variables: {['Width', 'Height']} (also serve as dependent variables for printing parameters)")
print(f"- Target variable: 'Experiment_mean(MPa)'")
print(f"- Dataset split ratio: Training:Validation:Test = 7:1.5:1.5 (stratified sampling based on target variable)")

# Define XGBoost parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# Define seed values to test
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]  # 9 experiments to verify stability
print(f"\nWill use the following seeds for experiments: {seeds}")

# Store results from all experiments
all_results = []
all_predictions = []
final_models = None
test_data = None

# Track best model for each model type (based on test set R²)
best_models = {
    'mediation': {'model': None, 'r2': -np.inf, 'seed': None},
    'direct': {'model': None, 'r2': -np.inf, 'seed': None},
    'hybrid': {'model': None, 'r2': -np.inf, 'seed': None}
}

# Run multiple experiments
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"Starting experiment {i + 1}/{len(seeds)}, seed value: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, test = run_mediation_experiment(seed, data, param_grid)
    all_results.append(results)
    all_predictions.append(predictions)

    # Track best model for each model type (based on test set R²)
    current_r2 = results['mediation']['test']['R2']
    if current_r2 > best_models['mediation']['r2']:
        best_models['mediation'] = {'model': models, 'r2': current_r2, 'seed': seed}

    current_r2 = results['direct']['test']['R2']
    if current_r2 > best_models['direct']['r2']:
        best_models['direct'] = {'model': models, 'r2': current_r2, 'seed': seed}

    current_r2 = results['hybrid']['test']['R2']
    if current_r2 > best_models['hybrid']['r2']:
        best_models['hybrid'] = {'model': models, 'r2': current_r2, 'seed': seed}

    # Save models and test data from last experiment for visualization
    if i == len(seeds) - 1:
        final_models = models
        test_data = test

    # Output evaluation results for this experiment
    print(f"\nEvaluation results for experiment {i + 1} (4 decimal places):")
    print(f"Mediation model - Test set R²: {results['mediation']['test']['R2']:.4f}")
    print(f"Direct model - Test set R²: {results['direct']['test']['R2']:.4f}")
    print(f"Hybrid model - Test set R²: {results['hybrid']['test']['R2']:.4f}")

# Save the best three models
print("\n" + "=" * 50)
print("Saving best models:")
for model_type in ['mediation', 'direct', 'hybrid']:
    model_info = best_models[model_type]
    model_path = os.path.join("models", f"best_xgb_{model_type}_model(multi seeds)_CV_seed_{model_info['seed']}.pkl")
    joblib.dump(model_info['model'], model_path)
    print(f"- Best {model_type} model (seed {model_info['seed']}, R²={model_info['r2']:.4f}) saved to: {model_path}")


# Calculate mean, standard deviation, and coefficient of variation for all experiments
def calculate_stats(results_list):
    """Calculate statistics for multiple experiment results: mean ± standard deviation and coefficient of variation (CV)"""
    stats_results = {
        'mediation': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        },
        'direct': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        },
        'hybrid': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        }
    }

    # Collect all results
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in stats_results[model_type][dataset_type]:
                values = [res[model_type][dataset_type][metric] for res in results_list
                          if metric in res[model_type][dataset_type]]

                # Calculate statistics
                stats_results[model_type][dataset_type][metric]['mean'] = np.mean(values)
                stats_results[model_type][dataset_type][metric]['std'] = np.std(values)
                # Coefficient of variation = standard deviation / mean (handle division by zero)
                mean_val = stats_results[model_type][dataset_type][metric]['mean']
                if mean_val != 0:
                    stats_results[model_type][dataset_type][metric]['cv'] = (
                            stats_results[model_type][dataset_type][metric]['std'] / mean_val
                    )
                else:
                    stats_results[model_type][dataset_type][metric]['cv'] = 0

    return stats_results


# Calculate statistical results
stats_results = calculate_stats(all_results)

# Output statistical results
print('\n' + '=' * 50)
print("Statistical evaluation results from multiple experiments (4 decimal places):")
print('=' * 50)

for model_type, model_name in [
    ('mediation', 'Mediation model (printing parameters→width/height→mechanical modulus)'),
    ('direct', 'Direct model (printing parameters directly→mechanical modulus)'),
    ('hybrid', 'Hybrid model (printing parameters+width/height→mechanical modulus)')
]:
    print(f"\n{model_name}:")
    print("Validation set statistics:")
    print(
        f"  Mean Squared Error (MSE): {stats_results[model_type]['val']['MSE']['mean']:.4f} ± {stats_results[model_type]['val']['MSE']['std']:.4f}, CV={stats_results[model_type]['val']['MSE']['cv']:.4f}")
    print(
        f"  Coefficient of Determination (R2): {stats_results[model_type]['val']['R2']['mean']:.4f} ± {stats_results[model_type]['val']['R2']['std']:.4f}, CV={stats_results[model_type]['val']['R2']['cv']:.4f}")
    print("Test set statistics:")
    print(
        f"  Mean Squared Error (MSE): {stats_results[model_type]['test']['MSE']['mean']:.4f} ± {stats_results[model_type]['test']['MSE']['std']:.4f}, CV={stats_results[model_type]['test']['MSE']['cv']:.4f}")
    print(
        f"  Coefficient of Determination (R2): {stats_results[model_type]['test']['R2']['mean']:.4f} ± {stats_results[model_type]['test']['R2']['std']:.4f}, CV={stats_results[model_type]['test']['R2']['cv']:.4f}")
    print(
        f"  Mean Absolute Error (MAE): {stats_results[model_type]['test']['MAE']['mean']:.4f} ± {stats_results[model_type]['test']['MAE']['std']:.4f}, CV={stats_results[model_type]['test']['MAE']['cv']:.4f}")
    print(
        f"  Median Absolute Error (MedAE): {stats_results[model_type]['test']['MedAE']['mean']:.4f} ± {stats_results[model_type]['test']['MedAE']['std']:.4f}, CV={stats_results[model_type]['test']['MedAE']['cv']:.4f}")


# Mediation effect analysis: calculate mediation ratio
def calculate_mediation_effect(stats_results):
    """Calculate mediation effect ratio"""
    # Total effect (effect of direct model)
    total_effect = stats_results['direct']['test']['R2']['mean']

    # Direct effect (direct effect after controlling for mediators)
    # Approximated using difference between hybrid model and mediation model
    direct_effect = stats_results['hybrid']['test']['R2']['mean'] - stats_results['mediation']['test']['R2']['mean']

    # Mediation effect = total effect - direct effect
    mediation_effect = total_effect - direct_effect

    # Mediation ratio
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


# Calculate mediation effect
mediation_stats = calculate_mediation_effect(stats_results)

print('\n' + '=' * 50)
print("Mediation effect analysis results:")
print('=' * 50)
print(f"Total effect (direct model R²): {mediation_stats['total_effect']:.4f}")
print(f"Direct effect (after controlling for width/height): {mediation_stats['direct_effect']:.4f}")
print(f"Mediation effect (through width/height): {mediation_stats['mediation_effect']:.4f}")
print(f"Mediation ratio (mediation effect/total effect): {mediation_stats['mediation_ratio']:.2%}")

# Output best parameters for each model
print('\n' + '=' * 50)
print("Best parameters for each model (last experiment):")
print('=' * 50)
for model_type, params in final_models['best_params'].items():
    print(f"\n{model_type} model best parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")

# Plot feature importance analysis
plt.figure(figsize=(18, 6))

# 1. Impact of printing parameters on width
plt.subplot(1, 3, 1)
feature_importance_width = final_models['width'].feature_importances_
feature_names = final_models['features']['predictors']
plt.bar(feature_names, feature_importance_width)
plt.xlabel('Print Parameters')
plt.ylabel('Importance')
plt.title('The Importance of the impact of Print Parameters on Width')
plt.xticks(rotation=45)

# 2. Impact of printing parameters on height
plt.subplot(1, 3, 2)
feature_importance_height = final_models['height'].feature_importances_
plt.bar(feature_names, feature_importance_height)
plt.xlabel('Print Parameters')
plt.ylabel('Importance')
plt.title('The Importance of the impact of Print Parameters on Height')
plt.xticks(rotation=45)

# 3. Impact of 5 features in second layer of mediation model on mechanical modulus
plt.subplot(1, 3, 3)
feature_importance_mediation = final_models['mediation'].feature_importances_
feature_names_mediation = final_models['features']['mediation_second_layer']
plt.bar(feature_names_mediation, feature_importance_mediation)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Second Layer of Mediation Model')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join("XGB_Regression_Comparison", "XGB_Intermediary(multi seeds)_CV_feature_importance.png"), dpi=300)
plt.show()

# Plot predicted vs true values comparison for three models
plt.figure(figsize=(18, 6))

model_types = ['mediation', 'direct', 'hybrid']
model_names = ['Mediated Model', 'Direct Model', 'Mixed Model']

for i, (model_type, name) in enumerate(zip(model_types, model_names), 1):
    plt.subplot(1, 3, i)

    # Get prediction results from last experiment
    y_true = all_predictions[-1][model_type]['test']['y_true']
    y_pred = all_predictions[-1][model_type]['test']['y_pred']

    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    r2 = all_results[-1][model_type]['test']['R2']
    plt.title(f'{name} - True vs Predicted (R²={r2:.4f})')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("XGB_Regression_Comparison", "XGB_Intermediary(multi seeds)_CV_model_comparison_scatter.png"), dpi=300)
plt.show()

# Plot evaluation metrics comparison for three models
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
model_types = ['mediation', 'direct', 'hybrid']
model_names = ['Mediated Model', 'Direct Model', 'Mixed Model']

plt.figure(figsize=(16, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)

    values = [stats_results[mt]['test'][metric]['mean'] for mt in model_types]
    plt.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title(f'Model {metric} Comparison')
    plt.ylabel(metric)

    # Add value labels
    for j, v in enumerate(values):
        plt.text(j, v + 0.01, f'{v:.4f}', ha='center')

    # Limit R2 metric range to 0-1
    if metric == 'R2':
        plt.ylim(0, 1)

    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("XGB_Regression_Comparison", "XGB_Intermediary(multi seeds)_CV_model_metrics_comparison.png"), dpi=300)
plt.show()

# Export all prediction results and statistical metrics to Excel
output_file = get_unique_filename(os.path.join("prediction_results", "XGB_Intermediary(multi seeds)_CV_analysis_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Export prediction results for each experiment
    for exp_idx, predictions in enumerate(all_predictions):
        for model_type in ['mediation', 'direct', 'hybrid']:
            for dataset_type in ['val', 'test']:
                df = pd.DataFrame({
                    'true_values': predictions[model_type][dataset_type]['y_true'],
                    'predicted_values': predictions[model_type][dataset_type]['y_pred']
                })
                sheet_name = f'exp_{exp_idx + 1}_{model_type}_{dataset_type}'
                # Ensure sheet name doesn't exceed 31 characters (Excel limit)
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Export statistical results
    stats_df = pd.DataFrame()
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in stats_results[model_type][dataset_type]:
                stats_df.at[f'{model_type}_{dataset_type}', f'{metric}_mean'] = \
                stats_results[model_type][dataset_type][metric]['mean']
                stats_df.at[f'{model_type}_{dataset_type}', f'{metric}_std'] = \
                stats_results[model_type][dataset_type][metric]['std']
                stats_df.at[f'{model_type}_{dataset_type}', f'{metric}_cv'] = \
                stats_results[model_type][dataset_type][metric]['cv']

    stats_df.to_excel(writer, sheet_name='statistics_summary')

    # Export mediation effect analysis results
    mediation_df = pd.DataFrame([mediation_stats])
    mediation_df.to_excel(writer, sheet_name='mediation_analysis', index=False)

print(f"\nAll prediction results and statistical metrics exported to: {output_file}")

# Calculate and output total runtime
end_time = time.time()
total_time = end_time - start_time
print(f"\n{'=' * 50}")
print(f"Experiment completed!")
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {str(timedelta(seconds=total_time))}")
print(f"Logs saved to: {log_file}")
print(f"Visualization results saved to: XGB_Regression_Comparison folder")
print(f"Models saved to: models folder")

# Restore standard output
sys.stdout = sys.__stdout__
