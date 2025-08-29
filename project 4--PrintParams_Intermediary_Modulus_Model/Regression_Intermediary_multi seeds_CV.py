import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import joblib  # For saving best model parameters

# Seed settings consistent with RF
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]  # Total of 9 seeds


class Logger:
    """Redirect output to both console and file"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_unique_filename(base_name):
    """Generate unique filename with counter suffix if exists"""
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


def run_regression_experiment(seed):
    """Run single regression experiment and return results (including validation and test sets)"""
    np.random.seed(seed)

    # Load data
    data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
    selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
    data = data[selected_columns]

    # Define polynomial degrees
    poly_degrees = [2, 3]

    # Define variables - clear input parameter combinations for three model types
    predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # T (temperature), V (speed), F (feed rate)
    mediators = ['Height', 'Width']  # W (width), H (height)
    target = 'Experiment_mean(MPa)'  # Final target variable (mechanical modulus)

    # Dataset splitting: stratified sampling based on mechanical modulus
    # Training set (70%), validation set (15%), test set (15%)
    # First create bins for stratification

    # Dynamically adjust number of bins to ensure at least 4 samples per bin (for two stratifications)
    min_samples_per_bin = 4  # Ensure at least 4 samples per bin
    max_bins = 10

    # Calculate maximum possible number of bins
    max_possible_bins = len(data) // min_samples_per_bin
    num_bins = min(max_bins, max_possible_bins)

    # Ensure at least 2 bins
    num_bins = max(2, num_bins)

    # Create bins
    data['target_bin'] = pd.cut(data[target], bins=num_bins, labels=False)

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
            # Merge with neighbor with more samples
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

        # Break loop if only one bin remains
        if len(bin_counts) == 1:
            break

    # Output bin information to verify minimum 4 samples per bin
    print(f"Stratified sampling verification:")
    print(f"- Number of bins: {len(bin_counts)}")
    for bin_idx in bin_counts.index:
        print(f"  Bin {bin_idx}: {bin_counts[bin_idx]} samples")
    assert (bin_counts >= min_samples_per_bin).all() or len(bin_counts) == 1, \
        "Bins with fewer than 4 samples exist. Please check data or adjust binning strategy"

    # Use random sampling instead of stratified if all data in one bin
    stratify_param = data['target_bin'] if len(bin_counts) > 1 else None

    # First stratified split: training set and temporary set
    train_data, temp_data = train_test_split(
        data,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_param  # Stratify based on target variable
    )

    # Prepare stratification parameter for second split
    if stratify_param is not None:
        stratify_param_temp = temp_data['target_bin']
        # Check sample counts in temporary set bins
        temp_bin_counts = stratify_param_temp.value_counts()
        # Use random sampling if any bin has fewer than 2 samples
        if (temp_bin_counts < 2).any():
            stratify_param_temp = None
    else:
        stratify_param_temp = None

    # Second stratified split: validation and test sets from temporary set
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

    # Prepare model dictionaries
    models = {}  # Store all models
    results = {}  # Store all evaluation results (including val and test)
    model_names = {}  # Store model names
    model_params = {}  # Store model parameters

    # --------------------------
    # 1. Mediation model (nested): [T, V, F, Ŵ (predicted width), Ĥ (predicted height)]
    #    Geometric features source: First layer model predictions
    #    Core objective: High-precision prediction +解析中介机制
    #    Application scenarios: Research optimization, clinical personalized customization
    # --------------------------

    # 1.1 First step: Predict width and height using 3 printing parameters (first layer: 3 input variables)
    # Linear regression for width and height prediction
    for mediator in mediators:
        model = LinearRegression()
        # Explicitly use 3 printing parameters as input
        model.fit(train_data[predictors], train_data[mediator])
        key = f'mediator_linear_{mediator.lower()}'
        models[key] = model
        model_names[key] = f'Linear Regression (predict {mediator})'
        model_params[key] = {
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }

    # Polynomial regression for width and height prediction
    for degree in poly_degrees:
        for mediator in mediators:
            poly = PolynomialFeatures(degree=degree)
            # Explicitly use 3 printing parameters as input
            X_train_poly = poly.fit_transform(train_data[predictors])
            X_val_poly = poly.transform(val_data[predictors])

            best_alpha = None
            best_r2 = -np.inf
            best_model = None
            alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

            for alpha in alpha_values:
                model = Ridge(alpha=alpha, random_state=seed)
                model.fit(X_train_poly, train_data[mediator])
                val_pred = model.predict(X_val_poly)
                val_r2 = r2_score(val_data[mediator], val_pred)

                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_alpha = alpha
                    best_model = model

            key = f'mediator_poly{degree}_{mediator.lower()}'
            models[key] = {
                'model': best_model,
                'poly': poly
            }
            model_names[key] = f'Polynomial Regression (degree={degree}, predict {mediator})'
            model_params[key] = {
                'alpha': best_alpha,
                'coefficients': best_model.coef_,
                'intercept': best_model.intercept_,
                'poly_features': poly.get_feature_names_out(predictors)
            }

    # 1.2 Second step: Predict mechanical modulus using [3 printing parameters + 2 predicted width/height] (second layer: 5 input variables)
    def create_mediation_model(degree=1):
        width_model_key = f'mediator_linear_width' if degree == 1 else f'mediator_poly{degree}_width'
        height_model_key = f'mediator_linear_height' if degree == 1 else f'mediator_poly{degree}_height'

        # Predict width and height
        if degree == 1:
            train_pred_width = models[width_model_key].predict(train_data[predictors])
            train_pred_height = models[height_model_key].predict(train_data[predictors])
        else:
            train_pred_width = models[width_model_key]['model'].predict(
                models[width_model_key]['poly'].transform(train_data[predictors])
            )
            train_pred_height = models[height_model_key]['model'].predict(
                models[height_model_key]['poly'].transform(train_data[predictors])
            )

        # Create second layer features: 3 printing parameters + 2 predicted width/height (total 5 features)
        train_mediation_features = train_data[predictors].copy()
        train_mediation_features['predicted_width'] = train_pred_width
        train_mediation_features['predicted_height'] = train_pred_height

        # Verify feature count is 5
        assert train_mediation_features.shape[
                   1] == 5, f"Incorrect feature count for mediation model second layer: should be 5, actual {train_mediation_features.shape[1]}"

        # Create same features for validation set
        if degree == 1:
            val_pred_width = models[width_model_key].predict(val_data[predictors])
            val_pred_height = models[height_model_key].predict(val_data[predictors])
        else:
            val_pred_width = models[width_model_key]['model'].predict(
                models[width_model_key]['poly'].transform(val_data[predictors])
            )
            val_pred_height = models[height_model_key]['model'].predict(
                models[height_model_key]['poly'].transform(val_data[predictors])
            )

        val_mediation_features = val_data[predictors].copy()
        val_mediation_features['predicted_width'] = val_pred_width
        val_mediation_features['predicted_height'] = val_pred_height

        # Verify feature count is 5
        assert val_mediation_features.shape[
                   1] == 5, f"Incorrect validation feature count for mediation model second layer: should be 5, actual {val_mediation_features.shape[1]}"

        # Train final model (using 5 features)
        if degree == 1:
            final_model = LinearRegression()
            final_model.fit(train_mediation_features, train_data[target])
            return final_model, None, val_mediation_features, {
                'coefficients': final_model.coef_,
                'intercept': final_model.intercept_,
                'features': train_mediation_features.columns.tolist()
            }
        else:
            # Use degree=1 to ensure no new interaction features, maintaining 5 input features
            poly_final = PolynomialFeatures(degree=1)
            X_train_final = poly_final.fit_transform(train_mediation_features)
            X_val_final = poly_final.transform(val_mediation_features)

            best_alpha = None
            best_r2 = -np.inf
            best_model = None
            alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

            for alpha in alpha_values:
                model = Ridge(alpha=alpha, random_state=seed)
                model.fit(X_train_final, train_data[target])
                val_pred = model.predict(X_val_final)
                val_r2 = r2_score(val_data[target], val_pred)

                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_alpha = alpha
                    best_model = model

            return best_model, poly_final, val_mediation_features, {
                'alpha': best_alpha,
                'coefficients': best_model.coef_,
                'intercept': best_model.intercept_,
                'features': train_mediation_features.columns.tolist()
            }

    # Create mediation models with different degrees
    for degree in [1] + poly_degrees:
        model, poly, val_mediation_features, params = create_mediation_model(degree)
        key = f'mediation_model_degree{degree}'
        models[key] = {
            'model': model,
            'poly': poly,
            'degree': degree,
            'val_features': val_mediation_features  # Save validation features for evaluation
        }
        model_params[key] = params
        if degree == 1:
            model_names[key] = 'Mediation Model (Linear)'
        else:
            model_names[key] = f'Mediation Model (Polynomial degree={degree})'

    # --------------------------
    # 2. Direct model: [T (temperature), V (speed), F (feed rate)]
    #    Geometric features source: None
    #    Core objective: Fast preliminary performance prediction
    #    Application scenarios: Low precision requirements, rapid parameter screening
    # --------------------------

    # Linear direct model
    model = LinearRegression()
    model.fit(train_data[predictors], train_data[target])
    key = 'direct_linear'
    models[key] = model
    model_names[key] = 'Direct Linear Model'
    model_params[key] = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'features': predictors
    }

    # Polynomial direct models
    for degree in poly_degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(train_data[predictors])
        X_val_poly = poly.transform(val_data[predictors])  # Save validation features

        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_poly, train_data[target])
            val_pred = model.predict(X_val_poly)
            val_r2 = r2_score(val_data[target], val_pred)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_alpha = alpha
                best_model = model

        key = f'direct_poly{degree}'
        models[key] = {
            'model': best_model,
            'poly': poly,
            'val_features': X_val_poly  # Save validation features
        }
        model_names[key] = f'Direct Polynomial Model (degree={degree})'
        model_params[key] = {
            'alpha': best_alpha,
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_,
            'degree': degree,
            'poly_features': poly.get_feature_names_out(predictors)
        }

    # --------------------------
    # 3. Hybrid model: [T, V, F, W_true (actual width), H_true (actual height)]
    #    Geometric features source: Physical measurements (e.g., SEM)
    #    Core objective: Verify gain potential of geometric features
    #    Application scenarios: Model performance upper limit reference
    # --------------------------

    hybrid_features = predictors + mediators  # 3 printing parameters + 2 actual width/height

    # Linear hybrid model
    model = LinearRegression()
    model.fit(train_data[hybrid_features], train_data[target])
    key = 'hybrid_linear'
    models[key] = model
    model_names[key] = 'Hybrid Linear Model'
    model_params[key] = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'features': hybrid_features
    }

    # Polynomial hybrid models
    for degree in poly_degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(train_data[hybrid_features])
        X_val_poly = poly.transform(val_data[hybrid_features])  # Save validation features

        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_poly, train_data[target])
            val_pred = model.predict(X_val_poly)
            val_r2 = r2_score(val_data[target], val_pred)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_alpha = alpha
                best_model = model

        key = f'hybrid_poly{degree}'
        models[key] = {
            'model': best_model,
            'poly': poly,
            'val_features': X_val_poly  # Save validation features
        }
        model_names[key] = f'Hybrid Polynomial Model (degree={degree})'
        model_params[key] = {
            'alpha': best_alpha,
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_,
            'degree': degree,
            'poly_features': poly.get_feature_names_out(hybrid_features)
        }

    # --------------------------
    # Evaluate all models (calculate for both validation and test sets)
    # --------------------------

    # Evaluate mediation models
    for degree in [1] + poly_degrees:
        key = f'mediation_model_degree{degree}'
        model_info = models[key]
        degree_mediator = model_info['degree']

        # Validation set evaluation
        val_mediation_features = model_info['val_features']
        if model_info['poly'] is None:
            y_pred_val = model_info['model'].predict(val_mediation_features)
        else:
            X_val_final = model_info['poly'].transform(val_mediation_features)
            y_pred_val = model_info['model'].predict(X_val_final)
        y_true_val = val_data[target]

        # Test set evaluation
        width_model_key = f'mediator_linear_width' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_width'
        height_model_key = f'mediator_linear_height' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_height'

        # Generate predicted width and height for test set
        if degree_mediator == 1:
            test_pred_width = models[width_model_key].predict(test_data[predictors])
            test_pred_height = models[height_model_key].predict(test_data[predictors])
        else:
            test_pred_width = models[width_model_key]['model'].predict(
                models[width_model_key]['poly'].transform(test_data[predictors])
            )
            test_pred_height = models[height_model_key]['model'].predict(
                models[height_model_key]['poly'].transform(test_data[predictors])
            )

        # Create 5 features for test set: 3 printing parameters + 2 predicted width/height
        test_mediation_features = test_data[predictors].copy()
        test_mediation_features['predicted_width'] = test_pred_width
        test_mediation_features['predicted_height'] = test_pred_height

        # Verify feature count is 5
        assert test_mediation_features.shape[
                   1] == 5, f"Incorrect test feature count for mediation model second layer: should be 5, actual {test_mediation_features.shape[1]}"

        # Predict and evaluate
        if model_info['poly'] is None:
            y_pred_test = model_info['model'].predict(test_mediation_features)
        else:
            X_test_final = model_info['poly'].transform(test_mediation_features)
            y_pred_test = model_info['model'].predict(X_test_final)
        y_true_test = test_data[target]

        # Store validation and test results
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    # Evaluate direct models
    key = 'direct_linear'
    model = models[key]
    # Validation set
    y_pred_val = model.predict(val_data[predictors])
    y_true_val = val_data[target]
    # Test set
    y_pred_test = model.predict(test_data[predictors])
    y_true_test = test_data[target]
    results[key] = {
        'val': {
            'MSE': mean_squared_error(y_true_val, y_pred_val),
            'R2': r2_score(y_true_val, y_pred_val),
            'MAE': mean_absolute_error(y_true_val, y_pred_val),
            'MedAE': median_absolute_error(y_true_val, y_pred_val)
        },
        'test': {
            'MSE': mean_squared_error(y_true_test, y_pred_test),
            'R2': r2_score(y_true_test, y_pred_test),
            'MAE': mean_absolute_error(y_true_test, y_pred_test),
            'MedAE': median_absolute_error(y_true_test, y_pred_test)
        }
    }

    for degree in poly_degrees:
        key = f'direct_poly{degree}'
        model_info = models[key]
        # Validation set
        y_pred_val = model_info['model'].predict(model_info['val_features'])
        y_true_val = val_data[target]
        # Test set
        X_test_poly = model_info['poly'].transform(test_data[predictors])
        y_pred_test = model_info['model'].predict(X_test_poly)
        y_true_test = test_data[target]
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    # Evaluate hybrid models
    key = 'hybrid_linear'
    model = models[key]
    # Validation set
    y_pred_val = model.predict(val_data[hybrid_features])
    y_true_val = val_data[target]
    # Test set
    y_pred_test = model.predict(test_data[hybrid_features])
    y_true_test = test_data[target]
    results[key] = {
        'val': {
            'MSE': mean_squared_error(y_true_val, y_pred_val),
            'R2': r2_score(y_true_val, y_pred_val),
            'MAE': mean_absolute_error(y_true_val, y_pred_val),
            'MedAE': median_absolute_error(y_true_val, y_pred_val)
        },
        'test': {
            'MSE': mean_squared_error(y_true_test, y_pred_test),
            'R2': r2_score(y_true_test, y_pred_test),
            'MAE': mean_absolute_error(y_true_test, y_pred_test),
            'MedAE': median_absolute_error(y_true_test, y_pred_test)
        }
    }

    for degree in poly_degrees:
        key = f'hybrid_poly{degree}'
        model_info = models[key]
        # Validation set
        y_pred_val = model_info['model'].predict(model_info['val_features'])
        y_true_val = val_data[target]
        # Test set
        X_test_poly = model_info['poly'].transform(test_data[hybrid_features])
        y_pred_test = model_info['model'].predict(X_test_poly)
        y_true_test = test_data[target]
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    return results, model_names, model_params, val_data[target], test_data[target]


def calculate_statistics(results_list):
    """Calculate statistics for multiple experiments: mean, standard deviation, and coefficient of variation (including validation and test sets)"""
    if not results_list:
        return {}

    # Initialize statistics results dictionary
    stats_results = {}
    model_keys = results_list[0].keys()

    # Initialize metric lists for each model (including val and test)
    for model_key in model_keys:
        stats_results[model_key] = {
            'val': {
                'MSE': {'mean': 0, 'std': 0, 'cv': 0},
                'R2': {'mean': 0, 'std': 0, 'cv': 0},
                'MAE': {'mean': 0, 'std': 0, 'cv': 0},
                'MedAE': {'mean': 0, 'std': 0, 'cv': 0}
            },
            'test': {
                'MSE': {'mean': 0, 'std': 0, 'cv': 0},
                'R2': {'mean': 0, 'std': 0, 'cv': 0},
                'MAE': {'mean': 0, 'std': 0, 'cv': 0},
                'MedAE': {'mean': 0, 'std': 0, 'cv': 0}
            }
        }

    # Collect all experiment results
    for model_key in model_keys:
        for dataset_type in ['val', 'test']:
            for metric in ['MSE', 'R2', 'MAE', 'MedAE']:
                values = [result[model_key][dataset_type][metric] for result in results_list]
                mean_val = np.mean(values)
                std_val = np.std(values)
                # Coefficient of variation = standard deviation / mean (handle case when mean is 0)
                cv_val = std_val / mean_val if mean_val != 0 else 0

                stats_results[model_key][dataset_type][metric]['mean'] = mean_val
                stats_results[model_key][dataset_type][metric]['std'] = std_val
                stats_results[model_key][dataset_type][metric]['cv'] = cv_val

    return stats_results


def calculate_mediation_effect(stats_results):
    """Calculate mediation effect ratio"""
    # Total effect (effect of direct model)
    total_effect = stats_results['direct_linear']['test']['R2']['mean']

    # Direct effect (direct effect after controlling for mediator variables)
    direct_effect = stats_results['hybrid_linear']['test']['R2']['mean'] - \
                    stats_results['mediation_model_degree1']['test']['R2']['mean']

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


def save_statistics_table(stats_results, model_names, output_file):
    """Save statistics results as CSV table"""
    rows = []

    # Add header
    rows.append(["Model Type", "Dataset Type", "Metric", "Mean", "Standard Deviation", "Mean ± Std",
                 "Coefficient of Variation (CV)"])

    # Iterate through all models and metrics
    for model_key in stats_results:
        model_name = model_names[model_key]
        for dataset_type in ['val', 'test']:
            dataset_name = "Validation Set" if dataset_type == 'val' else "Test Set"
            for metric in ['MSE', 'R2', 'MAE', 'MedAE']:
                metric_name = {
                    'MSE': 'Mean Squared Error',
                    'R2': 'Coefficient of Determination',
                    'MAE': 'Mean Absolute Error',
                    'MedAE': 'Median Absolute Error'
                }[metric]

                mean_val = stats_results[model_key][dataset_type][metric]['mean']
                std_val = stats_results[model_key][dataset_type][metric]['std']
                cv_val = stats_results[model_key][dataset_type][metric]['cv']

                # Format data
                mean_str = f"{mean_val:.4f}"
                std_str = f"{std_val:.4f}"
                mean_std_str = f"{mean_val:.4f} ± {std_val:.4f}"
                cv_str = f"{cv_val:.4f}"

                rows.append([model_name, dataset_name, metric_name, mean_str, std_str, mean_std_str, cv_str])

    # Create DataFrame and save as CSV
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Statistics table saved to: {output_file}")


# Main program
if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("model_parameters", exist_ok=True)  # Model parameters directory
    # Create directory for best parameters per model
    os.makedirs(os.path.join("model_parameters", "best_per_model"), exist_ok=True)

    # Start timing
    start_time = time.time()

    # Generate unique log filename
    base_log_file = "Regression_Intermediary(multi seeds)_regression_log.txt"
    log_file = get_unique_filename(os.path.join("outputs", base_log_file))

    # Redirect output stream
    sys.stdout = Logger(log_file)

    print(f"Starting mediation effect regression analysis. Log will be saved to {log_file}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using seed list: {seeds}")
    print("Using stratified sampling based on mechanical modulus for dataset partitioning")
    print("Model parameter combinations:")
    print("- Direct model: [T (temperature), V (speed), F (feed rate)], no geometric features")
    print(
        "- Mediation model: [T, V, F, Ŵ (predicted width), Ĥ (predicted height)], geometric features from first layer model predictions")
    print(
        "- Hybrid model: [T, V, F, W_true (actual width), H_true (actual height)], geometric features from physical measurements")
    print("=" * 50)

    # Run multiple experiments
    results_list = []
    all_model_params = []  # Store model parameters for all seeds
    model_names = None
    y_true_val = None
    y_true_test = None

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 30}")
        print(f"Running experiment {i + 1}/{len(seeds)}, seed: {seed}")
        print(f"{'=' * 30}")

        results, exp_model_names, exp_model_params, exp_y_val, exp_y_test = run_regression_experiment(seed)
        results_list.append(results)
        all_model_params.append(exp_model_params)

        # Save model names (consistent across all experiments)
        if model_names is None:
            model_names = exp_model_names
            y_true_val = exp_y_val
            y_true_test = exp_y_test

        # Output evaluation results for current seed (including validation and test sets)
        print(f"\nExperiment {i + 1} model evaluation results:")
        print('-' * 60)
        for model_key in results:
            print(f'{model_names[model_key]}:')
            print(f'  Validation set:')
            print(f'    MSE: {results[model_key]["val"]["MSE"]:.4f}')
            print(f'    R2: {results[model_key]["val"]["R2"]:.4f}')
            print(f'    MAE: {results[model_key]["val"]["MAE"]:.4f}')
            print(f'    MedAE: {results[model_key]["val"]["MedAE"]:.4f}')
            print(f'  Test set:')
            print(f'    MSE: {results[model_key]["test"]["MSE"]:.4f}')
            print(f'    R2: {results[model_key]["test"]["R2"]:.4f}')
            print(f'    MAE: {results[model_key]["test"]["MAE"]:.4f}')
            print(f'    MedAE: {results[model_key]["test"]["MedAE"]:.4f}')
            print('-' * 60)

    # Calculate statistics (mean, standard deviation, coefficient of variation)
    print("\n" + "=" * 50)
    print("Calculating statistics across multiple experiments...")
    stats_results = calculate_statistics(results_list)

    # Output statistical evaluation results (including validation and test sets)
    print("\n" + "=" * 50)
    print("All models statistical evaluation results (4 decimal places):")
    print('=' * 50)
    for model_key in stats_results:
        print(f'\n{model_names[model_key]} statistical evaluation:')
        print(f'  Validation set:')
        print(
            f'    Mean Squared Error (MSE): {stats_results[model_key]["val"]["MSE"]["mean"]:.4f} ± {stats_results[model_key]["val"]["MSE"]["std"]:.4f}, CV={stats_results[model_key]["val"]["MSE"]["cv"]:.4f}')
        print(
            f'    Coefficient of Determination (R2): {stats_results[model_key]["val"]["R2"]["mean"]:.4f} ± {stats_results[model_key]["val"]["R2"]["std"]:.4f}, CV={stats_results[model_key]["val"]["R2"]["cv"]:.4f}')
        print(
            f'    Mean Absolute Error (MAE): {stats_results[model_key]["val"]["MAE"]["mean"]:.4f} ± {stats_results[model_key]["val"]["MAE"]["std"]:.4f}, CV={stats_results[model_key]["val"]["MAE"]["cv"]:.4f}')
        print(
            f'    Median Absolute Error (MedAE): {stats_results[model_key]["val"]["MedAE"]["mean"]:.4f} ± {stats_results[model_key]["val"]["MedAE"]["std"]:.4f}, CV={stats_results[model_key]["val"]["MedAE"]["cv"]:.4f}')
        print(f'  Test set:')
        print(
            f'    Mean Squared Error (MSE): {stats_results[model_key]["test"]["MSE"]["mean"]:.4f} ± {stats_results[model_key]["test"]["MSE"]["std"]:.4f}, CV={stats_results[model_key]["test"]["MSE"]["cv"]:.4f}')
        print(
            f'    Coefficient of Determination (R2): {stats_results[model_key]["test"]["R2"]["mean"]:.4f} ± {stats_results[model_key]["test"]["R2"]["std"]:.4f}, CV={stats_results[model_key]["test"]["R2"]["cv"]:.4f}')
        print(
            f'    Mean Absolute Error (MAE): {stats_results[model_key]["test"]["MAE"]["mean"]:.4f} ± {stats_results[model_key]["test"]["MAE"]["std"]:.4f}, CV={stats_results[model_key]["test"]["MAE"]["cv"]:.4f}')
        print(
            f'    Median Absolute Error (MedAE): {stats_results[model_key]["test"]["MedAE"]["mean"]:.4f} ± {stats_results[model_key]["test"]["MedAE"]["std"]:.4f}, CV={stats_results[model_key]["test"]["MedAE"]["cv"]:.4f}')
        print('-' * 60)

    # Calculate and output mediation effect
    print("\n" + "=" * 50)
    print("Mediation effect analysis results:")
    print("=" * 50)
    mediation_effect = calculate_mediation_effect(stats_results)
    print(f"Total effect (R² from direct model): {mediation_effect['total_effect']:.4f}")
    print(f"Direct effect: {mediation_effect['direct_effect']:.4f}")
    print(f"Mediation effect: {mediation_effect['mediation_effect']:.4f}")
    print(
        f"Mediation ratio: {mediation_effect['mediation_ratio']:.4f} ({mediation_effect['mediation_ratio'] * 100:.2f}%)")

    # Save statistics table
    stats_table_file = get_unique_filename(os.path.join("outputs", "regression_statistics.csv"))
    save_statistics_table(stats_results, model_names, stats_table_file)

    # Calculate and output total running time
    end_time = time.time()
    total_time = timedelta(seconds=int(end_time - start_time))
    print("\n" + "=" * 50)
    print(f"Analysis completed successfully!")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total running time: {total_time}")
    print(f"Results log saved to: {log_file}")
    print(f"Statistics table saved to: {stats_table_file}")
    print("=" * 50)

    # Restore standard output
    sys.stdout = sys.stdout.terminal
