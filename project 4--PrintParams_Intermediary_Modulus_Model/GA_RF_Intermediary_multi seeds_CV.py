import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import random
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from deap import base, creator, tools, algorithms
import joblib
from datetime import timedelta

# Define seed range (consistent with RF version: base seed ±4, total 9 seeds)
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]
print(f"Will use the following seeds for experiments: {seeds}")

# Global variable definitions - to resolve scope issues
predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # T (temperature), V (speed), F (feed rate)
mediators = ['Height', 'Width']  # Mediator variables: H (height), W (width)
target = 'Experiment_mean(MPa)'  # Final target variable: mechanical modulus

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


def train_ga_rf(X_train, y_train, X_val, y_val, feature_set_name, seed):
    """Train Random Forest model optimized with Genetic Algorithm"""
    # Genetic Algorithm parameter settings
    POPULATION_SIZE = 50  # Population size
    NGEN = 15  # Number of generations
    CXPB_INIT = 0.6  # Initial crossover probability
    CXPB_FINAL = 0.95  # Final crossover probability
    MUTPB_INIT = 0.4  # Initial mutation probability
    MUTPB_FINAL = 0.05  # Final mutation probability

    print(f"\n{feature_set_name} feature set - GA parameters:")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Number of generations: {NGEN}")
    print(f"Initial crossover probability: {CXPB_INIT}")
    print(f"Final crossover probability: {CXPB_FINAL}")
    print(f"Initial mutation probability: {MUTPB_INIT}")
    print(f"Final mutation probability: {MUTPB_FINAL}")

    # Define genetic algorithm components: create classes only if they don't exist
    if "FitnessMax" not in creator.__dict__:  # Check if class already exists
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define parameter ranges and types
    toolbox = base.Toolbox()
    # Initialize random number generator with current seed
    toolbox.register("random", random.Random, seed)

    # Fix: Get random number generator instance
    rng = toolbox.random()

    # Register tool functions using RNG instance
    toolbox.register("n_estimators", rng.randint, 50, 200)  # Number of decision trees
    toolbox.register("max_depth", lambda: rng.choice([None, 5, 10, 15, 20]))  # Possible values for max_depth
    toolbox.register("min_samples_split", rng.randint, 2, 10)  # Minimum samples for split
    toolbox.register("min_samples_leaf", rng.randint, 1, 4)  # Minimum samples for leaf nodes
    toolbox.register("max_features", lambda: rng.choice(['sqrt', 'log2', None]))  # Maximum features

    # Create individuals and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n_estimators, toolbox.max_depth,
                      toolbox.min_samples_split, toolbox.min_samples_leaf,
                      toolbox.max_features), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define fitness function
    def evalRF(individual):
        n_est, max_d, min_split, min_leaf, max_feat = individual

        # Create Random Forest model (using current seed)
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_feat,
            random_state=seed,
            n_jobs=-1
        )

        # Train on training set
        rf.fit(X_train, y_train)
        # Evaluate on validation set
        y_pred = rf.predict(X_val)
        # Use negative MSE as fitness (since we want to maximize)
        return -mean_squared_error(y_val, y_pred),

    # Adaptive genetic operators: dynamically adjust crossover and mutation probabilities based on generation
    def adaptive_crossover_mutation(generation):
        progress = generation / NGEN
        cxpb = CXPB_INIT + (CXPB_FINAL - CXPB_INIT) * progress
        mutpb = MUTPB_INIT - (MUTPB_INIT - MUTPB_FINAL) * progress
        return cxpb, mutpb

    # Register genetic operations, custom mutation handling max_depth
    def custom_mutate(individual):
        for i in range(len(individual)):
            if rng.random() < MUTPB_INIT:  # Use RNG instance
                if i == 1:  # Special handling for max_depth
                    individual[i] = rng.choice([None, 5, 10, 15, 20])
                elif i == 0:  # n_estimators
                    individual[i] = rng.randint(50, 300)
                elif i == 2:  # min_samples_split
                    individual[i] = rng.randint(2, 10)
                elif i == 3:  # min_samples_leaf
                    individual[i] = rng.randint(1, 4)
                elif i == 4:  # max_features
                    individual[i] = rng.choice(['sqrt', 'log2', None])
        return individual,

    toolbox.register("evaluate", evalRF)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", custom_mutate)  # Custom mutation
    toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection strategy

    # Start genetic algorithm optimization
    print(f"{feature_set_name} feature set - Starting GA optimization of Random Forest parameters...")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)  # Save best individual

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run genetic algorithm, manually implementing adaptive crossover and mutation probabilities
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    # Start iterations
    for gen in range(1, NGEN + 1):
        # Adaptively adjust crossover and mutation probabilities
        cxpb, mutpb = adaptive_crossover_mutation(gen)

        # Select next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < cxpb:  # Use RNG instance
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if rng.random() < mutpb:  # Use RNG instance
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate individuals without fitness values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update population
        pop[:] = offspring

        # Update hall of fame and logbook
        hof.update(pop)
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind),** record)
        print(logbook.stream)

    # Get best parameters
    best_params = hof[0]
    print(f"{feature_set_name} feature set - Best parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, "
          f"min_samples_split={best_params[2]}, min_samples_leaf={best_params[3]}, "
          f"max_features={best_params[4]}")

    # Create and train final Random Forest model with best parameters (using current seed)
    ga_rf = RandomForestRegressor(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
        min_samples_leaf=best_params[3],
        max_features=best_params[4],
        random_state=seed,
        n_jobs=-1
    )

    # Train final model on training set
    ga_rf.fit(X_train, y_train)

    return ga_rf, best_params


def run_ga_mediation_experiment(seed, data):
    """Run a single GA-RF mediation effect experiment"""
    # Set current experiment seed
    np.random.seed(seed)
    random.seed(seed)


    # Dataset splitting: stratified sampling based on mechanical modulus, ensuring at least 4 samples per bin
    min_samples_per_bin = 4
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

    # 1.1 Train width and height prediction models (first layer of mediation model: 3 input variables [T, V, F])
    model_width, params_width = train_ga_rf(
        train_data[predictors], train_data['Width'],
        val_data[predictors], val_data['Width'],
        "Width_Predictor",
        seed  # Pass current seed
    )

    model_height, params_height = train_ga_rf(
        train_data[predictors], train_data['Height'],
        val_data[predictors], val_data['Height'],
        "Height_Predictor",
        seed  # Pass current seed
    )

    # Generate predicted width and height features, combine with original 3 printing parameters (second layer of mediation model: 5 input variables [T, V, F, Ŵ, Ĥ])
    # Training set features
    train_pred_width = model_width.predict(train_data[predictors])
    train_pred_height = model_height.predict(train_data[predictors])
    train_mediation_features = train_data[predictors].copy()
    train_mediation_features['predicted_width'] = train_pred_width  # Ŵ (predicted width)
    train_mediation_features['predicted_height'] = train_pred_height  # Ĥ (predicted height)

    # Verify feature count is 5
    assert train_mediation_features.shape[1] == 5, \
        f"Mediation model second layer training feature count error: should be 5, actual {train_mediation_features.shape[1]}"

    # Validation set features
    val_pred_width = model_width.predict(val_data[predictors])
    val_pred_height = model_height.predict(val_data[predictors])
    val_mediation_features = val_data[predictors].copy()
    val_mediation_features['predicted_width'] = val_pred_width
    val_mediation_features['predicted_height'] = val_pred_height

    # Verify feature count is 5
    assert val_mediation_features.shape[1] == 5, \
        f"Mediation model second layer validation feature count error: should be 5, actual {val_mediation_features.shape[1]}"

    # Test set features
    test_pred_width = model_width.predict(test_data[predictors])
    test_pred_height = model_height.predict(test_data[predictors])
    test_mediation_features = test_data[predictors].copy()
    test_mediation_features['predicted_width'] = test_pred_width
    test_mediation_features['predicted_height'] = test_pred_height

    # Verify feature count is 5
    assert test_mediation_features.shape[1] == 5, \
        f"Mediation model second layer test feature count error: should be 5, actual {test_mediation_features.shape[1]}"

    # 1.2 Train mediation model (nested model: 5 features [T, V, F, Ŵ, Ĥ])
    model_mediation, params_mediation = train_ga_rf(
        train_mediation_features, train_data[target],
        val_mediation_features, val_data[target],
        "Mediation_Model",
        seed  # Pass current seed
    )

    # 2. Train direct model (input parameters: 3 printing parameters [T, V, F])
    model_direct, params_direct = train_ga_rf(
        train_data[predictors], train_data[target],
        val_data[predictors], val_data[target],
        "Direct_Model",
        seed  # Pass current seed
    )

    # 3. Train hybrid model (input parameters: 3 printing parameters + 2 actual dimensions [T, V, F, W_true, H_true])
    hybrid_features = predictors + mediators
    model_hybrid, params_hybrid = train_ga_rf(
        train_data[hybrid_features], train_data[target],
        val_data[hybrid_features], val_data[target],
        "Hybrid_Model",
        seed  # Pass current seed
    )

    # Model predictions
    y_pred_val_mediation = model_mediation.predict(val_mediation_features)
    y_pred_test_mediation = model_mediation.predict(test_mediation_features)

    y_pred_val_direct = model_direct.predict(val_data[predictors])
    y_pred_test_direct = model_direct.predict(test_data[predictors])

    y_pred_val_hybrid = model_hybrid.predict(val_data[hybrid_features])
    y_pred_test_hybrid = model_hybrid.predict(test_data[hybrid_features])

    # Calculate evaluation metrics
    results = {
        'mediation': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_mediation),
                'R2': r2_score(val_data[target], y_pred_val_mediation)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_mediation),
                'R2': r2_score(test_data[target], y_pred_test_mediation),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_mediation),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_mediation)
            }
        },
        'direct': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_direct),
                'R2': r2_score(val_data[target], y_pred_val_direct)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_direct),
                'R2': r2_score(test_data[target], y_pred_test_direct),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_direct),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_direct)
            }
        },
        'hybrid': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_hybrid),
                'R2': r2_score(val_data[target], y_pred_val_hybrid)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_hybrid),
                'R2': r2_score(test_data[target], y_pred_test_hybrid),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_hybrid),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_hybrid)
            }
        }
    }

    # Save prediction results and models
    predictions = {
        'mediation': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_mediation},
                      'test': {'y_true': test_data[target], 'y_pred': y_pred_test_mediation}},
        'direct': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_direct},
                   'test': {'y_true': test_data[target], 'y_pred': y_pred_test_direct}},
        'hybrid': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_hybrid},
                   'test': {'y_true': test_data[target], 'y_pred': y_pred_test_hybrid}}
    }

    models = {
        'width': model_width,
        'height': model_height,
        'mediation': model_mediation,
        'direct': model_direct,
        'hybrid': model_hybrid,
        'features': {
            'predictors': predictors,
            'mediators': mediators,
            'hybrid': hybrid_features,
            'mediation': list(train_mediation_features.columns)  # Explicitly list 5 features of mediation model
        }
    }

    return results, predictions, models, test_data


# Create output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# Start timing
start_time = time.time()

# Generate unique log filename
base_log_file = "GA_RF_Intermediary(multi seeds)_CV_model_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# Redirect output stream
sys.stdout = Logger(log_file)

print(f"Starting mediation effect GA-RF model analysis, logs will be saved to {log_file}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# Load data
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("Data preparation complete:")
print(f"- Included features: {list(data.columns)}")
print(f"- Printing parameters (independent variables): {predictors} [T (temperature), V (speed), F (feed rate)]")
print(f"- Mediator variables: {mediators} [H (height), W (width)]")
print(f"- Target variable: 'Experiment_mean(MPa)' [mechanical modulus]")
print("\nModel structure details:")
print("1. Direct model: [T, V, F] → mechanical modulus (fast preliminary performance prediction)")
print("2. Mediation model (nested): [T, V, F] → [Ŵ, Ĥ] → mechanical modulus (high-precision prediction + analytical mediation mechanism)")
print("3. Hybrid model: [T, V, F, W_true, H_true] → mechanical modulus (verify gain potential of geometric features)")

# Store all experiment results
all_results = []
all_predictions = []
final_models = None
test_data = None

# Run multiple experiments (9 experiments consistent with RF version)
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"Starting experiment {i + 1}/{len(seeds)}, seed value: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, test = run_ga_mediation_experiment(seed, data)
    all_results.append(results)
    all_predictions.append(predictions)

    # Save models and test data from last experiment for visualization
    if i == len(seeds) - 1:
        final_models = models
        test_data = test

    # Output evaluation results for this experiment
    print(f"\nEvaluation results for experiment {i + 1} (4 decimal places):")
    print(f"Mediation model - Test set R²: {results['mediation']['test']['R2']:.4f}")
    print(f"Direct model - Test set R²: {results['direct']['test']['R2']:.4f}")
    print(f"Hybrid model - Test set R²: {results['hybrid']['test']['R2']:.4f}")


# Calculate mean, standard deviation and coefficient of variation for all experiments
def calculate_stats(results_list):
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

    # Collect metric values from all seeds
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in stats_results[model_type][dataset_type]:
                values = [res[model_type][dataset_type][metric] for res in results_list]
                mean_val = np.mean(values)
                std_val = np.std(values)
                # Coefficient of variation = standard deviation / mean (handle case when mean is 0)
                cv_val = std_val / mean_val if mean_val != 0 else 0

                stats_results[model_type][dataset_type][metric]['mean'] = mean_val
                stats_results[model_type][dataset_type][metric]['std'] = std_val
                stats_results[model_type][dataset_type][metric]['cv'] = cv_val

    return stats_results


# Calculate statistical results
stats_results = calculate_stats(all_results)

# Output statistical results
print('\n' + '=' * 50)
print("Statistical evaluation results from multiple experiments (4 decimal places):")
print("Format: mean ± standard deviation (coefficient of variation)")
print('=' * 50)

model_info = [
    ('mediation', 'Mediation Model', 'printing parameters→dimensions→mechanical modulus', 'research optimization, clinical personalized customization'),
    ('direct', 'Direct Model', 'printing parameters directly→mechanical modulus', 'low precision requirements, rapid parameter screening'),
    ('hybrid', 'Hybrid Model', 'printing parameters+dimensions→mechanical modulus', 'model performance upper limit reference')
]

for model_type, model_name, param_desc, scenario in model_info:
    print(f"\n{model_name}:")
    print(f"  Parameter combination: {param_desc}")
    print(f"  Application scenario: {scenario}")
    print("  Validation set evaluation:")
    print(
        f"    Mean Squared Error (MSE): {stats_results[model_type]['val']['MSE']['mean']:.4f} ± {stats_results[model_type]['val']['MSE']['std']:.4f} ({stats_results[model_type]['val']['MSE']['cv']:.2%})")
    print(
        f"    Coefficient of Determination (R2): {stats_results[model_type]['val']['R2']['mean']:.4f} ± {stats_results[model_type]['val']['R2']['std']:.4f} ({stats_results[model_type]['val']['R2']['cv']:.2%})")
    print("  Test set evaluation:")
    print(
        f"    Mean Squared Error (MSE): {stats_results[model_type]['test']['MSE']['mean']:.4f} ± {stats_results[model_type]['test']['MSE']['std']:.4f} ({stats_results[model_type]['test']['MSE']['cv']:.2%})")
    print(
        f"    Coefficient of Determination (R2): {stats_results[model_type]['test']['R2']['mean']:.4f} ± {stats_results[model_type]['test']['R2']['std']:.4f} ({stats_results[model_type]['test']['R2']['cv']:.2%})")
    print(
        f"    Mean Absolute Error (MAE): {stats_results[model_type]['test']['MAE']['mean']:.4f} ± {stats_results[model_type]['test']['MAE']['std']:.4f} ({stats_results[model_type]['test']['MAE']['cv']:.2%})")
    print(
        f"    Median Absolute Error (MedAE): {stats_results[model_type]['test']['MedAE']['mean']:.4f} ± {stats_results[model_type]['test']['MedAE']['std']:.4f} ({stats_results[model_type]['test']['MedAE']['cv']:.2%})")


# Mediation effect analysis
def calculate_mediation_effect(stats_results):
    total_effect = stats_results['direct']['test']['R2']['mean']
    direct_effect = stats_results['hybrid']['test']['R2']['mean'] - stats_results['mediation']['test']['R2']['mean']
    mediation_effect = total_effect - direct_effect
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


mediation_stats = calculate_mediation_effect(stats_results)

print('\n' + '=' * 50)
print("Mediation effect analysis results:")
print('=' * 50)
print(f"Total effect (direct model R²): {mediation_stats['total_effect']:.4f}")
print(f"Direct effect (after controlling for dimensions): {mediation_stats['direct_effect']:.4f}")
print(f"Mediation effect (through dimensions): {mediation_stats['mediation_effect']:.4f}")
print(f"Mediation ratio (mediation effect/total effect): {mediation_stats['mediation_ratio']:.2%}")

# Plot feature importance analysis
plt.figure(figsize=(18, 10))

# 1. Width predictor feature importance
plt.subplot(2, 3, 1)
feature_importance_width = final_models['width'].feature_importances_
feature_names = final_models['features']['predictors']
plt.bar(feature_names, feature_importance_width)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Width Predictor - Feature Importance')
plt.xticks(rotation=45)

# 2. Height predictor feature importance
plt.subplot(2, 3, 2)
feature_importance_height = final_models['height'].feature_importances_
plt.bar(feature_names, feature_importance_height)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Height Predictor - Feature Importance')
plt.xticks(rotation=45)

# 3. Mediation model feature importance
plt.subplot(2, 3, 3)
feature_importance_mediation = final_models['mediation'].feature_importances_
mediation_feature_names = final_models['features']['mediation']
plt.bar(mediation_feature_names, feature_importance_mediation)
plt.xlabel('Mediated Features')
plt.ylabel('Importance')
plt.title('Mediation Model - Feature Importance')
plt.xticks(rotation=45)

# 4. Direct model feature importance
plt.subplot(2, 3, 4)
feature_importance_direct = final_models['direct'].feature_importances_
plt.bar(feature_names, feature_importance_direct)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Direct Model - Feature Importance')
plt.xticks(rotation=45)

# 5. Hybrid model feature importance
plt.subplot(2, 3, 5)
feature_importance_hybrid = final_models['hybrid'].feature_importances_
hybrid_feature_names = final_models['features']['hybrid']
plt.bar(hybrid_feature_names, feature_importance_hybrid)
plt.xlabel('Hybrid Features')
plt.ylabel('Importance')
plt.title('Hybrid Model - Feature Importance')
plt.xticks(rotation=45)

# Adjust layout and save image
plt.tight_layout()
feature_importance_plot_path = get_unique_filename(os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_CV_feature_importance_comparison.png"))
plt.savefig(feature_importance_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFeature importance comparison plot saved to: {feature_importance_plot_path}")

# Plot predicted vs actual values comparison
plt.figure(figsize=(18, 6))

# Mediation model
plt.subplot(1, 3, 1)
plt.scatter(test_data[target], all_predictions[-1]['mediation']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Mediation Model (R² = {stats_results["mediation"]["test"]["R2"]["mean"]:.4f})')

# Direct model
plt.subplot(1, 3, 2)
plt.scatter(test_data[target], all_predictions[-1]['direct']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Direct Model (R² = {stats_results["direct"]["test"]["R2"]["mean"]:.4f})')

# Hybrid model
plt.subplot(1, 3, 3)
plt.scatter(test_data[target], all_predictions[-1]['hybrid']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Hybrid Model (R² = {stats_results["hybrid"]["test"]["R2"]["mean"]:.4f})')

# Adjust layout and save image
plt.tight_layout()
prediction_comparison_plot_path = get_unique_filename(os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_CV_prediction_comparison.png"))
plt.savefig(prediction_comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Prediction comparison plot saved to: {prediction_comparison_plot_path}")

# Calculate and output total runtime
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {str(timedelta(seconds=int(total_time)))}")
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\nAnalysis complete! All results saved to outputs folder.")

# Restore standard output
sys.stdout = sys.stdout.terminal
