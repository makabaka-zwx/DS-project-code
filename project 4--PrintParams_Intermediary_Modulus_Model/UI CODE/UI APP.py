from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from io import BytesIO
import base64

# --------------------------
# Global Initialization Configuration
# --------------------------
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # Maintain JSON return order

# Set Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Model path configuration (corresponding to 9 models: 3 each for GA-RF/Regression/RF)
MODEL_DIR = "../models/"
# Define 9 model metadata (including feature ranges, precision, descriptions)
MODELS_META = {
    "GA-RF Models": [
        {
            "key": "ga_direct",
            "name": "GA-RF-Direct Model",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_direct_model.pkl"),
            "desc": "Genetic algorithm optimized RF, input only (T/V/F), suitable for high-precision rapid screening",
            "acc": "Test set R²≈0.62, MSE≈2033.17",
            "speed": "Prediction time <0.09s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)",
                                         "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)",
                                   "precision": "10mm/min"}
            }
        },
        {
            "key": "ga_mediation",
            "name": "GA-RF-Mediation Model",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_mediation_model.pkl"),
            "desc": "Genetic algorithm optimized RF, input (T/V/F + predicted width/height), suitable for core process optimization",
            "acc": "Test set R²≈0.68, MSE≈1553.29",
            "speed": "Prediction time <0.12s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Predicted Width (mm)", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Predicted Height (mm)", "precision": "0.01mm"}
            }
        },
        {
            "key": "ga_hybrid",
            "name": "GA-RF-Hybrid Model",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_hybrid_model.pkl"),
            "desc": "Genetic algorithm optimized RF, input (T/V/F + actual width/height), suitable for high-precision verification",
            "acc": "Test set R²≈0.67, MSE≈1587.83",
            "speed": "Prediction time <0.1s/sample",
    "features": ["printing_temperature", "feed_rate", "printing_speed", "Height", "Width"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Actual Width (mm)", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Actual Height (mm)", "precision": "0.01mm"}
            }
        }
    ],
    "Regression Models": [
        {
            "key": "reg_direct",
            "name": "Regression-Direct Model",
            "path": "../model_parameters/best_per_model/direct_linear/best_params_seed_2520153.pkl",
            "desc": "Input only printing parameters (T/V/F), fast initial prediction, suitable for low-precision requirements",
            "acc": "Test set R²≈0.71, MSE≈1173.55",
            "speed": "Prediction time <0.05s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"}
            }
        },
        {
            "key": "reg_mediation",
            "name": "Regression-Mediation Model",
            "path": "../model_parameters/best_per_model/mediation_model_degree1/best_params_seed_2520153.pkl",
            "desc": "Input (T/V/F + predicted width/height), high-precision prediction + mediation mechanism analysis, suitable for research optimization",
            "acc": "Test set R²≈0.71, MSE≈1173.55",
            "speed": "Prediction time <0.08s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Predicted Width (mm)", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Predicted Height (mm)", "precision": "0.01mm"}
            }
        },
        {
            "key": "reg_hybrid",
            "name": "Regression-Hybrid Model",
            "path": "../model_parameters/best_per_model/hybrid_linear/best_params_seed_2520156.pkl",
            "desc": "Input (T/V/F + actual width/height), performance upper limit reference, suitable for model verification",
            "acc": "Test set R²≈0.42, MSE≈3814.94",
            "speed": "Prediction time <0.06s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "Width", "Height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Actual Width (mm)", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Actual Height (mm)", "precision": "0.01mm"}
            }
        }
    ],
    "Random Forest Models": [
        {
            "key": "rf_direct",
            "name": "RF-Direct Model",
            "path": os.path.join(MODEL_DIR, "best_direct_model_seed_2520156.pkl"),
            "desc": "Input only printing parameters (T/V/F), balanced precision and speed, suitable for batch parameter screening",
            "acc": "Test set R²≈0.73, MSE≈1579.19",
            "speed": "Prediction time <0.07s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"}
            }
        },
        {
            "key": "rf_mediation",
            "name": "RF-Mediation Model",
            "path": os.path.join(MODEL_DIR, "best_mediation_model_seed_2520155.pkl"),
            "desc": "Input (T/V/F + predicted width/height), high precision + interpretability, suitable for clinical personalized customization",
            "acc": "Test set R²≈0.72, MSE≈1455.33",
            "speed": "Prediction time <0.1s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Predicted Width (mm)", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Predicted Height (mm)", "precision": "0.01mm"}
            }
        },
        {
            "key": "rf_hybrid",
            "name": "RF-Hybrid Model",
            "path": os.path.join(MODEL_DIR, "best_hybrid_model_seed_2520154.pkl"),
            "desc": "Input (T/V/F + actual width/height), performance upper limit reference, suitable for model reliability verification",
            "acc": "Test set R²≈0.67, MSE≈1596.16",
            "speed": "Prediction time <0.08s/sample",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "Width", "Height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "Nozzle Temperature (℃)", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "Feed Rate (%)", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "Printing Speed (mm/min)", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "Actual Width (mm)", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "Actual Height (mm)", "precision": "0.01mm"}
            }
        }
    ]
}

# Target elastic modulus and tolerance configuration
TARGET_E_CONFIG = {
    "min": 50.0, "max": 500.0, "step": 0.01, "desc": "Elastic Modulus (MPa)", "precision": "0.01MPa"
}
TOLERANCE_CONFIG = {
    "min": 0.1, "max": 100.0, "step": 0.1, "desc": "Tolerance (MPa)", "precision": "0.1MPa", "default": 1.0
}

# Load all models (preload at startup, cache model objects)
loaded_models = {}
for model_type, models in MODELS_META.items():
    for model in models:
        model_key = model["key"]
        model_path = model["path"]
        try:
            if os.path.exists(model_path):
                # Compatible with .pkl and .joblib formats
                if model_path.endswith(".joblib"):
                    loaded_models[model_key] = joblib.load(model_path)
                else:
                    loaded_obj = joblib.load(model_path)
                    loaded_models[model_key] = loaded_obj["model"] if isinstance(loaded_obj, dict) else loaded_obj
                model["loaded"] = True
            else:
                model["loaded"] = False
                model["error"] = "Model file does not exist"
        except Exception as e:
            model["loaded"] = False
            model["error"] = f"Loading failed: {str(e)[:50]}"  # Truncate to first 50 characters to avoid overlength

# --------------------------
# Utility Functions (Precomputation, Visualization, Parameter Filtering)
# --------------------------
def precompute_sample_data(model_key, n_samples=10000):
    """Pre-generate sample data (random sampling according to model feature ranges, cached at startup)"""
    # Find model metadata
    model_meta = None
    for model_type, models in MODELS_META.items():
        for m in models:
            if m["key"] == model_key:
                model_meta = m
                break
        if model_meta:
            break
    if not model_meta or not model_meta["loaded"]:
        print(f"❌ Precomputation failed: Model {model_key} not loaded or metadata missing")  # New log
        return None

    # Generate sampling points according to feature ranges
    features = model_meta["features"]
    feature_ranges = model_meta["feature_ranges"]
    samples = []
    for feat in features:  # Generate in features order to ensure consistency with training
        if feat not in feature_ranges:
            raise ValueError(f"Feature {feat} not configured in feature_ranges, please check MODELS_META")
        feat_cfg = feature_ranges[feat]
        # Generate sampling points by step (avoid out-of-range due to random sampling)
        if feat_cfg["step"] > 0:
            # Generate uniform sampling points
            sample = np.arange(feat_cfg["min"], feat_cfg["max"] + feat_cfg["step"], feat_cfg["step"])
            # If insufficient sampling points, supplement with random sampling (but limited to feature range)
            if len(sample) < n_samples:
                supplement_sample = np.random.uniform(feat_cfg["min"], feat_cfg["max"], n_samples - len(sample))
                sample = np.concatenate([sample, supplement_sample])
        else:
            # Direct random sampling when no step is set
            sample = np.random.uniform(feat_cfg["min"], feat_cfg["max"], n_samples)
        # Control sample quantity (avoid exceeding n_samples)
        sample = sample[:n_samples]
        samples.append(sample)
        print(f"✅ Generated {feat} feature sampling points, quantity: {len(sample)}")  # New log to check feature sampling status

    # Combine sampling data (feature names and order are exactly consistent with features)
    sample_df = pd.DataFrame(np.column_stack(samples), columns=features)
    print(f"✅ Completed combining sampling data, features: {features}, sample size: {len(sample_df)}")  # New log

    # Load model and predict (ensure input features are consistent with training)
    model = loaded_models[model_key]
    try:
        sample_df["pred_E"] = model.predict(sample_df[features])  # Explicitly pass features consistent with training
        print(f"✅ Model {model_key} prediction completed, prediction column 'pred_E' added")  # New log
    except ValueError as e:
        # If error still occurs, print feature comparison information for debugging
        if hasattr(model, 'feature_names_in_'):
            train_features = model.feature_names_in_
            predict_features = sample_df.columns.tolist()
            print(f"Model {model_key} training features: {train_features}")
            print(f"Model {model_key} prediction features: {predict_features}")
            print(f"Are feature names consistent: {train_features == predict_features}")
            print(f"Are feature counts consistent: {len(train_features) == len(predict_features)}")
        raise e  # Re-throw error for easier localization
    # Sort by elastic modulus and return
    sample_df = sample_df.sort_values("pred_E").reset_index(drop=True)
    return sample_df


# Precompute sampling data for all loaded models (execute at startup)
precomputed_data = {}
for model_type, models in MODELS_META.items():
    for model in models:
        if model["loaded"]:
            print(f"Precomputing sampling data for model {model['name']}...")
            try:
                precomputed_data[model["key"]] = precompute_sample_data(model["key"], n_samples=10000)
                print(f"✅ Model {model['name']} precomputation completed")
            except ValueError as e:
                # For feature mismatch errors, print more detailed comparison information
                if "feature names" in str(e).lower():
                    model_obj = loaded_models[model["key"]]
                    if hasattr(model_obj, 'feature_names_in_'):
                        train_feats = model_obj.feature_names_in_.tolist()
                        config_feats = model["features"]
                        print(f"❌ Feature mismatch details:")
                        print(f"  Training feature order: {train_feats}")
                        print(f"  Configuration feature order: {config_feats}")
                precomputed_data[model["key"]] = None
                print(f"❌ Model {model['name']} precomputation failed: {str(e)}")
        else:
            print(f"❌ Model {model['name']} not loaded, skipping precomputation (reason: {model['error']})")

def filter_top10_params(model_key, target_E_min, target_E_max, tolerance=1.0):
    """Filter Top-10 parameter combinations by target elastic modulus"""
    # Check precomputed data
    if model_key not in precomputed_data:
        print(f"❌ Filtering failed: Model {model_key} precomputed data missing")  # New log
        return {"success": False, "msg": "Model has no precomputed sampling data"}

    sample_df = precomputed_data[model_key]
    print(f"✅ Starting filtering, model {model_key} precomputed data size: {len(sample_df)}")  # New log
    # Calculate filtering range
    lower_bound = target_E_min - tolerance
    upper_bound = target_E_max + tolerance
    # Filter eligible samples
    filtered = sample_df[
        (sample_df["pred_E"] >= lower_bound) &
        (sample_df["pred_E"] <= upper_bound)
    ].copy()
    print(f"✅ Filtered data size: {len(filtered)}, filtering range: {lower_bound}-{upper_bound}")  # New log
    if len(filtered) == 0:
        return {"success": False, "msg": f"No eligible parameter combinations found ({lower_bound:.2f}-{upper_bound:.2f}MPa)"}

    # Calculate average difference from target range (for sorting)
    target_avg = (target_E_min + target_E_max) / 2
    filtered["E_diff"] = np.abs(filtered["pred_E"] - target_avg)
    # Sort by difference in ascending order, take Top-10
    top10 = filtered.sort_values("E_diff").head(10).reset_index(drop=True)
    print(f"✅ Top-10 data generated, quantity: {len(top10)}")  # New log

    # Calculate parameter statistics (min/max/avg)
    model_meta = next(m for mt in MODELS_META.values() for m in mt if m["key"] == model_key)
    features = model_meta["features"]
    param_stats = {}
    for feat in features:
        feat_cfg = model_meta["feature_ranges"][feat]
        param_stats[feat] = {
            "desc": feat_cfg["desc"],
            "min": round(top10[feat].min(), 2),
            "max": round(top10[feat].max(), 2),
            "avg": round(top10[feat].mean(), 2),
            "count": len(top10)
        }

    # Format Top-10 results (fix: unify variable names and precision logic)
    result_list = []
    for idx, row in top10.iterrows():
        res = {"rank": idx + 1}
        for feat in features:
            feat_cfg = model_meta["feature_ranges"][feat]
            # Retain decimals according to feature precision (integer precision: ℃/mm/min; two decimals: other features)
            if "℃" in feat_cfg["desc"] or "mm/min" in feat_cfg["desc"]:
                res[feat] = round(row[feat], 0)  # Integer precision
            else:
                res[feat] = round(row[feat], 2)  # Two decimal places
        res["pred_E"] = round(row["pred_E"], 2)  # Elastic modulus retains two decimal places
        res["E_diff"] = round(row["E_diff"], 2)  # Difference retains two decimal places
        result_list.append(res)
    # print(result_list,param_stats,len(filtered),{"min": target_E_min, "max": target_E_max, "avg": target_avg})
    return {
        "success": True,
        "top10": result_list,
        "param_stats": param_stats,
        "total_filtered": len(filtered),
        "target_range": {"min": target_E_min, "max": target_E_max, "avg": target_avg}
    }


def plot_E_distribution(model_key, target_E_min, target_E_max, tolerance=1.0):
    """Plot elastic modulus distribution histogram (full distribution vs eligible distribution)"""
    if model_key not in precomputed_data:
        return None

    # Supplement: Get current model's metadata (for chart title)
    model_meta = None
    for model_type, models in MODELS_META.items():
        for m in models:
            if m["key"] == model_key:
                model_meta = m
                break
        if model_meta:
            break
    if not model_meta:
        return None

    sample_df = precomputed_data[model_key]
    lower_bound = target_E_min - tolerance
    upper_bound = target_E_max + tolerance
    # Filter eligible elastic moduli
    filtered_E = sample_df[
        (sample_df["pred_E"] >= lower_bound) &
        (sample_df["pred_E"] <= upper_bound)
    ]["pred_E"]

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 4))
    # Full distribution
    ax.hist(sample_df["pred_E"], bins=50, alpha=0.5, label="Full Prediction Distribution", color="#1f77b4")
    # Eligible distribution
    ax.hist(filtered_E, bins=20, alpha=0.8, label=f"Eligible Distribution ({lower_bound:.1f}-{upper_bound:.1f}MPa)", color="#ff7f0e")
    # Mark target range
    ax.axvspan(target_E_min, target_E_max, alpha=0.3, color="green", label=f"Target Range ({target_E_min}-{target_E_max}MPa)")

    ax.set_xlabel("Predicted Elastic Modulus (MPa)")
    ax.set_ylabel("Sample Quantity")
    ax.set_title(f"{model_meta['name']} Elastic Modulus Prediction Distribution Comparison")  # model_meta is now defined
    ax.legend()
    plt.tight_layout()

    # Convert to base64 encoding (for frontend display)
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return img_base64

def plot_feature_importance(model_key):
    """Plot feature importance chart (only supports RF/GA-RF models)"""
    # Check model type and loading status
    model_meta = None
    for model_type, models in MODELS_META.items():
        for m in models:
            if m["key"] == model_key:
                model_meta = m
                break
        if model_meta:
            break
    if not model_meta or not model_meta["loaded"]:
        return None
    if "RF" not in model_meta["name"]:
        return None

    model = loaded_models[model_key]
    features = model_meta["features"]
    importances = model.feature_importances_

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(features))
    # Map feature names to Chinese descriptions
    feat_names_cn = [model_meta["feature_ranges"][f]["desc"] for f in features]
    ax.barh(y_pos, importances, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(features)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names_cn)
    ax.set_xlabel("Feature Importance Weight")
    ax.set_title(f"{model_meta['name']} Feature Importance Analysis")

    # Add value labels
    for i, v in enumerate(importances):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")

    plt.tight_layout()
    # Convert to base64 encoding
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return img_base64

# --------------------------
# Flask Routes (Frontend-Backend Interaction)
# --------------------------
@app.route('/')
def index():
    """Render homepage (pass model metadata, input configuration)"""
    return render_template(
        "index.html",
        models_meta=MODELS_META,
        target_e_cfg=TARGET_E_CONFIG,
        tolerance_cfg=TOLERANCE_CONFIG
    )

@app.route('/api/filter-params', methods=['POST'])
def api_filter_params():
    """API: Filter Top-10 parameter combinations (receive JSON requests)"""
    data = request.json
    try:
        # Parse request parameters and validate
        model_key = data.get("model_key")
        target_E_min = float(data.get("target_E_min", 0))
        target_E_max = float(data.get("target_E_max", 0))
        tolerance = float(data.get("tolerance", TOLERANCE_CONFIG["default"]))

        print(f"✅ Request received: model_key={model_key}, target_E_min={target_E_min}, target_E_max={target_E_max}, tolerance={tolerance}")  # New log

        # Legality verification
        if not model_key:
            return jsonify({"success": False, "msg": "Please select a model"})
        if target_E_min < TARGET_E_CONFIG["min"] or target_E_max > TARGET_E_CONFIG["max"]:
            return jsonify({"success": False, "msg": f"Elastic modulus must be within {TARGET_E_CONFIG['min']}-{TARGET_E_CONFIG['max']}MPa range"})
        if target_E_min >= target_E_max:
            return jsonify({"success": False, "msg": "Minimum value must be less than maximum value"})
        if tolerance < TOLERANCE_CONFIG["min"] or tolerance > TOLERANCE_CONFIG["max"]:
            return jsonify({"success": False, "msg": f"Tolerance must be within {TOLERANCE_CONFIG['min']}-{TOLERANCE_CONFIG['max']}MPa range"})

        # Filter parameters and generate visualization
        filter_result = filter_top10_params(model_key, target_E_min, target_E_max, tolerance)
        if not filter_result["success"]:
            print(f"❌ Filtering failed: {filter_result['msg']}")  # New log
            return jsonify(filter_result)

        # Generate charts
        e_dist_img = plot_E_distribution(model_key, target_E_min, target_E_max, tolerance)
        fi_img = plot_feature_importance(model_key)

        # Supplement model performance backtracking information
        model_meta = next(m for mt in MODELS_META.values() for m in mt if m["key"] == model_key)
        filter_result["model_perf"] = {
            "name": model_meta["name"],
            "acc": model_meta["acc"],
            "speed": model_meta["speed"],
            "desc": model_meta["desc"]
        }
        filter_result["e_dist_img"] = e_dist_img
        filter_result["fi_img"] = fi_img

        print(f"✅ Request processing completed, returned data size: top10={len(filter_result['top10'])}, e_dist_img={'Yes' if e_dist_img else 'No'}, fi_img={'Yes' if fi_img else 'No'}")  # New log
        return jsonify(filter_result)
    except Exception as e:
        print(f"❌ Request processing exception: {str(e)}")  # New log
        return jsonify({"success": False, "msg": f"Request processing failed: {str(e)}"})

# --------------------------
# Start Service
# --------------------------
if __name__ == "__main__":
    # Ensure template directory exists
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created templates directory, please place index.html in this directory")
    # Start Flask service (allow LAN access)
    app.run(host="0.0.0.0", port=5001, debug=True)