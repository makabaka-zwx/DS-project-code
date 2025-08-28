from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from io import BytesIO
import base64

# --------------------------
# 全局初始化配置
# --------------------------
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # 保持JSON返回顺序

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 模型路径配置（对应9种模型：GA-RF/回归/RF各3种）
MODEL_DIR = "../models/"
# 定义9种模型元数据（含特征范围、精度、描述）
MODELS_META = {
    "GA-RF模型": [
        {
            "key": "ga_direct",
            "name": "GA-RF-直接模型",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_direct_model.pkl"),
            "desc": "遗传算法优化RF，仅输入（T/V/F），适用于高精度快速筛选",
            "acc": "测试集R²≈0.62，MSE≈2033.17",
            "speed": "预测时间<0.09s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）",
                                         "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）",
                                   "precision": "10mm/min"}
            }
        },
        {
            "key": "ga_mediation",
            "name": "GA-RF-中介模型",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_mediation_model.pkl"),
            "desc": "遗传算法优化RF，输入（T/V/F+预测宽高），适用于核心工艺优化",
            "acc": "测试集R²≈0.68，MSE≈1553.29",
            "speed": "预测时间<0.12s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "预测宽度（mm）", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "预测高度（mm）", "precision": "0.01mm"}
            }
        },
        {
            "key": "ga_hybrid",
            "name": "GA-RF-混合模型",
            "path": os.path.join(MODEL_DIR, "GA_RF_Intermediary(multi seeds)_CV_hybrid_model.pkl"),
            "desc": "遗传算法优化RF，输入（T/V/F+真实宽高），适用于高精度验证",
            "acc": "测试集R²≈0.67，MSE≈1587.83",
            "speed": "预测时间<0.1s/样本",
    "features": ["printing_temperature", "feed_rate", "printing_speed", "Height", "Width"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "真实宽度（mm）", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "真实高度（mm）", "precision": "0.01mm"}
            }
        }
    ],
    "回归模型": [
        {
            "key": "reg_direct",
            "name": "回归-直接模型",
            "path": "../model_parameters/best_per_model/direct_linear/best_params_seed_2520153.pkl",
            "desc": "仅输入打印参数（T/V/F），快速初步预测，适用于低精度需求",
            "acc": "测试集R²≈0.71，MSE≈1173.55",
            "speed": "预测时间<0.05s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"}
            }
        },
        {
            "key": "reg_mediation",
            "name": "回归-中介模型",
            "path": "../model_parameters/best_per_model/mediation_model_degree1/best_params_seed_2520153.pkl",
            "desc": "输入（T/V/F+预测宽高），高精度预测+中介机制解析，适用于科研优化",
            "acc": "测试集R²≈0.71，MSE≈1173.55",
            "speed": "预测时间<0.08s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "预测宽度（mm）", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "预测高度（mm）", "precision": "0.01mm"}
            }
        },
        {
            "key": "reg_hybrid",
            "name": "回归-混合模型",
            "path": "../model_parameters/best_per_model/hybrid_linear/best_params_seed_2520156.pkl",
            "desc": "输入（T/V/F+真实宽高），性能上限参照，适用于模型验证",
            "acc": "测试集R²≈0.42，MSE≈3814.94",
            "speed": "预测时间<0.06s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "Width", "Height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "真实宽度（mm）", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "真实高度（mm）", "precision": "0.01mm"}
            }
        }
    ],
    "随机森林模型": [
        {
            "key": "rf_direct",
            "name": "RF-直接模型",
            "path": os.path.join(MODEL_DIR, "best_direct_model_seed_2520156.pkl"),
            "desc": "仅输入打印参数（T/V/F），平衡精度与速度，适用于批量参数筛选",
            "acc": "测试集R²≈0.73，MSE≈1579.19",
            "speed": "预测时间<0.07s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"}
            }
        },
        {
            "key": "rf_mediation",
            "name": "RF-中介模型",
            "path": os.path.join(MODEL_DIR, "best_mediation_model_seed_2520155.pkl"),
            "desc": "输入（T/V/F+预测宽高），高精度+可解释性，适用于临床个性化定制",
            "acc": "测试集R²≈0.72，MSE≈1455.33",
            "speed": "预测时间<0.1s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "predicted_width", "predicted_height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "predicted_width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "预测宽度（mm）", "precision": "0.01mm"},
                "predicted_height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "预测高度（mm）", "precision": "0.01mm"}
            }
        },
        {
            "key": "rf_hybrid",
            "name": "RF-混合模型",
            "path": os.path.join(MODEL_DIR, "best_hybrid_model_seed_2520154.pkl"),
            "desc": "输入（T/V/F+真实宽高），性能上限参照，适用于模型可靠性验证",
            "acc": "测试集R²≈0.67，MSE≈1596.16",
            "speed": "预测时间<0.08s/样本",
            "features": ["printing_temperature", "feed_rate", "printing_speed", "Width", "Height"],
            "feature_ranges": {
                "printing_temperature": {"min": 180.0, "max": 250.0, "step": 1.0, "desc": "喷嘴温度（℃）", "precision": "1℃"},
                "feed_rate": {"min": 15.0, "max": 42.0, "step": 0.1, "desc": "进给率（%）", "precision": "0.1%"},
                "printing_speed": {"min": 100.0, "max": 4100.0, "step": 10.0, "desc": "打印速度（mm/min）", "precision": "10mm/min"},
                "Width": {"min": 0.3, "max": 0.6, "step": 0.01, "desc": "真实宽度（mm）", "precision": "0.01mm"},
                "Height": {"min": 0.25, "max": 0.55, "step": 0.01, "desc": "真实高度（mm）", "precision": "0.01mm"}
            }
        }
    ]
}

# 目标弹性模量与误差容忍度配置
TARGET_E_CONFIG = {
    "min": 50.0, "max": 500.0, "step": 0.01, "desc": "弹性模量（MPa）", "precision": "0.01MPa"
}
TOLERANCE_CONFIG = {
    "min": 0.1, "max": 100.0, "step": 0.1, "desc": "误差容忍度（MPa）", "precision": "0.1MPa", "default": 1.0
}

# 加载所有模型（启动时预加载，缓存模型对象）
loaded_models = {}
for model_type, models in MODELS_META.items():
    for model in models:
        model_key = model["key"]
        model_path = model["path"]
        try:
            if os.path.exists(model_path):
                # 兼容.pkl与.joblib格式
                if model_path.endswith(".joblib"):
                    loaded_models[model_key] = joblib.load(model_path)
                else:
                    loaded_obj = joblib.load(model_path)
                    loaded_models[model_key] = loaded_obj["model"] if isinstance(loaded_obj, dict) else loaded_obj
                model["loaded"] = True
            else:
                model["loaded"] = False
                model["error"] = "模型文件不存在"
        except Exception as e:
            model["loaded"] = False
            model["error"] = f"加载失败：{str(e)[:50]}"  # 截取前50字符避免过长

# --------------------------
# 工具函数（预计算、可视化、参数筛选）
# --------------------------
def precompute_sample_data(model_key, n_samples=10000):
    """预生成采样数据（按模型特征范围随机采样，启动时缓存）"""
    # 找到模型元数据
    model_meta = None
    for model_type, models in MODELS_META.items():
        for m in models:
            if m["key"] == model_key:
                model_meta = m
                break
        if model_meta:
            break
    if not model_meta or not model_meta["loaded"]:
        print(f"❌ 预计算失败：模型 {model_key} 未加载或元数据缺失")  # 新增日志
        return None

    # 按特征范围生成采样点
    features = model_meta["features"]
    feature_ranges = model_meta["feature_ranges"]
    samples = []
    for feat in features:  # 按 features 顺序生成，确保与训练时一致
        if feat not in feature_ranges:
            raise ValueError(f"特征 {feat} 在 feature_ranges 中未配置，请检查 MODELS_META")
        feat_cfg = feature_ranges[feat]
        # 按步长生成采样点（避免随机采样导致特征范围超出）
        if feat_cfg["step"] > 0:
            # 生成均匀采样点
            sample = np.arange(feat_cfg["min"], feat_cfg["max"] + feat_cfg["step"], feat_cfg["step"])
            # 若采样点数量不足，补充随机采样（但限制在特征范围内）
            if len(sample) < n_samples:
                supplement_sample = np.random.uniform(feat_cfg["min"], feat_cfg["max"], n_samples - len(sample))
                sample = np.concatenate([sample, supplement_sample])
        else:
            # 无步长时直接随机采样
            sample = np.random.uniform(feat_cfg["min"], feat_cfg["max"], n_samples)
        # 控制采样数量（避免超出 n_samples）
        sample = sample[:n_samples]
        samples.append(sample)
        print(f"✅ 生成特征 {feat} 采样点，数量：{len(sample)}")  # 新增日志，看特征采样是否正常

    # 组合采样数据（特征名称、顺序与 features 完全一致）
    sample_df = pd.DataFrame(np.column_stack(samples), columns=features)
    print(f"✅ 组合采样数据完成，特征：{features}，样本量：{len(sample_df)}")  # 新增日志

    # 加载模型并预测（确保输入特征与训练时一致）
    model = loaded_models[model_key]
    try:
        sample_df["pred_E"] = model.predict(sample_df[features])  # 明确传入与训练时一致的特征
        print(f"✅ 模型 {model_key} 预测完成，预测列 'pred_E' 已添加")  # 新增日志
    except ValueError as e:
        # 若仍报错，打印特征对比信息，辅助调试
        if hasattr(model, 'feature_names_in_'):
            train_features = model.feature_names_in_
            predict_features = sample_df.columns.tolist()
            print(f"模型 {model_key} 训练特征：{train_features}")
            print(f"模型 {model_key} 预测特征：{predict_features}")
            print(f"特征名称是否一致：{train_features == predict_features}")
            print(f"特征数量是否一致：{len(train_features) == len(predict_features)}")
        raise e  # 重新抛出错误，便于定位
    # 按弹性模量排序并返回
    sample_df = sample_df.sort_values("pred_E").reset_index(drop=True)
    return sample_df


# 预计算所有已加载模型的采样数据（启动时执行）
precomputed_data = {}
for model_type, models in MODELS_META.items():
    for model in models:
        if model["loaded"]:
            print(f"预计算模型 {model['name']} 的采样数据...")
            try:
                precomputed_data[model["key"]] = precompute_sample_data(model["key"], n_samples=10000)
                print(f"✅ 模型 {model['name']} 预计算完成")
            except ValueError as e:
                # 针对特征不匹配错误，打印更详细的对比信息
                if "feature names" in str(e).lower():
                    model_obj = loaded_models[model["key"]]
                    if hasattr(model_obj, 'feature_names_in_'):
                        train_feats = model_obj.feature_names_in_.tolist()
                        config_feats = model["features"]
                        print(f"❌ 特征不匹配详情：")
                        print(f"  训练时特征顺序：{train_feats}")
                        print(f"  配置时特征顺序：{config_feats}")
                precomputed_data[model["key"]] = None
                print(f"❌ 模型 {model['name']} 预计算失败：{str(e)}")
        else:
            print(f"❌ 模型 {model['name']} 未加载，跳过预计算（原因：{model['error']}）")

def filter_top10_params(model_key, target_E_min, target_E_max, tolerance=1.0):
    """按目标弹性模量筛选Top-10参数组合"""
    # 检查预计算数据
    if model_key not in precomputed_data:
        print(f"❌ 筛选失败：模型 {model_key} 预计算数据缺失")  # 新增日志
        return {"success": False, "msg": "模型未预计算采样数据"}

    sample_df = precomputed_data[model_key]
    print(f"✅ 开始筛选，模型 {model_key} 预计算数据量：{len(sample_df)}")  # 新增日志
    # 计算筛选范围
    lower_bound = target_E_min - tolerance
    upper_bound = target_E_max + tolerance
    # 筛选符合条件的样本
    filtered = sample_df[
        (sample_df["pred_E"] >= lower_bound) &
        (sample_df["pred_E"] <= upper_bound)
    ].copy()
    print(f"✅ 筛选后数据量：{len(filtered)}，筛选范围：{lower_bound}-{upper_bound}")  # 新增日志
    if len(filtered) == 0:
        return {"success": False, "msg": f"未找到符合条件的参数组合（{lower_bound:.2f}-{upper_bound:.2f}MPa）"}

    # 计算与目标范围的平均差值（用于排序）
    target_avg = (target_E_min + target_E_max) / 2
    filtered["E_diff"] = np.abs(filtered["pred_E"] - target_avg)
    # 按差值升序排序，取Top-10
    top10 = filtered.sort_values("E_diff").head(10).reset_index(drop=True)
    print(f"✅ Top-10 数据生成，数量：{len(top10)}")  # 新增日志

    # 计算参数统计信息（min/max/avg）
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

    # 格式化Top-10结果（修正：统一变量名与精度逻辑）
    result_list = []
    for idx, row in top10.iterrows():
        res = {"rank": idx + 1}
        for feat in features:
            feat_cfg = model_meta["feature_ranges"][feat]
            # 按特征精度保留小数（整数精度：℃/mm/min；两位小数：其他特征）
            if "℃" in feat_cfg["desc"] or "mm/min" in feat_cfg["desc"]:
                res[feat] = round(row[feat], 0)  # 整数精度
            else:
                res[feat] = round(row[feat], 2)  # 两位小数
        res["pred_E"] = round(row["pred_E"], 2)  # 弹性模量保留两位小数
        res["E_diff"] = round(row["E_diff"], 2)  # 差值保留两位小数
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
    """绘制弹性模量分布直方图（全部分布vs符合条件分布）"""
    if model_key not in precomputed_data:
        return None

    # 补充：获取当前模型的元数据（用于图表标题）
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
    # 筛选符合条件的弹性模量
    filtered_E = sample_df[
        (sample_df["pred_E"] >= lower_bound) &
        (sample_df["pred_E"] <= upper_bound)
    ]["pred_E"]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 4))
    # 全部分布
    ax.hist(sample_df["pred_E"], bins=50, alpha=0.5, label="全量预测分布", color="#1f77b4")
    # 符合条件分布
    ax.hist(filtered_E, bins=20, alpha=0.8, label=f"符合条件分布（{lower_bound:.1f}-{upper_bound:.1f}MPa）", color="#ff7f0e")
    # 标注目标范围
    ax.axvspan(target_E_min, target_E_max, alpha=0.3, color="green", label=f"目标范围（{target_E_min}-{target_E_max}MPa）")

    ax.set_xlabel("预测弹性模量（MPa）")
    ax.set_ylabel("样本数量")
    ax.set_title(f"{model_meta['name']} 弹性模量预测分布对比")  # 现在model_meta已定义
    ax.legend()
    plt.tight_layout()

    # 转换为base64编码（前端展示）
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return img_base64

def plot_feature_importance(model_key):
    """绘制特征重要性图（仅支持RF/GA-RF模型）"""
    # 检查模型类型与加载状态
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

    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(features))
    # 映射特征名到中文描述
    feat_names_cn = [model_meta["feature_ranges"][f]["desc"] for f in features]
    ax.barh(y_pos, importances, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(features)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names_cn)
    ax.set_xlabel("特征重要性权重")
    ax.set_title(f"{model_meta['name']} 特征重要性分析")

    # 添加数值标签
    for i, v in enumerate(importances):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")

    plt.tight_layout()
    # 转换为base64编码
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return img_base64

# --------------------------
# Flask路由（前后端交互）
# --------------------------
@app.route('/')
def index():
    """渲染首页（传递模型元数据、输入配置）"""
    return render_template(
        "index.html",
        models_meta=MODELS_META,
        target_e_cfg=TARGET_E_CONFIG,
        tolerance_cfg=TOLERANCE_CONFIG
    )

@app.route('/api/filter-params', methods=['POST'])
def api_filter_params():
    """API：筛选Top-10参数组合（接收JSON请求）"""
    data = request.json
    try:
        # 解析请求参数并验证
        model_key = data.get("model_key")
        target_E_min = float(data.get("target_E_min", 0))
        target_E_max = float(data.get("target_E_max", 0))
        tolerance = float(data.get("tolerance", TOLERANCE_CONFIG["default"]))

        print(f"✅ 收到请求：model_key={model_key}, target_E_min={target_E_min}, target_E_max={target_E_max}, tolerance={tolerance}")  # 新增日志

        # 合法性校验
        if not model_key:
            return jsonify({"success": False, "msg": "请选择模型"})
        if target_E_min < TARGET_E_CONFIG["min"] or target_E_max > TARGET_E_CONFIG["max"]:
            return jsonify({"success": False, "msg": f"弹性模量需在{TARGET_E_CONFIG['min']}-{TARGET_E_CONFIG['max']}MPa范围内"})
        if target_E_min >= target_E_max:
            return jsonify({"success": False, "msg": "最小值需小于最大值"})
        if tolerance < TOLERANCE_CONFIG["min"] or tolerance > TOLERANCE_CONFIG["max"]:
            return jsonify({"success": False, "msg": f"误差容忍度需在{TOLERANCE_CONFIG['min']}-{TOLERANCE_CONFIG['max']}MPa范围内"})

        # 筛选参数并生成可视化
        filter_result = filter_top10_params(model_key, target_E_min, target_E_max, tolerance)
        if not filter_result["success"]:
            print(f"❌ 筛选失败：{filter_result['msg']}")  # 新增日志
            return jsonify(filter_result)

        # 生成图表
        e_dist_img = plot_E_distribution(model_key, target_E_min, target_E_max, tolerance)
        fi_img = plot_feature_importance(model_key)

        # 补充模型性能回溯信息
        model_meta = next(m for mt in MODELS_META.values() for m in mt if m["key"] == model_key)
        filter_result["model_perf"] = {
            "name": model_meta["name"],
            "acc": model_meta["acc"],
            "speed": model_meta["speed"],
            "desc": model_meta["desc"]
        }
        filter_result["e_dist_img"] = e_dist_img
        filter_result["fi_img"] = fi_img

        print(f"✅ 请求处理完成，返回数据量：top10={len(filter_result['top10'])}, e_dist_img={'有' if e_dist_img else '无'}, fi_img={'有' if fi_img else '无'}")  # 新增日志
        return jsonify(filter_result)
    except Exception as e:
        print(f"❌ 请求处理异常：{str(e)}")  # 新增日志
        return jsonify({"success": False, "msg": f"请求处理失败：{str(e)}"})

# --------------------------
# 启动服务
# --------------------------
if __name__ == "__main__":
    # 确保模板目录存在
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("已创建templates目录，请将index.html放入该目录")
    # 启动Flask服务（允许局域网访问）
    app.run(host="0.0.0.0", port=5001, debug=True)