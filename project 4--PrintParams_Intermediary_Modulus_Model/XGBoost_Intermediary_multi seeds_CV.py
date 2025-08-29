import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import openpyxl
import joblib  # 用于保存模型
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error


class Logger:
    """同时将输出流保存到控制台和文件"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        # 指定文件编码为utf-8
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_unique_filename(base_name):
    """生成唯一的文件名，如果已存在则添加序号后缀"""
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
    """运行中介效应实验并返回结果"""
    np.random.seed(seed)

    # 定义变量
    # 自变量：打印参数（3个）
    predictors_names = ['printing_temperature', 'feed_rate', 'printing_speed']  # T（温度）、V（速度）、F（进给率）
    # 中介变量：宽和高
    mediator_names = ['Width', 'Height']
    # 因变量：机械模量
    target_name = 'Experiment_mean(MPa)'

    # 数据集划分：基于机械模量的分层抽样
    # 训练集(70%)、验证集(15%)、测试集(15%)
    # 动态调整分箱数量，确保每个箱至少有4个样本（满足两次分层抽样要求）
    min_samples_per_bin = 4  # 确保每个分箱至少有4个样本
    max_bins = 10

    # 计算最大可能的分箱数量
    max_possible_bins = len(data) // min_samples_per_bin
    num_bins = min(max_bins, max_possible_bins)

    # 确保至少有2个分箱
    num_bins = max(2, num_bins)

    # 创建分箱
    data['target_bin'] = pd.cut(data[target_name], bins=num_bins, labels=False)

    # 检查每个分箱的样本数量，如果有分箱样本数不足，合并相邻分箱
    bin_counts = data['target_bin'].value_counts().sort_index()
    while (bin_counts < min_samples_per_bin).any():
        # 找到样本最少的分箱
        min_bin = bin_counts.idxmin()
        # 合并到相邻的分箱
        if min_bin == 0:
            data['target_bin'] = data['target_bin'].replace(1, 0)
        elif min_bin == len(bin_counts) - 1:
            data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
        else:
            # 合并到样本较多的相邻分箱
            left_count = bin_counts[min_bin - 1]
            right_count = bin_counts[min_bin + 1]
            if left_count >= right_count:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
            else:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin + 1)
        # 重新计算分箱数量
        bin_counts = data['target_bin'].value_counts().sort_index()
        # 重命名分箱标签，确保连续
        data['target_bin'] = pd.Categorical(data['target_bin']).codes
        bin_counts = data['target_bin'].value_counts().sort_index()

        # 如果只剩下一个分箱，无法再合并，只能打破循环
        if len(bin_counts) == 1:
            break

    # 输出分箱信息，验证每个分箱至少有4个样本
    print(f"分层抽样验证:")
    print(f"- 分箱数量: {len(bin_counts)}")
    for bin_idx in bin_counts.index:
        print(f"  分箱 {bin_idx}: {bin_counts[bin_idx]} 个样本")
    assert (bin_counts >= min_samples_per_bin).all() or len(bin_counts) == 1, \
        "存在样本数少于4的分箱，请检查数据或调整分箱策略"

    # 如果所有数据都在一个分箱中，使用随机抽样而非分层抽样
    stratify_param = data['target_bin'] if len(bin_counts) > 1 else None

    # 第一次分层抽样：划分训练集和临时集
    train_data, temp_data = train_test_split(
        data,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_param  # 基于目标变量分层
    )

    # 为第二次抽样准备分层参数
    if stratify_param is not None:
        stratify_param_temp = temp_data['target_bin']
        # 检查临时集中每个分箱的样本数
        temp_bin_counts = stratify_param_temp.value_counts()
        # 如果有分箱样本数不足2，改用随机抽样
        if (temp_bin_counts < 2).any():
            stratify_param_temp = None
    else:
        stratify_param_temp = None

    # 第二次分层抽样：从临时集中划分验证集和测试集
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_param_temp  # 基于目标变量分层
    )

    # 删除辅助列
    train_data = train_data.drop('target_bin', axis=1)
    val_data = val_data.drop('target_bin', axis=1)
    test_data = test_data.drop('target_bin', axis=1)

    # --------------------------
    # 1. 中介模型（嵌套）：
    #    第一层：3个打印参数 → 宽和高
    #    第二层：3个打印参数 + 预测的宽和高 → 机械模量（共5个特征）
    # --------------------------

    # 1.1 第一步：用3个打印参数预测宽和高（第一层）
    # 验证第一层输入特征数量为3
    assert len(predictors_names) == 3, f"中介模型第一层输入特征数量错误: 应为3，实际为{len(predictors_names)}"

    # 预测宽度
    xgb_width = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_width = GridSearchCV(xgb_width, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_width.fit(train_data[predictors_names], train_data['Width'])
    best_width = grid_width.best_estimator_

    # 预测高度
    xgb_height = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_height = GridSearchCV(xgb_height, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_height.fit(train_data[predictors_names], train_data['Height'])
    best_height = grid_height.best_estimator_

    # 生成预测的宽和高，与原始打印参数合并作为第二层特征（共5个特征）
    # 训练集特征
    train_data['predicted_Width'] = best_width.predict(train_data[predictors_names])
    train_data['predicted_Height'] = best_height.predict(train_data[predictors_names])
    mediation_train_features = train_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_train_features.shape[1] == 5, \
        f"中介模型第二层训练特征数量错误: 应为5，实际为{mediation_train_features.shape[1]}"

    # 验证集特征
    val_data['predicted_Width'] = best_width.predict(val_data[predictors_names])
    val_data['predicted_Height'] = best_height.predict(val_data[predictors_names])
    mediation_val_features = val_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_val_features.shape[1] == 5, \
        f"中介模型第二层验证特征数量错误: 应为5，实际为{mediation_val_features.shape[1]}"

    # 测试集特征
    test_data['predicted_Width'] = best_width.predict(test_data[predictors_names])
    test_data['predicted_Height'] = best_height.predict(test_data[predictors_names])
    mediation_test_features = test_data[predictors_names + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_test_features.shape[1] == 5, \
        f"中介模型第二层测试特征数量错误: 应为5，实际为{mediation_test_features.shape[1]}"

    # 1.2 第二步：用5个特征预测机械模量（第二层）
    xgb_mediation = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_mediation = GridSearchCV(xgb_mediation, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_mediation.fit(mediation_train_features, train_data[target_name])
    best_mediation = grid_mediation.best_estimator_

    # 在各数据集上预测
    y_pred_val_mediation = best_mediation.predict(mediation_val_features)
    y_pred_test_mediation = best_mediation.predict(mediation_test_features)

    # --------------------------
    # 2. 直接模型：3个打印参数直接预测机械模量
    # --------------------------
    # 验证输入特征数量为3
    assert len(predictors_names) == 3, f"直接模型输入特征数量错误: 应为3，实际为{len(predictors_names)}"

    xgb_direct = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_direct = GridSearchCV(xgb_direct, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_direct.fit(train_data[predictors_names], train_data[target_name])
    best_direct = grid_direct.best_estimator_

    # 在各数据集上预测
    y_pred_val_direct = best_direct.predict(val_data[predictors_names])
    y_pred_test_direct = best_direct.predict(test_data[predictors_names])

    # --------------------------
    # 3. 混合模型：3个打印参数 + 2个实际宽高 → 机械模量（共5个特征）
    # --------------------------
    hybrid_features = predictors_names + mediator_names  # T、V、F、W_true、H_true

    # 验证输入特征数量为5
    assert len(hybrid_features) == 5, f"混合模型输入特征数量错误: 应为5，实际为{len(hybrid_features)}"

    xgb_hybrid = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    grid_hybrid = GridSearchCV(xgb_hybrid, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_hybrid.fit(train_data[hybrid_features], train_data[target_name])
    best_hybrid = grid_hybrid.best_estimator_

    # 在各数据集上预测
    y_pred_val_hybrid = best_hybrid.predict(val_data[hybrid_features])
    y_pred_test_hybrid = best_hybrid.predict(test_data[hybrid_features])

    # --------------------------
    # 计算评估指标
    # --------------------------
    results = {
        'mediation': {  # 中介模型：3个打印参数→5个特征→机械模量
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
        'direct': {  # 直接模型：3个打印参数直接→机械模量
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
        'hybrid': {  # 混合模型：5个特征（3+2）→机械模量
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

    # 保存预测结果
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

    # 保存模型和最优参数
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


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)
os.makedirs("XGB_Regression_Comparison", exist_ok=True)
os.makedirs("models", exist_ok=True)  # 模型保存目录

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "XGB_Intermediary(multi seeds)_CV_effect_analysis_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始XGBoost中介效应分析实验，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据 - 不使用aspect_ratio，保留原始的宽和高
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Width', 'Height', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据准备完成:")
print(f"- 包含的特征: {list(data.columns)}")
print(f"- 打印参数(自变量): {['printing_temperature', 'feed_rate', 'printing_speed']}")
print(f"- 中介变量: {['Width', 'Height']} (同时作为打印参数的因变量)")
print(f"- 目标变量: 'Experiment_mean(MPa)'")
print(f"- 数据集划分比例: 训练集:验证集:测试集 = 7:1.5:1.5 (基于目标变量的分层抽样)")

# 定义XGBoost参数网格
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

# 定义要测试的种子值
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]  # 9次实验验证稳定性
print(f"\n将使用以下种子进行实验: {seeds}")

# 存储所有实验的结果
all_results = []
all_predictions = []
final_models = None
test_data = None

# 跟踪每个模型类型的最优模型（基于测试集R²）
best_models = {
    'mediation': {'model': None, 'r2': -np.inf, 'seed': None},
    'direct': {'model': None, 'r2': -np.inf, 'seed': None},
    'hybrid': {'model': None, 'r2': -np.inf, 'seed': None}
}

# 运行多次实验
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"开始第 {i + 1}/{len(seeds)} 次实验，种子值: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, test = run_mediation_experiment(seed, data, param_grid)
    all_results.append(results)
    all_predictions.append(predictions)

    # 跟踪每个模型类型的最优模型（基于测试集R²）
    current_r2 = results['mediation']['test']['R2']
    if current_r2 > best_models['mediation']['r2']:
        best_models['mediation'] = {'model': models, 'r2': current_r2, 'seed': seed}

    current_r2 = results['direct']['test']['R2']
    if current_r2 > best_models['direct']['r2']:
        best_models['direct'] = {'model': models, 'r2': current_r2, 'seed': seed}

    current_r2 = results['hybrid']['test']['R2']
    if current_r2 > best_models['hybrid']['r2']:
        best_models['hybrid'] = {'model': models, 'r2': current_r2, 'seed': seed}

    # 保存最后一次实验的模型和测试数据用于可视化
    if i == len(seeds) - 1:
        final_models = models
        test_data = test

    # 输出本次实验的评估结果
    print(f"\n第 {i + 1} 次实验评估结果（保留4位小数）:")
    print(f"中介模型 - 测试集R²: {results['mediation']['test']['R2']:.4f}")
    print(f"直接模型 - 测试集R²: {results['direct']['test']['R2']:.4f}")
    print(f"混合模型 - 测试集R²: {results['hybrid']['test']['R2']:.4f}")

# 保存最优的三种模型
print("\n" + "=" * 50)
print("保存最优模型:")
for model_type in ['mediation', 'direct', 'hybrid']:
    model_info = best_models[model_type]
    model_path = os.path.join("models", f"best_xgb_{model_type}_model(multi seeds)_CV_seed_{model_info['seed']}.pkl")
    joblib.dump(model_info['model'], model_path)
    print(f"- 最优{model_type}模型 (种子 {model_info['seed']}, R²={model_info['r2']:.4f}) 已保存至: {model_path}")


# 计算所有实验的平均值、标准差和变异系数
def calculate_stats(results_list):
    """计算多次实验结果的统计量：平均值 ± 标准差 和 变异系数(CV)"""
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

    # 收集所有结果
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in stats_results[model_type][dataset_type]:
                values = [res[model_type][dataset_type][metric] for res in results_list
                          if metric in res[model_type][dataset_type]]

                # 计算统计量
                stats_results[model_type][dataset_type][metric]['mean'] = np.mean(values)
                stats_results[model_type][dataset_type][metric]['std'] = np.std(values)
                # 变异系数 = 标准差 / 平均值 (处理除以零的情况)
                mean_val = stats_results[model_type][dataset_type][metric]['mean']
                if mean_val != 0:
                    stats_results[model_type][dataset_type][metric]['cv'] = (
                            stats_results[model_type][dataset_type][metric]['std'] / mean_val
                    )
                else:
                    stats_results[model_type][dataset_type][metric]['cv'] = 0

    return stats_results


# 计算统计结果
stats_results = calculate_stats(all_results)

# 输出统计结果
print('\n' + '=' * 50)
print("多次实验的统计评估结果（保留4位小数）:")
print('=' * 50)

for model_type, model_name in [
    ('mediation', '中介模型 (打印参数→宽高→机械模量)'),
    ('direct', '直接模型 (打印参数直接→机械模量)'),
    ('hybrid', '混合模型 (打印参数+宽高→机械模量)')
]:
    print(f"\n{model_name}:")
    print("验证集统计:")
    print(
        f"  均方误差 (MSE): {stats_results[model_type]['val']['MSE']['mean']:.4f} ± {stats_results[model_type]['val']['MSE']['std']:.4f}, CV={stats_results[model_type]['val']['MSE']['cv']:.4f}")
    print(
        f"  决定系数 (R2): {stats_results[model_type]['val']['R2']['mean']:.4f} ± {stats_results[model_type]['val']['R2']['std']:.4f}, CV={stats_results[model_type]['val']['R2']['cv']:.4f}")
    print("测试集统计:")
    print(
        f"  均方误差 (MSE): {stats_results[model_type]['test']['MSE']['mean']:.4f} ± {stats_results[model_type]['test']['MSE']['std']:.4f}, CV={stats_results[model_type]['test']['MSE']['cv']:.4f}")
    print(
        f"  决定系数 (R2): {stats_results[model_type]['test']['R2']['mean']:.4f} ± {stats_results[model_type]['test']['R2']['std']:.4f}, CV={stats_results[model_type]['test']['R2']['cv']:.4f}")
    print(
        f"  平均绝对误差 (MAE): {stats_results[model_type]['test']['MAE']['mean']:.4f} ± {stats_results[model_type]['test']['MAE']['std']:.4f}, CV={stats_results[model_type]['test']['MAE']['cv']:.4f}")
    print(
        f"  中位数绝对误差 (MedAE): {stats_results[model_type]['test']['MedAE']['mean']:.4f} ± {stats_results[model_type]['test']['MedAE']['std']:.4f}, CV={stats_results[model_type]['test']['MedAE']['cv']:.4f}")


# 中介效应分析：计算中介比例
def calculate_mediation_effect(stats_results):
    """计算中介效应比例"""
    # 总效应 (直接模型的效应)
    total_effect = stats_results['direct']['test']['R2']['mean']

    # 直接效应 (控制中介变量后的直接效应)
    # 这里用混合模型与中介模型的差异近似
    direct_effect = stats_results['hybrid']['test']['R2']['mean'] - stats_results['mediation']['test']['R2']['mean']

    # 中介效应 = 总效应 - 直接效应
    mediation_effect = total_effect - direct_effect

    # 中介比例
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


# 计算中介效应
mediation_stats = calculate_mediation_effect(stats_results)

print('\n' + '=' * 50)
print("中介效应分析结果:")
print('=' * 50)
print(f"总效应 (直接模型R²): {mediation_stats['total_effect']:.4f}")
print(f"直接效应 (控制宽高后): {mediation_stats['direct_effect']:.4f}")
print(f"中介效应 (通过宽高): {mediation_stats['mediation_effect']:.4f}")
print(f"中介比例 (中介效应/总效应): {mediation_stats['mediation_ratio']:.2%}")

# 输出各模型最优参数
print('\n' + '=' * 50)
print("各模型最优参数 (最后一次实验):")
print('=' * 50)
for model_type, params in final_models['best_params'].items():
    print(f"\n{model_type}模型最优参数:")
    for param, value in params.items():
        print(f"  {param}: {value}")

# 绘制特征重要性分析
plt.figure(figsize=(18, 6))

# 1. 打印参数对宽度的影响
plt.subplot(1, 3, 1)
feature_importance_width = final_models['width'].feature_importances_
feature_names = final_models['features']['predictors']
plt.bar(feature_names, feature_importance_width)
plt.xlabel('Print Parameters')
plt.ylabel('Importance')
plt.title('The Importance of the impact of Print Parameters on Width')
plt.xticks(rotation=45)

# 2. 打印参数对高度的影响
plt.subplot(1, 3, 2)
feature_importance_height = final_models['height'].feature_importances_
plt.bar(feature_names, feature_importance_height)
plt.xlabel('Print Parameters')
plt.ylabel('Importance')
plt.title('The Importance of the impact of Print Parameters on Height')
plt.xticks(rotation=45)

# 3. 中介模型第二层的5个特征对机械模量的影响
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

# 绘制三种模型的预测值与真实值对比
plt.figure(figsize=(18, 6))

model_types = ['mediation', 'direct', 'hybrid']
model_names = ['Mediated Model', 'Direct Model', 'Mixed Model']

for i, (model_type, name) in enumerate(zip(model_types, model_names), 1):
    plt.subplot(1, 3, i)

    # 获取最后一次实验的预测结果
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

# 绘制三种模型的评估指标对比
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

    # 添加数值标签
    for j, v in enumerate(values):
        plt.text(j, v + 0.01, f'{v:.4f}', ha='center')

    # R2指标范围限制在0-1
    if metric == 'R2':
        plt.ylim(0, 1)

    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("XGB_Regression_Comparison", "XGB_Intermediary(multi seeds)_CV_model_metrics_comparison.png"), dpi=300)
plt.show()

# 导出所有预测结果和统计指标到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "XGB_Intermediary(multi seeds)_CV_analysis_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 导出每次实验的预测结果
    for exp_idx, predictions in enumerate(all_predictions):
        for model_type in ['mediation', 'direct', 'hybrid']:
            for dataset_type in ['val', 'test']:
                df = pd.DataFrame({
                    'true_values': predictions[model_type][dataset_type]['y_true'],
                    'predicted_values': predictions[model_type][dataset_type]['y_pred']
                })
                sheet_name = f'exp_{exp_idx + 1}_{model_type}_{dataset_type}'
                # 确保工作表名称不超过31个字符（Excel限制）
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    # 导出统计结果
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

    # 导出中介效应分析结果
    mediation_df = pd.DataFrame([mediation_stats])
    mediation_df.to_excel(writer, sheet_name='mediation_analysis', index=False)

print(f"\n所有预测结果和统计指标已导出至: {output_file}")

# 计算并输出总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f"\n{'=' * 50}")
print(f"实验完成!")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {str(timedelta(seconds=total_time))}")
print(f"日志已保存至: {log_file}")
print(f"可视化结果已保存至: XGB_Regression_Comparison 文件夹")
print(f"模型已保存至: models 文件夹")

# 恢复标准输出
sys.stdout = sys.__stdout__
