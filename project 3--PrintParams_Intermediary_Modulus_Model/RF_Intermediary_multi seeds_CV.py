import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import openpyxl
from sklearn.ensemble import RandomForestRegressor
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
    predictors = ['printing_temperature', 'feed_rate', 'printing_speed']
    # 中介变量：宽和高
    mediators = ['Width', 'Height']
    # 因变量：机械模量
    target = 'Experiment_mean(MPa)'

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
    data['target_bin'] = pd.cut(data[target], bins=num_bins, labels=False)

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
    # 1. 中介模型：
    #    第一层：3个打印参数 → 宽和高
    #    第二层：3个打印参数 + 预测的宽和高 → 机械模量（共5个特征）
    # --------------------------

    # 1.1 第一步：用3个打印参数预测宽和高（第一层）
    # 预测宽度
    rf_width = RandomForestRegressor(random_state=seed)
    grid_width = GridSearchCV(rf_width, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_width.fit(train_data[predictors], train_data['Width'])
    best_width = grid_width.best_estimator_

    # 预测高度
    rf_height = RandomForestRegressor(random_state=seed)
    grid_height = GridSearchCV(rf_height, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_height.fit(train_data[predictors], train_data['Height'])
    best_height = grid_height.best_estimator_

    # 生成预测的宽和高，与原始打印参数合并作为第二层特征（共5个特征）
    # 训练集特征
    train_data['predicted_Width'] = best_width.predict(train_data[predictors])
    train_data['predicted_Height'] = best_height.predict(train_data[predictors])
    mediation_train_features = train_data[predictors + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_train_features.shape[1] == 5, \
        f"中介模型第二层训练特征数量错误: 应为5，实际为{mediation_train_features.shape[1]}"

    # 验证集特征
    val_data['predicted_Width'] = best_width.predict(val_data[predictors])
    val_data['predicted_Height'] = best_height.predict(val_data[predictors])
    mediation_val_features = val_data[predictors + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_val_features.shape[1] == 5, \
        f"中介模型第二层验证特征数量错误: 应为5，实际为{mediation_val_features.shape[1]}"

    # 测试集特征
    test_data['predicted_Width'] = best_width.predict(test_data[predictors])
    test_data['predicted_Height'] = best_height.predict(test_data[predictors])
    mediation_test_features = test_data[predictors + ['predicted_Width', 'predicted_Height']]

    # 验证特征数量是否为5
    assert mediation_test_features.shape[1] == 5, \
        f"中介模型第二层测试特征数量错误: 应为5，实际为{mediation_test_features.shape[1]}"

    # 1.2 第二步：用5个特征预测机械模量（第二层）
    rf_mediation = RandomForestRegressor(random_state=seed)
    grid_mediation = GridSearchCV(rf_mediation, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_mediation.fit(mediation_train_features, train_data[target])
    best_mediation = grid_mediation.best_estimator_

    # 在各数据集上预测
    y_pred_val_mediation = best_mediation.predict(mediation_val_features)
    y_pred_test_mediation = best_mediation.predict(mediation_test_features)

    # --------------------------
    # 2. 直接模型：3个打印参数直接预测机械模量
    # --------------------------
    rf_direct = RandomForestRegressor(random_state=seed)
    grid_direct = GridSearchCV(rf_direct, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_direct.fit(train_data[predictors], train_data[target])
    best_direct = grid_direct.best_estimator_

    # 在各数据集上预测
    y_pred_val_direct = best_direct.predict(val_data[predictors])
    y_pred_test_direct = best_direct.predict(test_data[predictors])

    # --------------------------
    # 3. 混合模型：3个打印参数 + 2个实际宽高 → 机械模量（共5个特征）
    # --------------------------
    hybrid_features = predictors + mediators

    rf_hybrid = RandomForestRegressor(random_state=seed)
    grid_hybrid = GridSearchCV(rf_hybrid, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_hybrid.fit(train_data[hybrid_features], train_data[target])
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
        'direct': {  # 直接模型：3个打印参数直接→机械模量
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
        'hybrid': {  # 混合模型：5个特征（3+2）→机械模量
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

    # 保存预测结果
    predictions = {
        'mediation': {
            'val': {'y_true': val_data[target], 'y_pred': y_pred_val_mediation},
            'test': {'y_true': test_data[target], 'y_pred': y_pred_test_mediation}
        },
        'direct': {
            'val': {'y_true': val_data[target], 'y_pred': y_pred_val_direct},
            'test': {'y_true': test_data[target], 'y_pred': y_pred_test_direct}
        },
        'hybrid': {
            'val': {'y_true': val_data[target], 'y_pred': y_pred_val_hybrid},
            'test': {'y_true': test_data[target], 'y_pred': y_pred_test_hybrid}
        }
    }

    # 保存模型
    models = {
        'width': best_width,
        'height': best_height,
        'mediation': best_mediation,
        'direct': best_direct,
        'hybrid': best_hybrid,
        'features': {
            'predictors': predictors,
            'mediators': mediators,
            'hybrid': hybrid_features,
            'mediation_second_layer': mediation_train_features.columns.tolist()
        }
    }

    return results, predictions, models, test_data


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "RF_Intermediary_effect_analysis_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始中介效应分析实验，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据 - 不使用aspect_ratio，保留原始的宽和高
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Width', 'Height', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据准备完成:")
print(f"- 包含的特征: {list(data.columns)}")
print(f"- 打印参数(自变量): ['printing_temperature', 'feed_rate', 'printing_speed']")
print(f"- 中介变量: ['Width', 'Height']")
print(f"- 目标变量: 'Experiment_mean(MPa)'")

# 定义参数网格，用于GridSearchCV进行参数调整
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 定义要测试的种子值（原种子±4之内，9次）
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]
print(f"\n将使用以下种子进行实验: {seeds}")

# 存储所有实验的结果
all_results = []
all_predictions = []
final_models = None
test_data = None

# 运行多次实验
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"开始第 {i + 1}/{len(seeds)} 次实验，种子值: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, test = run_mediation_experiment(seed, data, param_grid)
    all_results.append(results)
    all_predictions.append(predictions)

    # 保存最后一次实验的模型和测试数据用于可视化
    if i == len(seeds) - 1:
        final_models = models
        test_data = test

    # 输出本次实验的评估结果
    print(f"\n第 {i + 1} 次实验评估结果（保留4位小数）:")
    print(f"中介模型 - 测试集R²: {results['mediation']['test']['R2']:.4f}")
    print(f"直接模型 - 测试集R²: {results['direct']['test']['R2']:.4f}")
    print(f"混合模型 - 测试集R²: {results['hybrid']['test']['R2']:.4f}")


# 计算所有实验的平均值
def calculate_averages(results_list):
    """计算多次实验结果的平均值"""
    avg_results = {
        'mediation': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        },
        'direct': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        },
        'hybrid': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        }
    }

    # 收集所有结果
    for res in results_list:
        for model_type in ['mediation', 'direct', 'hybrid']:
            for dataset_type in ['val', 'test']:
                for metric in avg_results[model_type][dataset_type]:
                    if metric in res[model_type][dataset_type]:
                        avg_results[model_type][dataset_type][metric].append(
                            res[model_type][dataset_type][metric])

    # 计算平均值
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in avg_results[model_type][dataset_type]:
                avg_results[model_type][dataset_type][metric] = np.mean(
                    avg_results[model_type][dataset_type][metric])

    return avg_results


# 计算平均值
average_results = calculate_averages(all_results)

# 输出平均值结果
print('\n' + '=' * 50)
print("多次实验的平均评估结果（保留4位小数）:")
print('=' * 50)

for model_type, model_name in [
    ('mediation', '中介模型 (打印参数→宽高→机械模量)'),
    ('direct', '直接模型 (打印参数直接→机械模量)'),
    ('hybrid', '混合模型 (打印参数+宽高→机械模量)')
]:
    print(f"\n{model_name}:")
    print("验证集平均评估:")
    print(f"  均方误差 (MSE): {average_results[model_type]['val']['MSE']:.4f}")
    print(f"  决定系数 (R2): {average_results[model_type]['val']['R2']:.4f}")
    print("测试集平均评估:")
    print(f"  均方误差 (MSE): {average_results[model_type]['test']['MSE']:.4f}")
    print(f"  决定系数 (R2): {average_results[model_type]['test']['R2']:.4f}")
    print(f"  平均绝对误差 (MAE): {average_results[model_type]['test']['MAE']:.4f}")
    print(f"  中位数绝对误差 (MedAE): {average_results[model_type]['test']['MedAE']:.4f}")


# 中介效应分析：计算中介比例
def calculate_mediation_effect(average_results):
    """计算中介效应比例"""
    # 总效应 (直接模型的效应)
    total_effect = average_results['direct']['test']['R2']

    # 直接效应 (控制中介变量后的直接效应)
    # 这里用混合模型与中介模型的差异近似
    direct_effect = average_results['hybrid']['test']['R2'] - average_results['mediation']['test']['R2']

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
mediation_stats = calculate_mediation_effect(average_results)

print('\n' + '=' * 50)
print("中介效应分析结果:")
print('=' * 50)
print(f"总效应 (直接模型R²): {mediation_stats['total_effect']:.4f}")
print(f"直接效应 (控制宽高后): {mediation_stats['direct_effect']:.4f}")
print(f"中介效应 (通过宽高): {mediation_stats['mediation_effect']:.4f}")
print(f"中介比例 (中介效应/总效应): {mediation_stats['mediation_ratio']:.2%}")

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

# 3. 第二层中介模型的5个特征对机械模量的影响
plt.subplot(1, 3, 3)
feature_importance_mediation = final_models['mediation'].feature_importances_
feature_names_mediation = final_models['features']['mediation_second_layer']
plt.bar(feature_names_mediation, feature_importance_mediation)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Second Layer of Mediation Model')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join("outputs", "RF_Intermediary_feature_importance.png"), dpi=300)
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
plt.savefig(os.path.join("outputs", "RF_Intermediary_model_comparison_scatter.png"), dpi=300)
plt.show()

# 绘制三种模型的评估指标对比
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
model_types = ['mediation', 'direct', 'hybrid']
model_names = ['Mediated Model', 'Direct Model', 'Mixed Model']

plt.figure(figsize=(16, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)

    values = [average_results[mt]['test'][metric] for mt in model_types]
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
plt.savefig(os.path.join("outputs", "RF_Intermediary_model_metrics_comparison.png"), dpi=300)
plt.show()

# 导出所有预测结果和平均值到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "RF_Intermediary_analysis_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 导出每次实验的预测结果
    for exp_idx, predictions in enumerate(all_predictions):
        for model_type, model_name in zip(model_types, model_names):
            # 验证集结果
            df_val = pd.DataFrame({
                'True Values': predictions[model_type]['val']['y_true'],
                'Predicted Values': predictions[model_type]['val']['y_pred'].round(4),
                'Error': (predictions[model_type]['val']['y_true'] - predictions[model_type]['val']['y_pred']).round(4)
            })
            df_val.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_{model_type}_val', index=False)

            # 测试集结果
            df_test = pd.DataFrame({
                'True Values': predictions[model_type]['test']['y_true'],
                'Predicted Values': predictions[model_type]['test']['y_pred'].round(4),
                'Error': (predictions[model_type]['test']['y_true'] - predictions[model_type]['test']['y_pred']).round(
                    4)
            })
            df_test.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_{model_type}_test', index=False)

    # 导出每次实验的评估指标
    metrics_data = []
    for exp_idx, results in enumerate(all_results):
        metrics_data.append({
            '实验编号': exp_idx + 1,
            '种子值': seeds[exp_idx],
            # 中介模型
            'mediation_val_MSE': round(results['mediation']['val']['MSE'], 4),
            'mediation_val_R2': round(results['mediation']['val']['R2'], 4),
            'mediation_test_MSE': round(results['mediation']['test']['MSE'], 4),
            'mediation_test_R2': round(results['mediation']['test']['R2'], 4),
            # 直接模型
            'direct_val_MSE': round(results['direct']['val']['MSE'], 4),
            'direct_val_R2': round(results['direct']['val']['R2'], 4),
            'direct_test_MSE': round(results['direct']['test']['MSE'], 4),
            'direct_test_R2': round(results['direct']['test']['R2'], 4),
            # 混合模型
            'hybrid_val_MSE': round(results['hybrid']['val']['MSE'], 4),
            'hybrid_val_R2': round(results['hybrid']['val']['R2'], 4),
            'hybrid_test_MSE': round(results['hybrid']['test']['MSE'], 4),
            'hybrid_test_R2': round(results['hybrid']['test']['R2'], 4)
        })

    df_all_metrics = pd.DataFrame(metrics_data)
    df_all_metrics.to_excel(writer, sheet_name='all_experiments_metrics', index=False)

    # 导出平均评估指标
    avg_metrics_data = {
        'Metric': ['MSE', 'R2', 'MAE', 'MedAE'],
        '中介模型 (验证集)': [
            round(average_results['mediation']['val']['MSE'], 4),
            round(average_results['mediation']['val']['R2'], 4),
            None,
            None
        ],
        '中介模型 (测试集)': [
            round(average_results['mediation']['test']['MSE'], 4),
            round(average_results['mediation']['test']['R2'], 4),
            round(average_results['mediation']['test']['MAE'], 4),
            round(average_results['mediation']['test']['MedAE'], 4)
        ],
        '直接模型 (验证集)': [
            round(average_results['direct']['val']['MSE'], 4),
            round(average_results['direct']['val']['R2'], 4),
            None,
            None
        ],
        '直接模型 (测试集)': [
            round(average_results['direct']['test']['MSE'], 4),
            round(average_results['direct']['test']['R2'], 4),
            round(average_results['direct']['test']['MAE'], 4),
            round(average_results['direct']['test']['MedAE'], 4)
        ],
        '混合模型 (测试集)': [
            round(average_results['hybrid']['test']['MSE'], 4),
            round(average_results['hybrid']['test']['R2'], 4),
            round(average_results['hybrid']['test']['MAE'], 4),
            round(average_results['hybrid']['test']['MedAE'], 4)
        ]
    }
    df_avg_metrics = pd.DataFrame(avg_metrics_data)
    df_avg_metrics.to_excel(writer, sheet_name='average_metrics', index=False)

    # 导出中介效应分析结果
    mediation_data = pd.DataFrame([{
        '总效应 (直接模型R²)': round(mediation_stats['total_effect'], 4),
        '直接效应': round(mediation_stats['direct_effect'], 4),
        '中介效应': round(mediation_stats['mediation_effect'], 4),
        '中介比例': f"{round(mediation_stats['mediation_ratio'] * 100, 2)}%"
    }])
    mediation_data.to_excel(writer, sheet_name='mediation_analysis', index=False)

print(f"\n所有预测结果已导出至: {output_file}")

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"所有实验完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__
