import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import openpyxl  # 用于Excel文件操作


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


def run_experiment(seed, data_with_ratio, data_with_dim, param_grid):
    """运行单次实验并返回结果"""
    np.random.seed(seed)

    # 特征集1: aspect_ratio
    X_ratio = data_with_ratio.drop('Experiment_mean(MPa)', axis=1)
    y_ratio = data_with_ratio['Experiment_mean(MPa)']

    # 划分训练集、验证集和测试集 (7:1.5:1.5)
    X_train_ratio, X_temp_ratio, y_train_ratio, y_temp_ratio = train_test_split(
        X_ratio, y_ratio, test_size=0.3, random_state=seed
    )
    X_val_ratio, X_test_ratio, y_val_ratio, y_test_ratio = train_test_split(
        X_temp_ratio, y_temp_ratio, test_size=0.5, random_state=seed
    )

    # 特征集2: Height+Width
    X_dim = data_with_dim.drop('Experiment_mean(MPa)', axis=1)
    y_dim = data_with_dim['Experiment_mean(MPa)']

    # 划分训练集、验证集和测试集 (7:1.5:1.5)
    X_train_dim, X_temp_dim, y_train_dim, y_temp_dim = train_test_split(
        X_dim, y_dim, test_size=0.3, random_state=seed
    )
    X_val_dim, X_test_dim, y_val_dim, y_test_dim = train_test_split(
        X_temp_dim, y_temp_dim, test_size=0.5, random_state=seed
    )

    # 训练特征集1的RF模型 (aspect_ratio)
    rf_ratio = RandomForestRegressor(random_state=seed)
    grid_search_ratio = GridSearchCV(rf_ratio, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_search_ratio.fit(X_train_ratio, y_train_ratio)
    best_rf_ratio = grid_search_ratio.best_estimator_

    # 在验证集和测试集上评估
    y_pred_val_ratio = best_rf_ratio.predict(X_val_ratio)
    y_pred_test_ratio = best_rf_ratio.predict(X_test_ratio)

    # 训练特征集2的RF模型 (Height+Width)
    rf_dim = RandomForestRegressor(random_state=seed)
    grid_search_dim = GridSearchCV(rf_dim, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
    grid_search_dim.fit(X_train_dim, y_train_dim)
    best_rf_dim = grid_search_dim.best_estimator_

    # 在验证集和测试集上评估
    y_pred_val_dim = best_rf_dim.predict(X_val_dim)
    y_pred_test_dim = best_rf_dim.predict(X_test_dim)

    # 计算评估指标
    results = {
        'ratio': {
            'val': {
                'MSE': mean_squared_error(y_val_ratio, y_pred_val_ratio),
                'R2': r2_score(y_val_ratio, y_pred_val_ratio)
            },
            'test': {
                'MSE': mean_squared_error(y_test_ratio, y_pred_test_ratio),
                'R2': r2_score(y_test_ratio, y_pred_test_ratio),
                'MAE': mean_absolute_error(y_test_ratio, y_pred_test_ratio),
                'MedAE': median_absolute_error(y_test_ratio, y_pred_test_ratio)
            }
        },
        'dim': {
            'val': {
                'MSE': mean_squared_error(y_val_dim, y_pred_val_dim),
                'R2': r2_score(y_val_dim, y_pred_val_dim)
            },
            'test': {
                'MSE': mean_squared_error(y_test_dim, y_pred_test_dim),
                'R2': r2_score(y_test_dim, y_pred_test_dim),
                'MAE': mean_absolute_error(y_test_dim, y_pred_test_dim),
                'MedAE': median_absolute_error(y_test_dim, y_pred_test_dim)
            }
        }
    }

    # 保存预测结果
    predictions = {
        'ratio': {
            'val': {'y_true': y_val_ratio, 'y_pred': y_pred_val_ratio},
            'test': {'y_true': y_test_ratio, 'y_pred': y_pred_test_ratio}
        },
        'dim': {
            'val': {'y_true': y_val_dim, 'y_pred': y_pred_val_dim},
            'test': {'y_true': y_test_dim, 'y_pred': y_pred_test_dim}
        }
    }

    # 保存模型
    models = {
        'ratio': best_rf_ratio,
        'dim': best_rf_dim
    }

    return results, predictions, models, X_ratio, X_dim


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "RF_model_training_log_multi_seed.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始多种子模型训练，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 准备两种特征集
# 特征集1: 使用aspect_ratio
data_with_ratio = data.copy()
data_with_ratio['aspect_ratio'] = data_with_ratio['Height'] / data_with_ratio['Width']
selected_columns_ratio = ['printing_temperature', 'feed_rate', 'printing_speed', 'aspect_ratio', 'Experiment_mean(MPa)']
data_with_ratio = data_with_ratio[selected_columns_ratio]

# 特征集2: 使用Height和Width
selected_columns_dim = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
data_with_dim = data[selected_columns_dim]

print("数据准备完成:")
print(f"- 特征集1 (aspect_ratio): {list(data_with_ratio.columns)}")
print(f"- 特征集2 (Height+Width): {list(data_with_dim.columns)}")

# 定义参数网格，用于GridSearchCV进行参数调整
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 定义要测试的种子值（原种子±4之内，共9次）
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]
print(f"\n将使用以下种子进行实验: {seeds}")

# 存储所有实验的结果
all_results = []
all_predictions = []

# 运行多次实验
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"开始第 {i + 1}/{len(seeds)} 次实验，种子值: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, X_ratio, X_dim = run_experiment(seed, data_with_ratio, data_with_dim, param_grid)
    all_results.append(results)
    all_predictions.append(predictions)

    # 输出本次实验的评估结果
    print(f"\n第 {i + 1} 次实验评估结果（保留4位小数）:")
    print(f"使用aspect_ratio特征的RF模型 - 测试集R²: {results['ratio']['test']['R2']:.4f}")
    print(f"使用Height+Width特征的RF模型 - 测试集R²: {results['dim']['test']['R2']:.4f}")


# 计算所有实验的平均值
def calculate_averages(results_list):
    """计算多次实验结果的平均值"""
    avg_results = {
        'ratio': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        },
        'dim': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        }
    }

    # 收集所有结果
    for res in results_list:
        for feature_set in ['ratio', 'dim']:
            for dataset_type in ['val', 'test']:
                for metric in avg_results[feature_set][dataset_type]:
                    if metric in res[feature_set][dataset_type]:
                        avg_results[feature_set][dataset_type][metric].append(res[feature_set][dataset_type][metric])

    # 计算平均值
    for feature_set in ['ratio', 'dim']:
        for dataset_type in ['val', 'test']:
            for metric in avg_results[feature_set][dataset_type]:
                avg_results[feature_set][dataset_type][metric] = np.mean(avg_results[feature_set][dataset_type][metric])

    return avg_results


# 计算平均值
average_results = calculate_averages(all_results)

# 输出平均值结果
print('\n' + '=' * 50)
print("多次实验的平均评估结果（保留4位小数）:")
print('=' * 50)

print("\n使用aspect_ratio特征的RF模型:")
print("验证集平均评估:")
print(f"  均方误差 (MSE): {average_results['ratio']['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {average_results['ratio']['val']['R2']:.4f}")
print("测试集平均评估:")
print(f"  均方误差 (MSE): {average_results['ratio']['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {average_results['ratio']['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {average_results['ratio']['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {average_results['ratio']['test']['MedAE']:.4f}")

print("\n使用Height+Width特征的RF模型:")
print("验证集平均评估:")
print(f"  均方误差 (MSE): {average_results['dim']['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {average_results['dim']['val']['R2']:.4f}")
print("测试集平均评估:")
print(f"  均方误差 (MSE): {average_results['dim']['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {average_results['dim']['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {average_results['dim']['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {average_results['dim']['test']['MedAE']:.4f}")

# 绘制最后一次实验的特征重要性分析（作为代表）
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
feature_importance_ratio = models['ratio'].feature_importances_
feature_names_ratio = X_ratio.columns
plt.bar(feature_names_ratio, feature_importance_ratio)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('RF Feature Importance (aspect_ratio)')
plt.xticks(rotation=45)

# Height+Width特征集
plt.subplot(1, 2, 2)
feature_importance_dim = models['dim'].feature_importances_
feature_names_dim = X_dim.columns
plt.bar(feature_names_dim, feature_importance_dim)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('RF Feature Importance (Height+Width)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join("outputs", "rf_feature_importance.png"), dpi=300)
plt.show()

# 导出所有预测结果和平均值到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "rf_multi_seed_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 导出每次实验的预测结果
    for exp_idx, predictions in enumerate(all_predictions):
        # aspect_ratio特征集的验证集结果
        df_ratio_val = pd.DataFrame({
            'True Values': predictions['ratio']['val']['y_true'],
            'Predicted Values': predictions['ratio']['val']['y_pred'].round(4),
            'Error': (predictions['ratio']['val']['y_true'] - predictions['ratio']['val']['y_pred']).round(4)
        })
        df_ratio_val.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_ratio_val', index=False)

        # aspect_ratio特征集的测试集结果
        df_ratio_test = pd.DataFrame({
            'True Values': predictions['ratio']['test']['y_true'],
            'Predicted Values': predictions['ratio']['test']['y_pred'].round(4),
            'Error': (predictions['ratio']['test']['y_true'] - predictions['ratio']['test']['y_pred']).round(4)
        })
        df_ratio_test.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_ratio_test', index=False)

        # Height+Width特征集的验证集结果
        df_dim_val = pd.DataFrame({
            'True Values': predictions['dim']['val']['y_true'],
            'Predicted Values': predictions['dim']['val']['y_pred'].round(4),
            'Error': (predictions['dim']['val']['y_true'] - predictions['dim']['val']['y_pred']).round(4)
        })
        df_dim_val.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_dim_val', index=False)

        # Height+Width特征集的测试集结果
        df_dim_test = pd.DataFrame({
            'True Values': predictions['dim']['test']['y_true'],
            'Predicted Values': predictions['dim']['test']['y_pred'].round(4),
            'Error': (predictions['dim']['test']['y_true'] - predictions['dim']['test']['y_pred']).round(4)
        })
        df_dim_test.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_dim_test', index=False)

    # 导出每次实验的评估指标
    metrics_data = []
    for exp_idx, results in enumerate(all_results):
        metrics_data.append({
            '实验编号': exp_idx + 1,
            '种子值': seeds[exp_idx],
            'aspect_ratio_val_MSE': round(results['ratio']['val']['MSE'], 4),
            'aspect_ratio_val_R2': round(results['ratio']['val']['R2'], 4),
            'aspect_ratio_test_MSE': round(results['ratio']['test']['MSE'], 4),
            'aspect_ratio_test_R2': round(results['ratio']['test']['R2'], 4),
            'aspect_ratio_test_MAE': round(results['ratio']['test']['MAE'], 4),
            'aspect_ratio_test_MedAE': round(results['ratio']['test']['MedAE'], 4),
            'dim_val_MSE': round(results['dim']['val']['MSE'], 4),
            'dim_val_R2': round(results['dim']['val']['R2'], 4),
            'dim_test_MSE': round(results['dim']['test']['MSE'], 4),
            'dim_test_R2': round(results['dim']['test']['R2'], 4),
            'dim_test_MAE': round(results['dim']['test']['MAE'], 4),
            'dim_test_MedAE': round(results['dim']['test']['MedAE'], 4)
        })

    df_all_metrics = pd.DataFrame(metrics_data)
    df_all_metrics.to_excel(writer, sheet_name='all_experiments_metrics', index=False)

    # 导出平均评估指标
    avg_metrics_data = {
        'Metric': ['MSE', 'R2', 'MAE', 'MedAE'],
        'aspect_ratio (validation)': [
            round(average_results['ratio']['val']['MSE'], 4),
            round(average_results['ratio']['val']['R2'], 4),
            None,
            None
        ],
        'aspect_ratio (test)': [
            round(average_results['ratio']['test']['MSE'], 4),
            round(average_results['ratio']['test']['R2'], 4),
            round(average_results['ratio']['test']['MAE'], 4),
            round(average_results['ratio']['test']['MedAE'], 4)
        ],
        'Height+Width (validation)': [
            round(average_results['dim']['val']['MSE'], 4),
            round(average_results['dim']['val']['R2'], 4),
            None,
            None
        ],
        'Height+Width (test)': [
            round(average_results['dim']['test']['MSE'], 4),
            round(average_results['dim']['test']['R2'], 4),
            round(average_results['dim']['test']['MAE'], 4),
            round(average_results['dim']['test']['MedAE'], 4)
        ]
    }
    df_avg_metrics = pd.DataFrame(avg_metrics_data)
    df_avg_metrics.to_excel(writer, sheet_name='average_metrics', index=False)

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
