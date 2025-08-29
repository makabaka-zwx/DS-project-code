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

seed = 2520157  # 随机种子
np.random.seed(seed)


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


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "RF_model_training_log3.0.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始模型训练，日志将保存到 {log_file}")
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

# 为两种特征集分别划分训练集、验证集和测试集 (7:1.5:1.5)
# 特征集1: aspect_ratio
X_ratio = data_with_ratio.drop('Experiment_mean(MPa)', axis=1)
y_ratio = data_with_ratio['Experiment_mean(MPa)']

# 先划分为训练集（70%）和临时测试集（30%）
X_train_ratio, X_temp_ratio, y_train_ratio, y_temp_ratio = train_test_split(
    X_ratio, y_ratio, test_size=0.3, random_state=seed
)
# 再将临时测试集划分为验证集（15%）和测试集（15%）
X_val_ratio, X_test_ratio, y_val_ratio, y_test_ratio = train_test_split(
    X_temp_ratio, y_temp_ratio, test_size=0.5, random_state=seed
)

# 特征集2: Height+Width
X_dim = data_with_dim.drop('Experiment_mean(MPa)', axis=1)
y_dim = data_with_dim['Experiment_mean(MPa)']

# 先划分为训练集（70%）和临时测试集（30%）
X_train_dim, X_temp_dim, y_train_dim, y_temp_dim = train_test_split(
    X_dim, y_dim, test_size=0.3, random_state=seed
)
# 再将临时测试集划分为验证集（15%）和测试集（15%）
X_val_dim, X_test_dim, y_val_dim, y_test_dim = train_test_split(
    X_temp_dim, y_temp_dim, test_size=0.5, random_state=seed
)

print("\n数据集划分完成:")
print(f"- aspect_ratio特征集: 训练集={X_train_ratio.shape}, 验证集={X_val_ratio.shape}, 测试集={X_test_ratio.shape}")
print(f"- Height+Width特征集: 训练集={X_train_dim.shape}, 验证集={X_val_dim.shape}, 测试集={X_test_dim.shape}")

# 定义参数网格，用于GridSearchCV进行参数调整
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 存储结果的字典
results = {}
models = {}
predictions = {}

# 训练特征集1的RF模型 (aspect_ratio)
print("\n开始aspect_ratio特征集的GridSearchCV参数搜索...")
rf_ratio = RandomForestRegressor(random_state=seed)
grid_search_ratio = GridSearchCV(rf_ratio, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
grid_search_ratio.fit(X_train_ratio, y_train_ratio)
best_rf_ratio = grid_search_ratio.best_estimator_

# 在验证集上评估
y_pred_val_ratio = best_rf_ratio.predict(X_val_ratio)
mse_val_ratio = mean_squared_error(y_val_ratio, y_pred_val_ratio)
r2_val_ratio = r2_score(y_val_ratio, y_pred_val_ratio)

# 在测试集上评估
y_pred_test_ratio = best_rf_ratio.predict(X_test_ratio)
mse_test_ratio = mean_squared_error(y_test_ratio, y_pred_test_ratio)
r2_test_ratio = r2_score(y_test_ratio, y_pred_test_ratio)
mae_test_ratio = mean_absolute_error(y_test_ratio, y_pred_test_ratio)
medae_test_ratio = median_absolute_error(y_test_ratio, y_pred_test_ratio)

# 保存结果
results['ratio'] = {
    'val': {'MSE': mse_val_ratio, 'R2': r2_val_ratio},
    'test': {'MSE': mse_test_ratio, 'R2': r2_test_ratio,
             'MAE': mae_test_ratio, 'MedAE': medae_test_ratio}
}
models['ratio'] = best_rf_ratio
predictions['ratio'] = {
    'val': {'y_true': y_val_ratio, 'y_pred': y_pred_val_ratio},
    'test': {'y_true': y_test_ratio, 'y_pred': y_pred_test_ratio}
}

print("aspect_ratio特征集的RF模型训练完成！")
print(f"  最优参数: {grid_search_ratio.best_params_}")
print(f"  验证集R²: {r2_val_ratio:.4f}")
print(f"  测试集R²: {r2_test_ratio:.4f}")

# 训练特征集2的RF模型 (Height+Width)
print("\n开始Height+Width特征集的GridSearchCV参数搜索...")
rf_dim = RandomForestRegressor(random_state=seed)
grid_search_dim = GridSearchCV(rf_dim, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
grid_search_dim.fit(X_train_dim, y_train_dim)
best_rf_dim = grid_search_dim.best_estimator_

# 在验证集上评估
y_pred_val_dim = best_rf_dim.predict(X_val_dim)
mse_val_dim = mean_squared_error(y_val_dim, y_pred_val_dim)
r2_val_dim = r2_score(y_val_dim, y_pred_val_dim)

# 在测试集上评估
y_pred_test_dim = best_rf_dim.predict(X_test_dim)
mse_test_dim = mean_squared_error(y_test_dim, y_pred_test_dim)
r2_test_dim = r2_score(y_test_dim, y_pred_test_dim)
mae_test_dim = mean_absolute_error(y_test_dim, y_pred_test_dim)
medae_test_dim = median_absolute_error(y_test_dim, y_pred_test_dim)

# 保存结果
results['dim'] = {
    'val': {'MSE': mse_val_dim, 'R2': r2_val_dim},
    'test': {'MSE': mse_test_dim, 'R2': r2_test_dim,
             'MAE': mae_test_dim, 'MedAE': medae_test_dim}
}
models['dim'] = best_rf_dim
predictions['dim'] = {
    'val': {'y_true': y_val_dim, 'y_pred': y_pred_val_dim},
    'test': {'y_true': y_test_dim, 'y_pred': y_pred_test_dim}
}

print("Height+Width特征集的RF模型训练完成！")
print(f"  最优参数: {grid_search_dim.best_params_}")
print(f"  验证集R²: {r2_val_dim:.4f}")
print(f"  测试集R²: {r2_test_dim:.4f}")

# 输出评估结果
print('\n模型评估结果（保留4位小数）:')
print('=' * 50)

print("\n使用aspect_ratio特征的RF模型:")
print("验证集评估:")
print(f"  均方误差 (MSE): {results['ratio']['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {results['ratio']['val']['R2']:.4f}")
print("测试集评估:")
print(f"  均方误差 (MSE): {results['ratio']['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {results['ratio']['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results['ratio']['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results['ratio']['test']['MedAE']:.4f}")

print("\n使用Height+Width特征的RF模型:")
print("验证集评估:")
print(f"  均方误差 (MSE): {results['dim']['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {results['dim']['val']['R2']:.4f}")
print("测试集评估:")
print(f"  均方误差 (MSE): {results['dim']['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {results['dim']['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results['dim']['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results['dim']['test']['MedAE']:.4f}")

# 特征重要性分析
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

# 绘制预测值与真实值的散点图
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
plt.scatter(predictions['ratio']['test']['y_true'], predictions['ratio']['test']['y_pred'])
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'RF (aspect_ratio) - R²={results["ratio"]["test"]["R2"]:.4f}')
plt.plot([predictions['ratio']['test']['y_true'].min(), predictions['ratio']['test']['y_true'].max()],
         [predictions['ratio']['test']['y_true'].min(), predictions['ratio']['test']['y_true'].max()], 'r--')

# Height+Width特征集
plt.subplot(1, 2, 2)
plt.scatter(predictions['dim']['test']['y_true'], predictions['dim']['test']['y_pred'])
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'RF (Height+Width) - R²={results["dim"]["test"]["R2"]:.4f}')
plt.plot([predictions['dim']['test']['y_true'].min(), predictions['dim']['test']['y_true'].max()],
         [predictions['dim']['test']['y_true'].min(), predictions['dim']['test']['y_true'].max()], 'r--')

plt.tight_layout()
plt.savefig(os.path.join("outputs", "rf_prediction_scatter.png"), dpi=300)
plt.show()

# 导出预测结果到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "rf_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # aspect_ratio特征集的验证集结果
    df_ratio_val = pd.DataFrame({
        'True Values': predictions['ratio']['val']['y_true'],
        'Predicted Values': predictions['ratio']['val']['y_pred'],
        'Error': predictions['ratio']['val']['y_true'] - predictions['ratio']['val']['y_pred']
    })
    df_ratio_val.to_excel(writer, sheet_name='ratio_validation', index=False)

    # aspect_ratio特征集的测试集结果
    df_ratio_test = pd.DataFrame({
        'True Values': predictions['ratio']['test']['y_true'],
        'Predicted Values': predictions['ratio']['test']['y_pred'],
        'Error': predictions['ratio']['test']['y_true'] - predictions['ratio']['test']['y_pred']
    })
    df_ratio_test.to_excel(writer, sheet_name='ratio_test', index=False)

    # Height+Width特征集的验证集结果
    df_dim_val = pd.DataFrame({
        'True Values': predictions['dim']['val']['y_true'],
        'Predicted Values': predictions['dim']['val']['y_pred'],
        'Error': predictions['dim']['val']['y_true'] - predictions['dim']['val']['y_pred']
    })
    df_dim_val.to_excel(writer, sheet_name='dim_validation', index=False)

    # Height+Width特征集的测试集结果
    df_dim_test = pd.DataFrame({
        'True Values': predictions['dim']['test']['y_true'],
        'Predicted Values': predictions['dim']['test']['y_pred'],
        'Error': predictions['dim']['test']['y_true'] - predictions['dim']['test']['y_pred']
    })
    df_dim_test.to_excel(writer, sheet_name='dim_test', index=False)

    # 模型评估指标汇总
    metrics_data = {
        'Metric': ['MSE', 'R2', 'MAE', 'MedAE'],
        'aspect_ratio (validation)': [
            results['ratio']['val']['MSE'],
            results['ratio']['val']['R2'],
            None,  # 验证集不计算MAE和MedAE
            None
        ],
        'aspect_ratio (test)': [
            results['ratio']['test']['MSE'],
            results['ratio']['test']['R2'],
            results['ratio']['test']['MAE'],
            results['ratio']['test']['MedAE']
        ],
        'Height+Width (validation)': [
            results['dim']['val']['MSE'],
            results['dim']['val']['R2'],
            None,  # 验证集不计算MAE和MedAE
            None
        ],
        'Height+Width (test)': [
            results['dim']['test']['MSE'],
            results['dim']['test']['R2'],
            results['dim']['test']['MAE'],
            results['dim']['test']['MedAE']
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_excel(writer, sheet_name='metrics_summary', index=False)

print(f"\n预测结果已导出至: {output_file}")

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__
