
# 使用XGBoost减少过拟合
# XGBoost实现 - 针对小样本低维度数据的精确预测
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os

seed = 2520157  # 随机种子

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

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "model_training_log_xgboost.txt"
log_file = get_unique_filename(base_log_file)

# 重定向输出流
sys.stdout = Logger('outputs/'+log_file)

print(f"开始模型训练，日志将保存到 {'outputs/'+log_file}")
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
selected_columns_dim = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data_with_dim = data[selected_columns_dim]

print("数据准备完成:")
print(f"- 特征集1 (aspect_ratio): {list(data_with_ratio.columns)}")
print(f"- 特征集2 (Height+Width): {list(data_with_dim.columns)}")

# 为两种特征集分别划分训练集和测试集
# 特征集1: aspect_ratio
X_ratio = data_with_ratio.drop('Experiment_mean(MPa)', axis=1)
y_ratio = data_with_ratio['Experiment_mean(MPa)']
X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(
    X_ratio, y_ratio, test_size=0.2, random_state=seed
)

# 特征集2: Height+Width
X_dim = data_with_dim.drop('Experiment_mean(MPa)', axis=1)
y_dim = data_with_dim['Experiment_mean(MPa)']
X_train_dim, X_test_dim, y_train_dim, y_test_dim = train_test_split(
    X_dim, y_dim, test_size=0.2, random_state=seed
)

# 定义XGBoost参数网格
# 缩减参数网格（示例）
param_grid = {
    'n_estimators': [50, 100],  # 减少取值数量
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_child_weight': [1, 2],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0.5, 1.0],
    'reg_lambda': [0.5, 1.0]
}

# 创建XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)

# 为两种特征集分别进行参数搜索和模型训练
results = {}
models = {}
model_names = {}

# 特征集1: aspect_ratio
print("\n开始aspect_ratio特征集的XGBoost参数搜索...")
grid_search_ratio = GridSearchCV(xgb_model, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
grid_search_ratio.fit(X_train_ratio, y_train_ratio)
best_xgb_ratio = grid_search_ratio.best_estimator_
y_pred_ratio = best_xgb_ratio.predict(X_test_ratio)

# 评估模型
mse_ratio = mean_squared_error(y_test_ratio, y_pred_ratio)
r2_ratio = r2_score(y_test_ratio, y_pred_ratio)
mae_ratio = mean_absolute_error(y_test_ratio, y_pred_ratio)
medae_ratio = median_absolute_error(y_test_ratio, y_pred_ratio)

results['xgb_ratio'] = {
    'MSE': mse_ratio, 'R2': r2_ratio, 'MAE': mae_ratio, 'MedAE': medae_ratio
}
models['xgb_ratio'] = best_xgb_ratio
model_names['xgb_ratio'] = 'XGBoost (with aspect_ratio)'

print("aspect_ratio特征集的XGBoost参数搜索完成！")

# 特征集2: Height+Width
print("\n开始Height+Width特征集的XGBoost参数搜索...")
grid_search_dim = GridSearchCV(xgb_model, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
grid_search_dim.fit(X_train_dim, y_train_dim)
best_xgb_dim = grid_search_dim.best_estimator_
y_pred_dim = best_xgb_dim.predict(X_test_dim)

# 评估模型
mse_dim = mean_squared_error(y_test_dim, y_pred_dim)
r2_dim = r2_score(y_test_dim, y_pred_dim)
mae_dim = mean_absolute_error(y_test_dim, y_pred_dim)
medae_dim = median_absolute_error(y_test_dim, y_pred_dim)

results['xgb_dim'] = {
    'MSE': mse_dim, 'R2': r2_dim, 'MAE': mae_dim, 'MedAE': medae_dim
}
models['xgb_dim'] = best_xgb_dim
model_names['xgb_dim'] = 'XGBoost (with Height+Width)'

print("Height+Width特征集的XGBoost参数搜索完成！")

# 输出评估结果
print("\n模型评估结果（保留4位小数）:")
print("=" * 50)

for key, result in results.items():
    print(f"\n{model_names[key]}评估：")
    print(f'均方误差 (MSE): {result["MSE"]:.4f}')
    print(f'决定系数 (R2): {result["R2"]:.4f}')
    print(f'平均绝对误差 (MAE): {result["MAE"]:.4f}')
    print(f'中位数绝对误差 (MedAE): {result["MedAE"]:.4f}')

# 输出最优参数
print("\n最优参数:")
print("-" * 30)
print("aspect_ratio特征集:")
for param, value in grid_search_ratio.best_params_.items():
    print(f"{param}: {value}")

print("\nHeight+Width特征集:")
for param, value in grid_search_dim.best_params_.items():
    print(f"{param}: {value}")

# 特征重要性分析
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
feature_importance_ratio = best_xgb_ratio.feature_importances_
feature_names_ratio = X_ratio.columns
plt.bar(feature_names_ratio, feature_importance_ratio)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importance (aspect_ratio)')
plt.xticks(rotation=45)

# Height+Width特征集
plt.subplot(1, 2, 2)
feature_importance_dim = best_xgb_dim.feature_importances_
feature_names_dim = X_dim.columns
plt.bar(feature_names_dim, feature_importance_dim)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importance (Height+Width)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join("Regression_Comparison", "xgb_feature_importance.png"), dpi=300)
plt.show()

# 绘制预测值与真实值的散点图
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
plt.scatter(y_test_ratio, y_pred_ratio)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'XGBoost (aspect_ratio) - R²={r2_ratio:.4f}')
plt.plot([y_test_ratio.min(), y_test_ratio.max()], [y_test_ratio.min(), y_test_ratio.max()], 'r--')

# Height+Width特征集
plt.subplot(1, 2, 2)
plt.scatter(y_test_dim, y_pred_dim)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'XGBoost (Height+Width) - R²={r2_dim:.4f}')
plt.plot([y_test_dim.min(), y_test_dim.max()], [y_test_dim.min(), y_test_dim.max()], 'r--')

plt.tight_layout()
plt.savefig(os.path.join("Regression_Comparison", "xgb_prediction_scatter.png"), dpi=300)
plt.show()

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {'outputs/'+log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__