import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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
base_log_file = "model_training_log2.0.txt"
log_file = get_unique_filename(base_log_file)

# 重定向输出流
sys.stdout = Logger('outputs/'+log_file)

print(f"开始模型训练，日志将保存到 {'outputs/'+log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 合并Height和Width为aspect_ratio
data['aspect_ratio'] = data['Height'] / data['Width']

# 选择需要的列（排除原始的Height和Width）
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'aspect_ratio', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据特征合并完成:")
print(f"- 新增特征: aspect_ratio (Height/Width)")

# 准备特征和目标变量
X = data.drop('Experiment_mean(MPa)', axis=1)
y = data['Experiment_mean(MPa)']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 定义参数网格，用于GridSearchCV进行参数调整
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 使用GridSearchCV进行参数搜索
rf = RandomForestRegressor(random_state=seed)
grid_search = GridSearchCV(rf, param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')

# 显示GridSearchCV进度
print("开始GridSearchCV参数搜索...")
grid_search.fit(X_train, y_train)
print("GridSearchCV参数搜索完成！")

# 获取最优的随机森林模型
best_rf = grid_search.best_estimator_

# 模型融合尝试：创建不同参数的随机森林模型并简单平均融合
print("开始模型融合训练...")
rf_models = []
for n_est in [100, 200]:
    for depth in [None, 10]:
        rf_model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=seed)
        rf_model.fit(X_train, y_train)
        rf_models.append(rf_model)
print("模型融合训练完成！")

# 计算融合后的预测结果
y_pred_fused_rf = np.mean([model.predict(X_test) for model in rf_models], axis=0)

# 与其他模型集成：使用投票回归器集成随机森林、支持向量机和神经网络
print("开始集成模型训练...")
svr = SVR()
mlp = MLPRegressor(random_state=seed)
voting_regressor = VotingRegressor([('rf', best_rf), ('svr', svr), ('mlp', mlp)])
voting_regressor.fit(X_train, y_train)
y_pred_ensemble = voting_regressor.predict(X_test)
print("集成模型训练完成！")

# 使用最优的随机森林模型进行预测
y_pred_best_rf = best_rf.predict(X_test)

# 评估最优的随机森林模型
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
medae_best_rf = median_absolute_error(y_test, y_pred_best_rf)

# 评估融合后的随机森林模型
mse_fused_rf = mean_squared_error(y_test, y_pred_fused_rf)
r2_fused_rf = r2_score(y_test, y_pred_fused_rf)
mae_fused_rf = mean_absolute_error(y_test, y_pred_fused_rf)
medae_fused_rf = median_absolute_error(y_test, y_pred_fused_rf)

# 评估集成模型
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
medae_ensemble = median_absolute_error(y_test, y_pred_ensemble)


# 修改输出格式，保留4位小数
print('最优随机森林模型评估：')
print(f'均方误差 (MSE): {mse_best_rf:.4f}')
print(f'决定系数 (R2): {r2_best_rf:.4f}')
print(f'平均绝对误差 (MAE): {mae_best_rf:.4f}')
print(f'中位数绝对误差 (MedAE): {medae_best_rf:.4f}')

print('\n融合后的随机森林模型评估：')
print(f'均方误差 (MSE): {mse_fused_rf:.4f}')
print(f'决定系数 (R2): {r2_fused_rf:.4f}')
print(f'平均绝对误差 (MAE): {mae_fused_rf:.4f}')
print(f'中位数绝对误差 (MedAE): {medae_fused_rf:.4f}')

print('\n集成模型评估：')
print(f'均方误差 (MSE): {mse_ensemble:.4f}')
print(f'决定系数 (R2): {r2_ensemble:.4f}')
print(f'平均绝对误差 (MAE): {mae_ensemble:.4f}')
print(f'中位数绝对误差 (MedAE): {medae_ensemble:.4f}')

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {'outputs/'+log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__

# 绘制最优随机森林模型预测值与真实值的散点图
plt.scatter(y_test, y_pred_best_rf)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title('Best RF Prediction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()