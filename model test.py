import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm用于进度条

seed = 2520157  # 随机种子

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 选择需要的列
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

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

print("开始GridSearchCV拟合...")

# 计算总参数组合数（用于进度条总数）
total_params = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * \
               len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * \
               len(param_grid['max_features'])

# 使用tqdm包装fit过程，并显示进度条
with tqdm(total=total_params, desc="GridSearchCV进度", unit="参数组合") as pbar:
    # 定义回调函数，每次参数组合评估完成后更新进度条
    def progress_callback(ifold, n_folds, i, n):
        if i == n - 1:  # 当完成一个参数组合的所有折交叉验证时
            pbar.update(1)


    # 设置verbose=10以触发回调（GridSearchCV的verbose参数控制日志级别）
    grid_search.fit(X_train, y_train, verbose=10, callback=progress_callback)

print("GridSearchCV拟合完成！")

# 获取最优的随机森林模型
best_rf = grid_search.best_estimator_

# 模型融合尝试：创建不同参数的随机森林模型并简单平均融合
rf_models = []
# 为模型融合过程也添加进度条
with tqdm(total=len([100, 200]) * len([None, 10]), desc="模型融合进度", unit="模型") as pbar:
    for n_est in [100, 200]:
        for depth in [None, 10]:
            rf_model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=seed)
            rf_model.fit(X_train, y_train)
            rf_models.append(rf_model)
            pbar.update(1)

# 计算融合后的预测结果
y_pred_fused_rf = np.mean([model.predict(X_test) for model in rf_models], axis=0)

# 与其他模型集成：使用投票回归器集成随机森林、支持向量机和神经网络
svr = SVR()
mlp = MLPRegressor(random_state=seed)
voting_regressor = VotingRegressor([('rf', best_rf), ('svr', svr), ('mlp', mlp)])
voting_regressor.fit(X_train, y_train)
y_pred_ensemble = voting_regressor.predict(X_test)

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

print('最优随机森林模型评估：')
print(f'均方误差 (MSE): {mse_best_rf}')
print(f'决定系数 (R2): {r2_best_rf}')
print(f'平均绝对误差 (MAE): {mae_best_rf}')
print(f'中位数绝对误差 (MedAE): {medae_best_rf}')

print('\n融合后的随机森林模型评估：')
print(f'均方误差 (MSE): {mse_fused_rf}')
print(f'决定系数 (R2): {r2_fused_rf}')
print(f'平均绝对误差 (MAE): {mae_fused_rf}')
print(f'中位数绝对误差 (MedAE): {medae_fused_rf}')

print('\n集成模型评估：')
print(f'均方误差 (MSE): {mse_ensemble}')
print(f'决定系数 (R2): {r2_ensemble}')
print(f'平均绝对误差 (MAE): {mae_ensemble}')
print(f'中位数绝对误差 (MedAE): {medae_ensemble}')

# 绘制最优随机森林模型预测值与真实值的散点图
plt.scatter(y_test, y_pred_best_rf)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title('Best RF Prediction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()