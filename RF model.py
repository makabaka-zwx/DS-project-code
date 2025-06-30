
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

seed = 2520157 # 随机种子

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 选择需要的列
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]
print(data)


# 准备特征和目标变量
X = data.drop('Experiment_mean(MPa)', axis=1)
y = data['Experiment_mean(MPa)']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归器
rf = RandomForestRegressor(n_estimators=100, random_state=seed)

# 在训练集上训练模型
rf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差 (MSE): {mse}')
print(f'决定系数 (R2): {r2}')

# 绘制预测值与真实值的散点图
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title('RF Prediction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()