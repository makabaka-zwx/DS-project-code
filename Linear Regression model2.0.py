# 这是线性回归模型替代RF模型的效果
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os

seed = 2520157  # 随机种子
np.random.seed(seed)


class Logger:
    """同时将输出流保存到控制台和文件"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_unique_filename(base_name):
    """生成唯一的日志文件名，如果已存在则添加序号后缀"""
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
os.makedirs("Linear Regression Plots", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "model_training_log_linear_regression2.0.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始模型训练，日志将保存到 {log_file}")
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

# 数据标准化（Standardization）
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 数据归一化（Normalization）
normalizer = MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

print("数据预处理完成:")
print(f"- 原始训练数据形状: {X_train.shape}")
print(f"- 标准化训练数据形状: {X_train_std.shape}")
print(f"- 归一化训练数据形状: {X_train_norm.shape}")

# 1. 普通线性回归模型（使用原始数据）
print("开始训练线性回归模型...")
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
print("线性回归模型训练完成！")

# 2. 标准化数据的线性回归模型
print("开始训练标准化数据的线性回归模型...")
linear_reg_std = LinearRegression()
linear_reg_std.fit(X_train_std, y_train)
y_pred_linear_std = linear_reg_std.predict(X_test_std)
print("标准化数据的线性回归模型训练完成！")

# 3. 归一化数据的线性回归模型
print("开始训练归一化数据的线性回归模型...")
linear_reg_norm = LinearRegression()
linear_reg_norm.fit(X_train_norm, y_train)
y_pred_linear_norm = linear_reg_norm.predict(X_test_norm)
print("归一化数据的线性回归模型训练完成！")

# 4. SVR模型（使用标准化数据）
print("开始训练SVR模型...")
svr = SVR()
svr.fit(X_train_std, y_train)
y_pred_svr = svr.predict(X_test_std)
print("SVR模型训练完成！")

# 5. MLP模型（使用归一化数据）
print("开始训练MLP模型...")
mlp = MLPRegressor(random_state=seed)
mlp.fit(X_train_norm, y_train)
y_pred_mlp = mlp.predict(X_test_norm)
print("MLP模型训练完成！")

# 评估各模型
models = {
    "Linear Regression": (y_test, y_pred_linear),
    "Linear Regression (Standardized)": (y_test, y_pred_linear_std),
    "Linear Regression (Normalization)": (y_test, y_pred_linear_norm),
    "SVR": (y_test, y_pred_svr),
    "MLP": (y_test, y_pred_mlp)
}

print('\n各模型评估结果（保留4位小数）:')
print('=' * 50)

for name, (y_true, y_pred) in models.items():
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    print(f'\n{name}模型评估：')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'中位数绝对误差 (MedAE): {medae:.4f}')

# 查看线性回归系数
print("\n线性回归模型系数:")
for i, col in enumerate(X.columns):
    print(f"{col}: {linear_reg.coef_[i]:.4f}")
print(f"截距: {linear_reg.intercept_:.4f}")

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__

# 绘制各模型预测值与真实值的散点图
plt.figure(figsize=(15, 10))
for i, (name, (y_true, y_pred)) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} (R²={r2_score(y_true, y_pred):.4f})')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

plt.tight_layout()
plt.savefig(os.path.join("Linear Regression Plots", "model_comparison2.0.png"), dpi=300)
plt.show()