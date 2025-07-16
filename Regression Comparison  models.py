
# 这个文件用来对比基于“aspect_ratio”与“Height和Width”的线性回归与多项式回归


import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
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
os.makedirs("Regression_Comparison", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "model_training_log_regression_comparison.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始模型训练，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 准备特征集1：使用aspect_ratio
data_with_ratio = data.copy()
data_with_ratio['aspect_ratio'] = data_with_ratio['Height'] / data_with_ratio['Width']
selected_columns_ratio = ['printing_temperature', 'feed_rate', 'printing_speed', 'aspect_ratio', 'Experiment_mean(MPa)']
data_with_ratio = data_with_ratio[selected_columns_ratio]

# 准备特征集2：使用Height和Width
selected_columns_dim = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
data_with_dim = data[selected_columns_dim]

print("数据准备完成:")
print(f"- 特征集1 (aspect_ratio): {list(data_with_ratio.columns)}")
print(f"- 特征集2 (Height+Width): {list(data_with_dim.columns)}")

# 定义多项式阶数
poly_degree = 2  # 可调整为2、3等

# 准备四种模型的数据
datasets = {}

# 数据集1: aspect_ratio + 线性回归
X_ratio = data_with_ratio.drop('Experiment_mean(MPa)', axis=1)
y_ratio = data_with_ratio['Experiment_mean(MPa)']
X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(
    X_ratio, y_ratio, test_size=0.2, random_state=seed
)
datasets['ratio_linear'] = {
    'X_train': X_train_ratio, 'X_test': X_test_ratio,
    'y_train': y_train_ratio, 'y_test': y_test_ratio
}

# 数据集2: aspect_ratio + 多项式回归
poly_ratio = PolynomialFeatures(degree=poly_degree)
X_train_poly_ratio = poly_ratio.fit_transform(X_train_ratio)
X_test_poly_ratio = poly_ratio.transform(X_test_ratio)
datasets['ratio_poly'] = {
    'X_train': X_train_poly_ratio, 'X_test': X_test_poly_ratio,
    'y_train': y_train_ratio, 'y_test': y_test_ratio
}

# 数据集3: Height+Width + 线性回归
X_dim = data_with_dim.drop('Experiment_mean(MPa)', axis=1)
y_dim = data_with_dim['Experiment_mean(MPa)']
X_train_dim, X_test_dim, y_train_dim, y_test_dim = train_test_split(
    X_dim, y_dim, test_size=0.2, random_state=seed
)
datasets['dim_linear'] = {
    'X_train': X_train_dim, 'X_test': X_test_dim,
    'y_train': y_train_dim, 'y_test': y_test_dim
}

# 数据集4: Height+Width + 多项式回归
poly_dim = PolynomialFeatures(degree=poly_degree)
X_train_poly_dim = poly_dim.fit_transform(X_train_dim)
X_test_poly_dim = poly_dim.transform(X_test_dim)
datasets['dim_poly'] = {
    'X_train': X_train_poly_dim, 'X_test': X_test_poly_dim,
    'y_train': y_train_dim, 'y_test': y_test_dim
}

print("\n数据集准备完成:")
for key, data in datasets.items():
    print(f"- {key}: 训练集形状={data['X_train'].shape}, 测试集形状={data['X_test'].shape}")

# 初始化并训练四种模型
models = {}
predictions = {}

# 1. aspect_ratio + 线性回归
print("\n1. 训练基于aspect_ratio的线性回归模型...")
models['ratio_linear'] = LinearRegression()
models['ratio_linear'].fit(datasets['ratio_linear']['X_train'], datasets['ratio_linear']['y_train'])
predictions['ratio_linear'] = models['ratio_linear'].predict(datasets['ratio_linear']['X_test'])
print("  模型训练完成！")

# 2. aspect_ratio + 多项式回归
print("2. 训练基于aspect_ratio的多项式回归模型...")
# 使用Ridge回归处理多项式过拟合
models['ratio_poly'] = Ridge(alpha=1.0, random_state=seed)
models['ratio_poly'].fit(datasets['ratio_poly']['X_train'], datasets['ratio_poly']['y_train'])
predictions['ratio_poly'] = models['ratio_poly'].predict(datasets['ratio_poly']['X_test'])
print("  模型训练完成！")

# 3. Height+Width + 线性回归
print("3. 训练基于Height+Width的线性回归模型...")
models['dim_linear'] = LinearRegression()
models['dim_linear'].fit(datasets['dim_linear']['X_train'], datasets['dim_linear']['y_train'])
predictions['dim_linear'] = models['dim_linear'].predict(datasets['dim_linear']['X_test'])
print("  模型训练完成！")

# 4. Height+Width + 多项式回归
print("4. 训练基于Height+Width的多项式回归模型...")
# 使用Ridge回归处理多项式过拟合
models['dim_poly'] = Ridge(alpha=1.0, random_state=seed)
models['dim_poly'].fit(datasets['dim_poly']['X_train'], datasets['dim_poly']['y_train'])
predictions['dim_poly'] = models['dim_poly'].predict(datasets['dim_poly']['X_test'])
print("  模型训练完成！")

# 模型命名
model_names = {
    'ratio_linear': '线性回归(使用aspect_ratio)',
    'ratio_poly': f'多项式回归(degree={poly_degree}, 使用aspect_ratio)',
    'dim_linear': '线性回归(使用Height+Width)',
    'dim_poly': f'多项式回归(degree={poly_degree}, 使用Height+Width)'
}

# 评估各模型
print('\n各模型评估结果（保留4位小数）:')
print('=' * 50)

results = {}

for key, y_pred in predictions.items():
    y_true = datasets[key]['y_test']
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    results[key] = {
        'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
    }

    print(f'\n{model_names[key]}评估：')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'中位数绝对误差 (MedAE): {medae:.4f}')

# 查看线性回归系数
print("\n线性回归模型系数:")
print("-" * 30)
for key in ['ratio_linear', 'dim_linear']:
    print(f"\n{model_names[key]}:")
    model = models[key]
    if key == 'ratio_linear':
        features = list(datasets[key]['X_train'].columns)
    else:
        features = list(datasets[key]['X_train'].columns)

    for i, col in enumerate(features):
        print(f"{col}: {model.coef_[i]:.4f}")
    print(f"截距: {model.intercept_:.4f}")

# 多项式回归特征重要性（仅显示前10个）
print("\n多项式回归特征重要性（绝对值前10）:")
print("-" * 30)
for key in ['ratio_poly', 'dim_poly']:
    print(f"\n{model_names[key]}:")
    model = models[key]
    if key == 'ratio_poly':
        feature_names = poly_ratio.get_feature_names_out(
            input_features=list(datasets['ratio_linear']['X_train'].columns))
    else:
        feature_names = poly_dim.get_feature_names_out(input_features=list(datasets['dim_linear']['X_train'].columns))

    # 获取特征重要性（系数绝对值）
    importances = np.abs(model.coef_)
    indices = np.argsort(importances)[-10:][::-1]  # 取前10个

    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

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
for i, key in enumerate(predictions.keys(), 1):
    plt.subplot(2, 2, i)
    y_true = datasets[key]['y_test']
    y_pred = predictions[key]
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_names[key]} (R²={results[key]["R2"]:.4f})')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

plt.tight_layout()
plt.savefig(os.path.join("Regression_Comparison", "model_comparison.png"), dpi=300)
plt.show()

# 绘制评估指标对比图
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    values = [results[key][metric] for key in predictions.keys()]
    bars = plt.bar([model_names[key] for key in predictions.keys()], values)

    if metric == 'R2':
        plt.ylim(0, 1)  # R2范围在0-1之间

    plt.title(f'Model {metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join("Regression_Comparison", "metrics_comparison.png"), dpi=300)
plt.show()

print("\n可视化图表已保存至: Regression_Comparison/")