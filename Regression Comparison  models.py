# 这个文件用来对比基于“aspect_ratio”与“Height和Width”的线性回归与多项式回归
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import PolynomialFeatures
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

# 定义多项式阶数列表
poly_degrees = [2, 3]

# 准备模型数据
datasets = {}
models = {}
predictions = {}
results = {}
model_names = {}

# 数据集1: aspect_ratio + 线性回归
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

key = 'ratio_linear'
datasets[key] = {
    'X_train': X_train_ratio, 'X_val': X_val_ratio, 'X_test': X_test_ratio,
    'y_train': y_train_ratio, 'y_val': y_val_ratio, 'y_test': y_test_ratio
}
model_names[key] = 'Linear Regression (with aspect_ratio)'

# 数据集2: Height+Width + 线性回归
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

key = 'dim_linear'
datasets[key] = {
    'X_train': X_train_dim, 'X_val': X_val_dim, 'X_test': X_test_dim,
    'y_train': y_train_dim, 'y_val': y_val_dim, 'y_test': y_test_dim
}
model_names[key] = 'Linear Regression (with Height+Width)'

# 多项式回归数据集
for degree in poly_degrees:
    # aspect_ratio + 多项式回归
    poly_ratio = PolynomialFeatures(degree=degree)
    X_train_poly_ratio = poly_ratio.fit_transform(X_train_ratio)
    X_val_poly_ratio = poly_ratio.transform(X_val_ratio)
    X_test_poly_ratio = poly_ratio.transform(X_test_ratio)

    key = f'ratio_poly_{degree}'
    datasets[key] = {
        'X_train': X_train_poly_ratio, 'X_val': X_val_poly_ratio, 'X_test': X_test_poly_ratio,
        'y_train': y_train_ratio, 'y_val': y_val_ratio, 'y_test': y_test_ratio
    }
    model_names[key] = f'Polynomial Regression(degree={degree}, with aspect_ratio)'

    # Height+Width + 多项式回归
    poly_dim = PolynomialFeatures(degree=degree)
    X_train_poly_dim = poly_dim.fit_transform(X_train_dim)
    X_val_poly_dim = poly_dim.transform(X_val_dim)
    X_test_poly_dim = poly_dim.transform(X_test_dim)

    key = f'dim_poly_{degree}'
    datasets[key] = {
        'X_train': X_train_poly_dim, 'X_val': X_val_poly_dim, 'X_test': X_test_poly_dim,
        'y_train': y_train_dim, 'y_val': y_val_dim, 'y_test': y_test_dim
    }
    model_names[key] = f'Polynomial Regression(degree={degree}, with Height+Width)'

print("\n数据集准备完成:")
for key, data in datasets.items():
    print(
        f"- {key}: 训练集形状={data['X_train'].shape}, 验证集形状={data['X_val'].shape}, 测试集形状={data['X_test'].shape}")

# 定义Ridge回归的正则化参数搜索空间
alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

# 训练所有模型并进行超参数优化
print("\n开始模型训练和超参数优化...")
for key in datasets.keys():
    print(f"\n训练和优化 {model_names[key]}...")

    if '_linear' in key:
        # 线性回归无需超参数优化
        models[key] = LinearRegression()
        models[key].fit(datasets[key]['X_train'], datasets[key]['y_train'])

        # 在验证集上评估
        y_val_pred = models[key].predict(datasets[key]['X_val'])
        val_r2 = r2_score(datasets[key]['y_val'], y_val_pred)

        print(f"  验证集R²: {val_r2:.4f}")
    else:
        # 多项式回归使用Ridge回归并进行超参数优化
        best_alpha = None
        best_r2 = -np.inf
        best_model = None

        # 遍历不同的alpha值，找到在验证集上性能最好的模型
        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(datasets[key]['X_train'], datasets[key]['y_train'])

            # 在验证集上评估
            y_val_pred = model.predict(datasets[key]['X_val'])
            val_r2 = r2_score(datasets[key]['y_val'], y_val_pred)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_alpha = alpha
                best_model = model

        models[key] = best_model
        print(f"  最优alpha: {best_alpha}, 验证集R²: {best_r2:.4f}")

    # 在测试集上进行最终评估
    predictions[key] = models[key].predict(datasets[key]['X_test'])
    print(f"  {model_names[key]} 训练完成！")

# 评估各模型
print('\n各模型测试集评估结果（保留4位小数）:')
print('=' * 50)

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
for key in list(datasets.keys()):
    if '_poly_' in key:
        print(f"\n{model_names[key]}:")
        model = models[key]
        degree = int(key.split('_')[-1])

        # 获取已经fit过的PolynomialFeatures实例
        if 'ratio' in key:
            poly = PolynomialFeatures(degree=degree)
            poly.fit(X_ratio)  # 先fit
            feature_names = poly.get_feature_names_out(input_features=list(X_ratio.columns))
        else:
            poly = PolynomialFeatures(degree=degree)
            poly.fit(X_dim)  # 先fit
            feature_names = poly.get_feature_names_out(input_features=list(X_dim.columns))

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
plt.figure(figsize=(15, 15))
keys = list(predictions.keys())
num_models = len(keys)
rows = (num_models + 2) // 3  # 每行3个图

for i, key in enumerate(keys, 1):
    plt.subplot(rows, 3, i)
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

# 绘制评估指标对比图（折线图）
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
plt.figure(figsize=(16, 10))

# 为每个模型设置颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
markers = ['o', 's', '^', 'D', 'x', '*', 'p']

# 绘制所有指标的折线图
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)

    # 为每个模型绘制折线
    for j, key in enumerate(keys):
        value = results[key][metric]
        # 使用模型名称的简化版本作为标签
        model_label = model_names[key].replace('Regression', '').replace('(with', '(').replace(')', '')
        plt.plot(j, value, marker=markers[j], color=colors[j], markersize=8,
                 label=model_label if i == 1 else "")  # 只在第一个子图显示图例

    # 设置图表属性
    plt.title(f'Model {metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(range(len(keys)), [f'M{i + 1}' for i in range(len(keys))], rotation=0)  # 使用M1,M2等简化x轴标签

    # 添加数值标签
    for j, key in enumerate(keys):
        value = results[key][metric]
        plt.annotate(f'{value:.4f}',
                     (j, value),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     rotation=45 if metric == 'R2' else 0)  # R2指标标签旋转45度避免重叠

    # 设置R2的y轴范围
    if metric == 'R2':
        plt.ylim(0, 1)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局并添加图例
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为右侧的图例留出空间
plt.figlegend(loc='center right', bbox_to_anchor=(1, 0.5),
              title='Models', fontsize='medium')

plt.savefig(os.path.join("Regression_Comparison", "metrics_comparison_line.png"), dpi=300, bbox_inches='tight')
plt.show()


print("\n可视化图表已保存至: Regression_Comparison/")