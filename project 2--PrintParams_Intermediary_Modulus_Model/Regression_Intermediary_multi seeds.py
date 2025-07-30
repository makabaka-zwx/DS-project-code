import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
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

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "Regression_Intermediary_regression_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始中介效应回归分析，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据 - 不考虑aspect_ratio，只保留宽高作为可能的中介变量
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据准备完成:")
print(f"- 包含的特征: {list(data.columns)}")
print(f"- 打印参数(自变量): ['printing_temperature', 'feed_rate', 'printing_speed']")
print(f"- 中介变量(同时作为因变量): ['Height', 'Width']")
print(f"- 目标变量: 'Experiment_mean(MPa)'")

# 定义多项式阶数列表
poly_degrees = [2, 3]

# 定义变量
predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # 打印参数
mediators = ['Height', 'Width']  # 中介变量
target = 'Experiment_mean(MPa)'  # 最终目标变量

# 数据集划分：训练集(70%)、验证集(15%)、测试集(15%)
# 先划分为训练集和临时集(30%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
# 再将临时集划分为验证集和测试集(各15%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

print("\n数据集划分完成:")
print(f"- 训练集样本数: {len(train_data)} ({len(train_data) / len(data):.1%})")
print(f"- 验证集样本数: {len(val_data)} ({len(val_data) / len(data):.1%})")
print(f"- 测试集样本数: {len(test_data)} ({len(test_data) / len(data):.1%})")

# 准备模型字典
models = {}  # 存储所有模型
predictions = {}  # 存储所有预测结果
results = {}  # 存储所有评估结果
model_names = {}  # 存储模型名称

# --------------------------
# 1. 中介模型：打印参数 → 宽高 → 机械模量
# --------------------------

# 1.1 第一步：用打印参数预测宽和高（将宽和高作为因变量）
# 为每个中介变量和多项式阶数创建模型

# 线性回归预测宽高
for mediator in mediators:
    # 训练模型
    model = LinearRegression()
    model.fit(train_data[predictors], train_data[mediator])

    # 存储模型
    key = f'mediator_linear_{mediator.lower()}'
    models[key] = model
    model_names[key] = f'Linear Regression (predict {mediator})'

    # 验证模型
    val_pred = model.predict(val_data[predictors])
    val_r2 = r2_score(val_data[mediator], val_pred)
    print(f"\n{model_names[key]} 验证集R²: {val_r2:.4f}")

# 多项式回归预测宽高
for degree in poly_degrees:
    for mediator in mediators:
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(train_data[predictors])
        X_val_poly = poly.transform(val_data[predictors])

        # 超参数优化
        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_poly, train_data[mediator])
            val_pred = model.predict(X_val_poly)
            val_r2 = r2_score(val_data[mediator], val_pred)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_alpha = alpha
                best_model = model

        # 存储模型和多项式转换器
        key = f'mediator_poly{degree}_{mediator.lower()}'
        models[key] = {
            'model': best_model,
            'poly': poly
        }
        model_names[key] = f'Polynomial Regression (degree={degree}, predict {mediator})'
        print(f"{model_names[key]} 最优alpha: {best_alpha}, 验证集R²: {best_r2:.4f}")


# 1.2 第二步：用预测的宽高预测机械模量（中介模型）
# 线性中介模型
def create_mediation_model(degree=1):
    """创建中介模型：先用打印参数预测宽高，再用预测的宽高预测机械模量"""
    # 预测宽高的模型
    width_model_key = f'mediator_linear_width' if degree == 1 else f'mediator_poly{degree}_width'
    height_model_key = f'mediator_linear_height' if degree == 1 else f'mediator_poly{degree}_height'

    # 从宽高预测机械模量的模型
    # 准备训练数据：用预测的宽高作为特征
    if degree == 1:
        train_pred_width = models[width_model_key].predict(train_data[predictors])
        train_pred_height = models[height_model_key].predict(train_data[predictors])
    else:
        train_pred_width = models[width_model_key]['model'].predict(
            models[width_model_key]['poly'].transform(train_data[predictors])
        )
        train_pred_height = models[height_model_key]['model'].predict(
            models[height_model_key]['poly'].transform(train_data[predictors])
        )

    train_mediation_features = pd.DataFrame({
        'predicted_width': train_pred_width,
        'predicted_height': train_pred_height
    })

    # 准备验证数据
    if degree == 1:
        val_pred_width = models[width_model_key].predict(val_data[predictors])
        val_pred_height = models[height_model_key].predict(val_data[predictors])
    else:
        val_pred_width = models[width_model_key]['model'].predict(
            models[width_model_key]['poly'].transform(val_data[predictors])
        )
        val_pred_height = models[height_model_key]['model'].predict(
            models[height_model_key]['poly'].transform(val_data[predictors])
        )

    val_mediation_features = pd.DataFrame({
        'predicted_width': val_pred_width,
        'predicted_height': val_pred_height
    })

    # 训练最终预测机械模量的模型
    if degree == 1:
        final_model = LinearRegression()
        final_model.fit(train_mediation_features, train_data[target])

        # 验证模型
        val_pred = final_model.predict(val_mediation_features)
        val_r2 = r2_score(val_data[target], val_pred)
        print(f"线性中介模型 验证集R²: {val_r2:.4f}")
        return final_model, None
    else:
        # 多项式模型使用Ridge回归
        poly_final = PolynomialFeatures(degree=1)  # 宽高到机械模量用线性即可
        X_train_final = poly_final.fit_transform(train_mediation_features)
        X_val_final = poly_final.transform(val_mediation_features)

        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_final, train_data[target])
            val_pred = model.predict(X_val_final)
            val_r2 = r2_score(val_data[target], val_pred)

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_alpha = alpha
                best_model = model

        print(f"多项式中介模型(degree={degree}) 最优alpha: {best_alpha}, 验证集R²: {best_r2:.4f}")
        return best_model, poly_final


# 创建不同阶数的中介模型
for degree in [1] + poly_degrees:  # 1代表线性
    model, poly = create_mediation_model(degree)
    key = f'mediation_model_degree{degree}'
    models[key] = {
        'model': model,
        'poly': poly,
        'degree': degree
    }
    model_names[key] = f'Mediation Model (degree={degree})'

# --------------------------
# 2. 直接模型：打印参数直接预测机械模量
# --------------------------

# 线性直接模型
model = LinearRegression()
model.fit(train_data[predictors], train_data[target])
val_pred = model.predict(val_data[predictors])
val_r2 = r2_score(val_data[target], val_pred)
key = 'direct_linear'
models[key] = model
model_names[key] = 'Direct Linear Model'
print(f"\n{model_names[key]} 验证集R²: {val_r2:.4f}")

# 多项式直接模型
for degree in poly_degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(train_data[predictors])
    X_val_poly = poly.transform(val_data[predictors])

    best_alpha = None
    best_r2 = -np.inf
    best_model = None
    alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

    for alpha in alpha_values:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_train_poly, train_data[target])
        val_pred = model.predict(X_val_poly)
        val_r2 = r2_score(val_data[target], val_pred)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_alpha = alpha
            best_model = model

    key = f'direct_poly{degree}'
    models[key] = {
        'model': best_model,
        'poly': poly
    }
    model_names[key] = f'Direct Polynomial Model (degree={degree})'
    print(f"{model_names[key]} 最优alpha: {best_alpha}, 验证集R²: {best_r2:.4f}")

# --------------------------
# 3. 混合模型：打印参数 + 实际宽高预测机械模量（作为参考）
# --------------------------

hybrid_features = predictors + mediators

# 线性混合模型
model = LinearRegression()
model.fit(train_data[hybrid_features], train_data[target])
val_pred = model.predict(val_data[hybrid_features])
val_r2 = r2_score(val_data[target], val_pred)
key = 'hybrid_linear'
models[key] = model
model_names[key] = 'Hybrid Linear Model'
print(f"\n{model_names[key]} 验证集R²: {val_r2:.4f}")

# 多项式混合模型
for degree in poly_degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(train_data[hybrid_features])
    X_val_poly = poly.transform(val_data[hybrid_features])

    best_alpha = None
    best_r2 = -np.inf
    best_model = None
    alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

    for alpha in alpha_values:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_train_poly, train_data[target])
        val_pred = model.predict(X_val_poly)
        val_r2 = r2_score(val_data[target], val_pred)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_alpha = alpha
            best_model = model

    key = f'hybrid_poly{degree}'
    models[key] = {
        'model': best_model,
        'poly': poly
    }
    model_names[key] = f'Hybrid Polynomial Model (degree={degree})'
    print(f"{model_names[key]} 最优alpha: {best_alpha}, 验证集R²: {best_r2:.4f}")

# --------------------------
# 在测试集上评估所有模型
# --------------------------

print('\n' + '=' * 50)
print("所有模型测试集评估结果（保留4位小数）:")
print('=' * 50)

# 评估中介模型
for degree in [1] + poly_degrees:
    key = f'mediation_model_degree{degree}'
    model_info = models[key]
    degree_mediator = model_info['degree']

    # 先预测宽高
    width_model_key = f'mediator_linear_width' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_width'
    height_model_key = f'mediator_linear_height' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_height'

    if degree_mediator == 1:
        test_pred_width = models[width_model_key].predict(test_data[predictors])
        test_pred_height = models[height_model_key].predict(test_data[predictors])
    else:
        test_pred_width = models[width_model_key]['model'].predict(
            models[width_model_key]['poly'].transform(test_data[predictors])
        )
        test_pred_height = models[height_model_key]['model'].predict(
            models[height_model_key]['poly'].transform(test_data[predictors])
        )

    # 构建中介特征
    test_mediation_features = pd.DataFrame({
        'predicted_width': test_pred_width,
        'predicted_height': test_pred_height
    })

    # 预测机械模量
    if model_info['poly'] is None:
        y_pred = model_info['model'].predict(test_mediation_features)
    else:
        X_test_final = model_info['poly'].transform(test_mediation_features)
        y_pred = model_info['model'].predict(X_test_final)

    # 评估
    y_true = test_data[target]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    predictions[key] = y_pred
    results[key] = {
        'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
    }

    print(f'\n{model_names[key]}评估：')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'中位数绝对误差 (MedAE): {medae:.4f}')

# 评估直接模型
key = 'direct_linear'
model = models[key]
y_pred = model.predict(test_data[predictors])
y_true = test_data[target]
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)

predictions[key] = y_pred
results[key] = {
    'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
}

print(f'\n{model_names[key]}评估：')
print(f'均方误差 (MSE): {mse:.4f}')
print(f'决定系数 (R2): {r2:.4f}')
print(f'平均绝对误差 (MAE): {mae:.4f}')
print(f'中位数绝对误差 (MedAE): {medae:.4f}')

for degree in poly_degrees:
    key = f'direct_poly{degree}'
    model_info = models[key]
    X_test_poly = model_info['poly'].transform(test_data[predictors])
    y_pred = model_info['model'].predict(X_test_poly)
    y_true = test_data[target]

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    predictions[key] = y_pred
    results[key] = {
        'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
    }

    print(f'\n{model_names[key]}评估：')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'中位数绝对误差 (MedAE): {medae:.4f}')

# 评估混合模型
key = 'hybrid_linear'
model = models[key]
y_pred = model.predict(test_data[hybrid_features])
y_true = test_data[target]
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)

predictions[key] = y_pred
results[key] = {
    'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
}

print(f'\n{model_names[key]}评估：')
print(f'均方误差 (MSE): {mse:.4f}')
print(f'决定系数 (R2): {r2:.4f}')
print(f'平均绝对误差 (MAE): {mae:.4f}')
print(f'中位数绝对误差 (MedAE): {medae:.4f}')

for degree in poly_degrees:
    key = f'hybrid_poly{degree}'
    model_info = models[key]
    X_test_poly = model_info['poly'].transform(test_data[hybrid_features])
    y_pred = model_info['model'].predict(X_test_poly)
    y_true = test_data[target]

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    predictions[key] = y_pred
    results[key] = {
        'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
    }

    print(f'\n{model_names[key]}评估：')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'中位数绝对误差 (MedAE): {medae:.4f}')

# 查看线性模型系数
print("\n线性模型系数分析:")
print("-" * 30)

# 打印参数对宽的影响系数
key = 'mediator_linear_width'
print(f"\n{model_names[key]} 系数:")
for i, col in enumerate(predictors):
    print(f"{col}: {models[key].coef_[i]:.4f}")
print(f"截距: {models[key].intercept_:.4f}")

# 打印参数对高的影响系数
key = 'mediator_linear_height'
print(f"\n{model_names[key]} 系数:")
for i, col in enumerate(predictors):
    print(f"{col}: {models[key].coef_[i]:.4f}")
print(f"截距: {models[key].intercept_:.4f}")

# 宽高对机械模量的影响系数（中介模型）
key = 'mediation_model_degree1'
print(f"\n{model_names[key]} 系数 (宽高对机械模量的影响):")
for i, col in enumerate(['predicted_width', 'predicted_height']):
    print(f"{col}: {models[key]['model'].coef_[i]:.4f}")
print(f"截距: {models[key]['model'].intercept_:.4f}")

# 打印参数对机械模量的直接影响系数
key = 'direct_linear'
print(f"\n{model_names[key]} 系数 (打印参数对机械模量的直接影响):")
for i, col in enumerate(predictors):
    print(f"{col}: {models[key].coef_[i]:.4f}")
print(f"截距: {models[key].intercept_:.4f}")

# 中介效应分析
print("\n" + "=" * 50)
print("中介效应分析结果:")
print("-" * 30)

# 提取关键模型的R²值
med_r2 = results['mediation_model_degree1']['R2']  # 中介模型
dir_r2 = results['direct_linear']['R2']  # 直接模型
hyb_r2 = results['hybrid_linear']['R2']  # 混合模型

# 计算中介效应比例
total_effect = dir_r2
direct_effect = hyb_r2 - med_r2  # 控制中介变量后的直接效应
mediation_effect = total_effect - direct_effect  # 中介效应 = 总效应 - 直接效应
mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

print(f"总效应 (直接模型R²): {total_effect:.4f}")
print(f"直接效应 (控制宽高后): {direct_effect:.4f}")
print(f"中介效应 (通过宽高): {mediation_effect:.4f}")
print(f"中介比例 (中介效应/总效应): {mediation_ratio:.2%}")

# 找出R²最高的模型
best_model_key = max(results, key=lambda k: results[k]['R2'])
best_model_name = model_names[best_model_key]
best_r2 = results[best_model_key]['R2']

print(f"\nR²最高的模型: {best_model_name} (R²={best_r2:.4f})")

# 绘制R²最高模型的拟合图像
plt.figure(figsize=(15, 6))

# 预测值与真实值散点图
plt.subplot(1, 2, 1)
y_true = test_data[target]
y_pred = predictions[best_model_key]
plt.scatter(y_true, y_pred, alpha=0.7)
plt.xlabel('True Values (MPa)')
plt.ylabel('Predicted Values (MPa)')
plt.title(f'{best_model_name} - Predicted vs True Values')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

# 添加R²和MSE信息
plt.annotate(f'R² = {best_r2:.4f}\nMSE = {results[best_model_key]["MSE"]:.4f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
             fontsize=10, ha='left', va='top')

# 残差图
plt.subplot(1, 2, 2)
residuals = y_true - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Value (MPa)')
plt.ylabel('Residual')
plt.title(f'{best_model_name} - Residual Plot')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(get_unique_filename(os.path.join("outputs", "Regression_Intermediary_best_model_fit.png")), dpi=300)
plt.show()

# 绘制各模型预测值与真实值的散点图
plt.figure(figsize=(18, 12))
keys = list(predictions.keys())
num_models = len(keys)
rows = (num_models + 2) // 3  # 每行3个图

for i, key in enumerate(keys, 1):
    plt.subplot(rows, 3, i)
    y_true = test_data[target]
    y_pred = predictions[key]
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_names[key]} (R²={results[key]["R2"]:.4f})')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(get_unique_filename(os.path.join("outputs", "Regression_Intermediary_model_comparison.png")), dpi=300)
plt.show()

# 绘制评估指标对比图
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
plt.figure(figsize=(16, 10))

# 为每个模型设置颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h', '+']

# 绘制所有指标的柱状图
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)

    # 为每个模型绘制柱状图
    values = [results[key][metric] for key in keys]
    x_pos = np.arange(len(keys))

    plt.bar(x_pos, values, color=colors[:len(keys)], alpha=0.7)

    # 设置图表属性
    plt.title(f'Model {metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(x_pos, [f'M{i + 1}' for i in range(len(keys))], rotation=0)

    # 添加数值标签
    for j, v in enumerate(values):
        plt.text(j, v + 0.01, f'{v:.4f}', ha='center', rotation=45, fontsize=8)

    # 设置R2的y轴范围
    if metric == 'R2':
        plt.ylim(0, 1)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局并添加图例
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为右侧的图例留出空间
plt.figlegend([model_names[key] for key in keys],
              loc='center right', bbox_to_anchor=(1, 0.5),
              title='Models', fontsize='small')

plt.savefig(get_unique_filename(os.path.join("outputs", "Regression_Intermediary_metrics_comparison.png")), dpi=300,
            bbox_inches='tight')
plt.show()

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练与评估完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")
print(f"可视化结果已保存到: outputs/")

# 恢复标准输出
sys.stdout = sys.__stdout__
