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

# 种子设置与RF保持一致
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]  # 共9个种子


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


def run_regression_experiment(seed):
    """运行单次回归实验并返回结果"""
    np.random.seed(seed)

    # 加载数据 - 不考虑aspect_ratio，只保留宽高作为可能的中介变量
    data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
    selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
    data = data[selected_columns]

    # 定义多项式阶数列表
    poly_degrees = [2, 3]

    # 定义变量
    predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # 打印参数
    mediators = ['Height', 'Width']  # 中介变量
    target = 'Experiment_mean(MPa)'  # 最终目标变量

    # 数据集划分：训练集(70%)、验证集(15%)、测试集(15%)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # 准备模型字典
    models = {}  # 存储所有模型
    predictions = {}  # 存储所有预测结果
    results = {}  # 存储所有评估结果
    model_names = {}  # 存储模型名称

    # --------------------------
    # 1. 中介模型：打印参数 → 宽高 → 机械模量
    # --------------------------

    # 1.1 第一步：用打印参数预测宽和高
    # 线性回归预测宽高
    for mediator in mediators:
        model = LinearRegression()
        model.fit(train_data[predictors], train_data[mediator])
        key = f'mediator_linear_{mediator.lower()}'
        models[key] = model
        model_names[key] = f'Linear Regression (predict {mediator})'

    # 多项式回归预测宽高
    for degree in poly_degrees:
        for mediator in mediators:
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(train_data[predictors])
            X_val_poly = poly.transform(val_data[predictors])

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

            key = f'mediator_poly{degree}_{mediator.lower()}'
            models[key] = {
                'model': best_model,
                'poly': poly
            }
            model_names[key] = f'Polynomial Regression (degree={degree}, predict {mediator})'

    # 1.2 第二步：用预测的宽高预测机械模量（中介模型）
    def create_mediation_model(degree=1):
        width_model_key = f'mediator_linear_width' if degree == 1 else f'mediator_poly{degree}_width'
        height_model_key = f'mediator_linear_height' if degree == 1 else f'mediator_poly{degree}_height'

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

        if degree == 1:
            final_model = LinearRegression()
            final_model.fit(train_mediation_features, train_data[target])
            return final_model, None
        else:
            poly_final = PolynomialFeatures(degree=1)
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

            return best_model, poly_final

    # 创建不同阶数的中介模型
    for degree in [1] + poly_degrees:
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
    key = 'direct_linear'
    models[key] = model
    model_names[key] = 'Direct Linear Model'

    # 多项式直接模型
    for degree in poly_degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(train_data[predictors])

        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_poly, train_data[target])
            val_pred = model.predict(poly.transform(val_data[predictors]))
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

    # --------------------------
    # 3. 混合模型：打印参数 + 实际宽高预测机械模量
    # --------------------------

    hybrid_features = predictors + mediators

    # 线性混合模型
    model = LinearRegression()
    model.fit(train_data[hybrid_features], train_data[target])
    key = 'hybrid_linear'
    models[key] = model
    model_names[key] = 'Hybrid Linear Model'

    # 多项式混合模型
    for degree in poly_degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(train_data[hybrid_features])

        best_alpha = None
        best_r2 = -np.inf
        best_model = None
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for alpha in alpha_values:
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_poly, train_data[target])
            val_pred = model.predict(poly.transform(val_data[hybrid_features]))
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

    # --------------------------
    # 在测试集上评估所有模型
    # --------------------------

    # 评估中介模型
    for degree in [1] + poly_degrees:
        key = f'mediation_model_degree{degree}'
        model_info = models[key]
        degree_mediator = model_info['degree']

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

        test_mediation_features = pd.DataFrame({
            'predicted_width': test_pred_width,
            'predicted_height': test_pred_height
        })

        if model_info['poly'] is None:
            y_pred = model_info['model'].predict(test_mediation_features)
        else:
            X_test_final = model_info['poly'].transform(test_mediation_features)
            y_pred = model_info['model'].predict(X_test_final)

        y_true = test_data[target]
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)

        predictions[key] = y_pred
        results[key] = {
            'MSE': mse, 'R2': r2, 'MAE': mae, 'MedAE': medae
        }

    # 评估直接模型
    key = 'direct_linear'
    model = models[key]
    y_pred = model.predict(test_data[predictors])
    y_true = test_data[target]
    results[key] = {
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MedAE': median_absolute_error(y_true, y_pred)
    }

    for degree in poly_degrees:
        key = f'direct_poly{degree}'
        model_info = models[key]
        X_test_poly = model_info['poly'].transform(test_data[predictors])
        y_pred = model_info['model'].predict(X_test_poly)
        y_true = test_data[target]
        results[key] = {
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MedAE': median_absolute_error(y_true, y_pred)
        }

    # 评估混合模型
    key = 'hybrid_linear'
    model = models[key]
    y_pred = model.predict(test_data[hybrid_features])
    y_true = test_data[target]
    results[key] = {
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MedAE': median_absolute_error(y_true, y_pred)
    }

    for degree in poly_degrees:
        key = f'hybrid_poly{degree}'
        model_info = models[key]
        X_test_poly = model_info['poly'].transform(test_data[hybrid_features])
        y_pred = model_info['model'].predict(X_test_poly)
        y_true = test_data[target]
        results[key] = {
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MedAE': median_absolute_error(y_true, y_pred)
        }

    return results, model_names, test_data[target]


def calculate_averages(results_list):
    """计算多次实验结果的平均值"""
    if not results_list:
        return {}

    # 初始化平均结果字典
    avg_results = {}
    model_keys = results_list[0].keys()

    # 为每个模型初始化指标列表
    for model_key in model_keys:
        avg_results[model_key] = {
            'MSE': [],
            'R2': [],
            'MAE': [],
            'MedAE': []
        }

    # 收集所有实验结果
    for result in results_list:
        for model_key in model_keys:
            for metric in avg_results[model_key]:
                avg_results[model_key][metric].append(result[model_key][metric])

    # 计算平均值
    for model_key in model_keys:
        for metric in avg_results[model_key]:
            avg_results[model_key][metric] = np.mean(avg_results[model_key][metric])

    return avg_results


def calculate_mediation_effect(average_results):
    """计算中介效应比例"""
    # 总效应 (直接模型的效应)
    total_effect = average_results['direct_linear']['R2']

    # 直接效应 (控制中介变量后的直接效应)
    direct_effect = average_results['hybrid_linear']['R2'] - average_results['mediation_model_degree1']['R2']

    # 中介效应 = 总效应 - 直接效应
    mediation_effect = total_effect - direct_effect

    # 中介比例
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


# 主程序
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)

    # 开始计时
    start_time = time.time()

    # 生成唯一的日志文件名
    base_log_file = "Regression_Intermediary(multi seeds)_regression_log.txt"
    log_file = get_unique_filename(os.path.join("outputs", base_log_file))

    # 重定向输出流
    sys.stdout = Logger(log_file)

    print(f"开始中介效应回归分析，日志将保存到 {log_file}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用种子列表: {seeds}")
    print("=" * 50)

    # 运行多次实验
    results_list = []
    all_predictions = {}
    model_names = None
    y_true = None

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 30}")
        print(f"运行实验 {i + 1}/{len(seeds)}，种子: {seed}")
        print(f"{'=' * 30}")

        results, exp_model_names, exp_y_true = run_regression_experiment(seed)
        results_list.append(results)

        # 保存模型名称（所有实验相同）
        if model_names is None:
            model_names = exp_model_names
            y_true = exp_y_true

        # 保存预测结果
        for model_key in results:
            if model_key not in all_predictions:
                all_predictions[model_key] = []
            # 这里简化处理，实际应用可能需要更复杂的存储方式

    # 计算平均结果
    print("\n" + "=" * 50)
    print("计算多次实验的平均结果...")
    avg_results = calculate_averages(results_list)

    # 输出平均评估结果
    print("\n" + "=" * 50)
    print("所有模型平均测试集评估结果（保留4位小数）:")
    print('=' * 50)
    for model_key in avg_results:
        print(f'\n{model_names[model_key]} 平均评估：')
        print(f'均方误差 (MSE): {avg_results[model_key]["MSE"]:.4f}')
        print(f'决定系数 (R2): {avg_results[model_key]["R2"]:.4f}')
        print(f'平均绝对误差 (MAE): {avg_results[model_key]["MAE"]:.4f}')
        print(f'中位数绝对误差 (MedAE): {avg_results[model_key]["MedAE"]:.4f}')

    # 中介效应分析（基于线性模型的平均结果）
    print("\n" + "=" * 50)
    print("中介效应分析结果 (基于平均结果):")
    print("-" * 30)
    mediation_stats = calculate_mediation_effect(avg_results)

    print(f"总效应 (直接模型平均R²): {mediation_stats['total_effect']:.4f}")
    print(f"直接效应 (控制宽高后平均): {mediation_stats['direct_effect']:.4f}")
    print(f"中介效应 (通过宽高平均): {mediation_stats['mediation_effect']:.4f}")
    print(f"中介比例 (中介效应/总效应): {mediation_stats['mediation_ratio']:.2%}")

    # 找出平均R²最高的模型
    best_model_key = max(avg_results, key=lambda k: avg_results[k]['R2'])
    best_model_name = model_names[best_model_key]
    best_r2 = avg_results[best_model_key]['R2']

    print(f"\n平均R²最高的模型: {best_model_name} (平均R²={best_r2:.4f})")




    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "=" * 50)
    print(f"所有 {len(seeds)} 次实验完成！总运行时间: {timedelta(seconds=int(total_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"详细日志已保存到: {log_file}")
    print(f"可视化结果已保存到: outputs/")

    # 恢复标准输出
    sys.stdout = sys.__stdout__