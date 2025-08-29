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
    """运行单次回归实验并返回结果（包含验证集和测试集）"""
    np.random.seed(seed)

    # 加载数据
    data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
    selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
    data = data[selected_columns]

    # 定义多项式阶数列表
    poly_degrees = [2, 3]

    # 定义变量
    predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # 打印参数（3个输入变量）
    mediators = ['Height', 'Width']  # 中介变量
    target = 'Experiment_mean(MPa)'  # 最终目标变量

    # 数据集划分：基于机械模量的分层抽样
    # 训练集(70%)、验证集(15%)、测试集(15%)
    # 首先创建分层所需的 bins

    # 动态调整分箱数量，确保每个箱至少有4个样本（满足两次分层抽样要求：train_test_split两次）
    min_samples_per_bin = 4  # 增加到4，因为我们要做两次分层抽样
    max_bins = 10

    # 计算最大可能的分箱数量
    max_possible_bins = len(data) // min_samples_per_bin
    num_bins = min(max_bins, max_possible_bins)

    # 确保至少有2个分箱
    num_bins = max(2, num_bins)

    # 创建分箱
    data['target_bin'] = pd.cut(data[target], bins=num_bins, labels=False)

    # 检查每个分箱的样本数量，如果有分箱样本数不足，合并相邻分箱
    bin_counts = data['target_bin'].value_counts().sort_index()
    while (bin_counts < min_samples_per_bin).any():
        # 找到样本最少的分箱
        min_bin = bin_counts.idxmin()
        # 合并到相邻的分箱
        if min_bin == 0:
            data['target_bin'] = data['target_bin'].replace(1, 0)
        elif min_bin == len(bin_counts) - 1:
            data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
        else:
            # 合并到样本较多的相邻分箱
            left_count = bin_counts[min_bin - 1]
            right_count = bin_counts[min_bin + 1]
            if left_count >= right_count:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin - 1)
            else:
                data['target_bin'] = data['target_bin'].replace(min_bin, min_bin + 1)
        # 重新计算分箱数量
        bin_counts = data['target_bin'].value_counts().sort_index()
        # 重命名分箱标签，确保连续
        data['target_bin'] = pd.Categorical(data['target_bin']).codes
        bin_counts = data['target_bin'].value_counts().sort_index()

        # 如果只剩下一个分箱，无法再合并，只能打破循环
        if len(bin_counts) == 1:
            break

    # 如果所有数据都在一个分箱中，使用随机抽样而非分层抽样
    stratify_param = data['target_bin'] if len(bin_counts) > 1 else None

    # 第一次分层抽样：划分训练集和临时集
    train_data, temp_data = train_test_split(
        data,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_param  # 基于目标变量分层
    )

    # 为第二次抽样准备分层参数
    if stratify_param is not None:
        stratify_param_temp = temp_data['target_bin']
        # 检查临时集中每个分箱的样本数
        temp_bin_counts = stratify_param_temp.value_counts()
        # 如果有分箱样本数不足2，改用随机抽样
        if (temp_bin_counts < 2).any():
            stratify_param_temp = None
    else:
        stratify_param_temp = None

    # 第二次分层抽样：从临时集中划分验证集和测试集
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_param_temp  # 基于目标变量分层
    )

    # 删除辅助列
    train_data = train_data.drop('target_bin', axis=1)
    val_data = val_data.drop('target_bin', axis=1)
    test_data = test_data.drop('target_bin', axis=1)

    # 准备模型字典
    models = {}  # 存储所有模型
    predictions = {}  # 存储所有预测结果
    results = {}  # 存储所有评估结果（包含val和test）
    model_names = {}  # 存储模型名称

    # --------------------------
    # 1. 中介模型：打印参数 → 宽高 → 机械模量
    #    第一层：3个输入变量（打印参数）
    #    第二层：5个输入变量（3个打印参数 + 2个预测的宽高）
    # --------------------------

    # 1.1 第一步：用打印参数预测宽和高（第一层：3个输入变量）
    # 线性回归预测宽高
    for mediator in mediators:
        model = LinearRegression()
        # 明确使用3个打印参数作为输入
        model.fit(train_data[predictors], train_data[mediator])
        key = f'mediator_linear_{mediator.lower()}'
        models[key] = model
        model_names[key] = f'Linear Regression (predict {mediator})'

    # 多项式回归预测宽高
    for degree in poly_degrees:
        for mediator in mediators:
            poly = PolynomialFeatures(degree=degree)
            # 明确使用3个打印参数作为输入
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

    # 1.2 第二步：用【打印参数 + 预测的宽高】预测机械模量（第二层：5个输入变量）
    def create_mediation_model(degree=1):
        width_model_key = f'mediator_linear_width' if degree == 1 else f'mediator_poly{degree}_width'
        height_model_key = f'mediator_linear_height' if degree == 1 else f'mediator_poly{degree}_height'

        # 预测宽高
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

        # 创建第二层特征：3个打印参数 + 2个预测的宽高（共5个特征）
        train_mediation_features = train_data[predictors].copy()
        train_mediation_features['predicted_width'] = train_pred_width
        train_mediation_features['predicted_height'] = train_pred_height

        # 验证特征数量是否为5
        assert train_mediation_features.shape[
                   1] == 5, f"中介模型第二层特征数量错误: 应为5，实际为{train_mediation_features.shape[1]}"

        # 为验证集创建相同的特征
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

        val_mediation_features = val_data[predictors].copy()
        val_mediation_features['predicted_width'] = val_pred_width
        val_mediation_features['predicted_height'] = val_pred_height

        # 验证特征数量是否为5
        assert val_mediation_features.shape[
                   1] == 5, f"中介模型第二层验证特征数量错误: 应为5，实际为{val_mediation_features.shape[1]}"

        # 训练最终模型（使用5个特征）
        if degree == 1:
            final_model = LinearRegression()
            final_model.fit(train_mediation_features, train_data[target])
            return final_model, None, val_mediation_features
        else:
            # 这里使用degree=1确保不会增加新的交互特征，保持5个输入特征
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

            return best_model, poly_final, val_mediation_features

    # 创建不同阶数的中介模型
    for degree in [1] + poly_degrees:
        model, poly, val_mediation_features = create_mediation_model(degree)
        key = f'mediation_model_degree{degree}'
        models[key] = {
            'model': model,
            'poly': poly,
            'degree': degree,
            'val_features': val_mediation_features  # 保存验证集特征用于评估
        }
        if degree == 1:
            model_names[key] = 'Mediation Model (Linear)'
        else:
            model_names[key] = f'Mediation Model (Polynomial degree={degree})'

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
        X_val_poly = poly.transform(val_data[predictors])  # 保存验证集特征

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
            'poly': poly,
            'val_features': X_val_poly  # 保存验证集特征
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
        X_val_poly = poly.transform(val_data[hybrid_features])  # 保存验证集特征

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
            'poly': poly,
            'val_features': X_val_poly  # 保存验证集特征
        }
        model_names[key] = f'Hybrid Polynomial Model (degree={degree})'

    # --------------------------
    # 评估所有模型（同时计算验证集和测试集）
    # --------------------------

    # 评估中介模型
    for degree in [1] + poly_degrees:
        key = f'mediation_model_degree{degree}'
        model_info = models[key]
        degree_mediator = model_info['degree']

        # 验证集评估
        val_mediation_features = model_info['val_features']
        if model_info['poly'] is None:
            y_pred_val = model_info['model'].predict(val_mediation_features)
        else:
            X_val_final = model_info['poly'].transform(val_mediation_features)
            y_pred_val = model_info['model'].predict(X_val_final)
        y_true_val = val_data[target]

        # 测试集评估
        width_model_key = f'mediator_linear_width' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_width'
        height_model_key = f'mediator_linear_height' if degree_mediator == 1 else f'mediator_poly{degree_mediator}_height'

        # 生成测试集的预测宽高
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

        # 创建测试集的5个特征：3个打印参数 + 2个预测宽高
        test_mediation_features = test_data[predictors].copy()
        test_mediation_features['predicted_width'] = test_pred_width
        test_mediation_features['predicted_height'] = test_pred_height

        # 验证特征数量是否为5
        assert test_mediation_features.shape[
                   1] == 5, f"中介模型第二层测试特征数量错误: 应为5，实际为{test_mediation_features.shape[1]}"

        # 预测并评估
        if model_info['poly'] is None:
            y_pred_test = model_info['model'].predict(test_mediation_features)
        else:
            X_test_final = model_info['poly'].transform(test_mediation_features)
            y_pred_test = model_info['model'].predict(X_test_final)
        y_true_test = test_data[target]

        # 存储验证集和测试集结果
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    # 评估直接模型
    key = 'direct_linear'
    model = models[key]
    # 验证集
    y_pred_val = model.predict(val_data[predictors])
    y_true_val = val_data[target]
    # 测试集
    y_pred_test = model.predict(test_data[predictors])
    y_true_test = test_data[target]
    results[key] = {
        'val': {
            'MSE': mean_squared_error(y_true_val, y_pred_val),
            'R2': r2_score(y_true_val, y_pred_val),
            'MAE': mean_absolute_error(y_true_val, y_pred_val),
            'MedAE': median_absolute_error(y_true_val, y_pred_val)
        },
        'test': {
            'MSE': mean_squared_error(y_true_test, y_pred_test),
            'R2': r2_score(y_true_test, y_pred_test),
            'MAE': mean_absolute_error(y_true_test, y_pred_test),
            'MedAE': median_absolute_error(y_true_test, y_pred_test)
        }
    }

    for degree in poly_degrees:
        key = f'direct_poly{degree}'
        model_info = models[key]
        # 验证集
        y_pred_val = model_info['model'].predict(model_info['val_features'])
        y_true_val = val_data[target]
        # 测试集
        X_test_poly = model_info['poly'].transform(test_data[predictors])
        y_pred_test = model_info['model'].predict(X_test_poly)
        y_true_test = test_data[target]
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    # 评估混合模型
    key = 'hybrid_linear'
    model = models[key]
    # 验证集
    y_pred_val = model.predict(val_data[hybrid_features])
    y_true_val = val_data[target]
    # 测试集
    y_pred_test = model.predict(test_data[hybrid_features])
    y_true_test = test_data[target]
    results[key] = {
        'val': {
            'MSE': mean_squared_error(y_true_val, y_pred_val),
            'R2': r2_score(y_true_val, y_pred_val),
            'MAE': mean_absolute_error(y_true_val, y_pred_val),
            'MedAE': median_absolute_error(y_true_val, y_pred_val)
        },
        'test': {
            'MSE': mean_squared_error(y_true_test, y_pred_test),
            'R2': r2_score(y_true_test, y_pred_test),
            'MAE': mean_absolute_error(y_true_test, y_pred_test),
            'MedAE': median_absolute_error(y_true_test, y_pred_test)
        }
    }

    for degree in poly_degrees:
        key = f'hybrid_poly{degree}'
        model_info = models[key]
        # 验证集
        y_pred_val = model_info['model'].predict(model_info['val_features'])
        y_true_val = val_data[target]
        # 测试集
        X_test_poly = model_info['poly'].transform(test_data[hybrid_features])
        y_pred_test = model_info['model'].predict(X_test_poly)
        y_true_test = test_data[target]
        results[key] = {
            'val': {
                'MSE': mean_squared_error(y_true_val, y_pred_val),
                'R2': r2_score(y_true_val, y_pred_val),
                'MAE': mean_absolute_error(y_true_val, y_pred_val),
                'MedAE': median_absolute_error(y_true_val, y_pred_val)
            },
            'test': {
                'MSE': mean_squared_error(y_true_test, y_pred_test),
                'R2': r2_score(y_true_test, y_pred_test),
                'MAE': mean_absolute_error(y_true_test, y_pred_test),
                'MedAE': median_absolute_error(y_true_test, y_pred_test)
            }
        }

    return results, model_names, val_data[target], test_data[target]


def calculate_averages(results_list):
    """计算多次实验结果的平均值（包含验证集和测试集）"""
    if not results_list:
        return {}

    # 初始化平均结果字典
    avg_results = {}
    model_keys = results_list[0].keys()

    # 为每个模型初始化指标列表（包含val和test）
    for model_key in model_keys:
        avg_results[model_key] = {
            'val': {
                'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []
            },
            'test': {
                'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []
            }
        }

    # 收集所有实验结果
    for result in results_list:
        for model_key in model_keys:
            for dataset_type in ['val', 'test']:
                for metric in avg_results[model_key][dataset_type]:
                    avg_results[model_key][dataset_type][metric].append(
                        result[model_key][dataset_type][metric]
                    )

    # 计算平均值
    for model_key in model_keys:
        for dataset_type in ['val', 'test']:
            for metric in avg_results[model_key][dataset_type]:
                avg_results[model_key][dataset_type][metric] = np.mean(
                    avg_results[model_key][dataset_type][metric]
                )

    return avg_results


def calculate_mediation_effect(average_results):
    """计算中介效应比例"""
    # 总效应 (直接模型的效应)
    total_effect = average_results['direct_linear']['test']['R2']

    # 直接效应 (控制中介变量后的直接效应)
    direct_effect = average_results['hybrid_linear']['test']['R2'] - average_results['mediation_model_degree1']['test'][
        'R2']

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
    print("使用基于机械模量的分层抽样进行数据集划分")
    print("中介模型结构: 第一层3个输入变量，第二层5个输入变量")
    print("=" * 50)

    # 运行多次实验
    results_list = []
    all_predictions = {}
    model_names = None
    y_true_val = None
    y_true_test = None

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 30}")
        print(f"运行实验 {i + 1}/{len(seeds)}，种子: {seed}")
        print(f"{'=' * 30}")

        results, exp_model_names, exp_y_val, exp_y_test = run_regression_experiment(seed)
        results_list.append(results)

        # 保存模型名称（所有实验相同）
        if model_names is None:
            model_names = exp_model_names
            y_true_val = exp_y_val
            y_true_test = exp_y_test

        # 输出当前种子的评估结果（包含验证集和测试集）
        print(f"\n实验 {i + 1} 各模型评估结果:")
        print('-' * 60)
        for model_key in results:
            print(f'{model_names[model_key]}:')
            print(f'  验证集:')
            print(f'    MSE: {results[model_key]["val"]["MSE"]:.4f}')
            print(f'    R2: {results[model_key]["val"]["R2"]:.4f}')
            print(f'    MAE: {results[model_key]["val"]["MAE"]:.4f}')
            print(f'    MedAE: {results[model_key]["val"]["MedAE"]:.4f}')
            print(f'  测试集:')
            print(f'    MSE: {results[model_key]["test"]["MSE"]:.4f}')
            print(f'    R2: {results[model_key]["test"]["R2"]:.4f}')
            print(f'    MAE: {results[model_key]["test"]["MAE"]:.4f}')
            print(f'    MedAE: {results[model_key]["test"]["MedAE"]:.4f}')
            print('-' * 60)

    # 计算平均结果
    print("\n" + "=" * 50)
    print("计算多次实验的平均结果...")
    avg_results = calculate_averages(results_list)

    # 输出平均评估结果（包含验证集和测试集）
    print("\n" + "=" * 50)
    print("所有模型平均评估结果（保留4位小数）:")
    print('=' * 50)
    for model_key in avg_results:
        print(f'\n{model_names[model_key]} 平均评估：')
        print(f'  验证集:')
        print(f'    均方误差 (MSE): {avg_results[model_key]["val"]["MSE"]:.4f}')
        print(f'    决定系数 (R2): {avg_results[model_key]["val"]["R2"]:.4f}')
        print(f'    平均绝对误差 (MAE): {avg_results[model_key]["val"]["MAE"]:.4f}')
        print(f'    中位数绝对误差 (MedAE): {avg_results[model_key]["val"]["MedAE"]:.4f}')
        print(f'  测试集:')
        print(f'    均方误差 (MSE): {avg_results[model_key]["test"]["MSE"]:.4f}')
        print(f'    决定系数 (R2): {avg_results[model_key]["test"]["R2"]:.4f}')
        print(f'    平均绝对误差 (MAE): {avg_results[model_key]["test"]["MAE"]:.4f}')
        print(f'    中位数绝对误差 (MedAE): {avg_results[model_key]["test"]["MedAE"]:.4f}')

    # 中介效应分析（基于线性模型的平均结果）
    print("\n" + "=" * 50)
    print("中介效应分析结果 (基于测试集平均结果):")
    print("-" * 30)
    mediation_stats = calculate_mediation_effect(avg_results)

    print(f"总效应 (直接模型平均R²): {mediation_stats['total_effect']:.4f}")
    print(f"直接效应 (控制宽高后平均): {mediation_stats['direct_effect']:.4f}")
    print(f"中介效应 (通过宽高平均): {mediation_stats['mediation_effect']:.4f}")
    print(f"中介比例 (中介效应/总效应): {mediation_stats['mediation_ratio']:.2%}")

    # 找出测试集平均R²最高的模型
    best_model_key = max(avg_results, key=lambda k: avg_results[k]['test']['R2'])
    best_model_name = model_names[best_model_key]
    best_r2 = avg_results[best_model_key]['test']['R2']

    print(f"\n测试集平均R²最高的模型: {best_model_name} (平均R²={best_r2:.4f})")

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
