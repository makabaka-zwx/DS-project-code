import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import random
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from deap import base, creator, tools, algorithms
import joblib
from datetime import timedelta

# 定义种子范围（与RF版本保持一致：基础种子±4，共9个种子）
base_seed = 2520157
seeds = [base_seed - 4 + i for i in range(9)]
print(f"将使用以下种子进行实验: {seeds}")

# 全局变量定义 - 解决作用域问题
predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # T（温度）、V（速度）、F（进给率）
mediators = ['Height', 'Width']  # 中介变量：H（高度）、W（宽度）
target = 'Experiment_mean(MPa)'  # 最终目标变量：机械模量

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


def train_ga_rf(X_train, y_train, X_val, y_val, feature_set_name, seed):
    """使用遗传算法优化并训练随机森林模型"""
    # 遗传算法参数设置
    POPULATION_SIZE = 50  # 种群大小
    NGEN = 15  # 迭代代数
    CXPB_INIT = 0.6  # 初始交叉概率
    CXPB_FINAL = 0.95  # 最终交叉概率
    MUTPB_INIT = 0.4  # 初始变异概率
    MUTPB_FINAL = 0.05  # 最终变异概率

    print(f"\n{feature_set_name}特征集 - 遗传算法参数:")
    print(f"种群大小: {POPULATION_SIZE}")
    print(f"迭代代数: {NGEN}")
    print(f"初始交叉概率: {CXPB_INIT}")
    print(f"最终交叉概率: {CXPB_FINAL}")
    print(f"初始变异概率: {MUTPB_INIT}")
    print(f"最终变异概率: {MUTPB_FINAL}")

    # 定义遗传算法：多次重复调用时，仅在类不存在时创建
    if "FitnessMax" not in creator.__dict__:  # 检查类是否已存在
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # 定义参数范围和类型
    toolbox = base.Toolbox()
    # 使用当前种子初始化随机数生成器
    toolbox.register("random", random.Random, seed)

    # 修复：获取随机数生成器实例
    rng = toolbox.random()

    # 使用随机数生成器实例注册工具函数
    toolbox.register("n_estimators", rng.randint, 50, 200)  # 决策树数量
    toolbox.register("max_depth", lambda: rng.choice([None, 5, 10, 15, 20]))  # max_depth的可能取值
    toolbox.register("min_samples_split", rng.randint, 2, 10)  # 最小分割样本数
    toolbox.register("min_samples_leaf", rng.randint, 1, 4)  # 最小叶子节点样本数
    toolbox.register("max_features", lambda: rng.choice(['sqrt', 'log2', None]))  # 最大特征数

    # 创建个体和种群
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n_estimators, toolbox.max_depth,
                      toolbox.min_samples_split, toolbox.min_samples_leaf,
                      toolbox.max_features), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 定义适应度函数
    def evalRF(individual):
        n_est, max_d, min_split, min_leaf, max_feat = individual

        # 创建随机森林模型（使用当前种子）
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_feat,
            random_state=seed,
            n_jobs=-1
        )

        # 在训练集上训练
        rf.fit(X_train, y_train)
        # 在验证集上评估
        y_pred = rf.predict(X_val)
        # 使用负MSE作为适应度（因为要最大化）
        return -mean_squared_error(y_val, y_pred),

    # 自适应遗传算子：根据迭代代数动态调整交叉和变异概率
    def adaptive_crossover_mutation(generation):
        progress = generation / NGEN
        cxpb = CXPB_INIT + (CXPB_FINAL - CXPB_INIT) * progress
        mutpb = MUTPB_INIT - (MUTPB_INIT - MUTPB_FINAL) * progress
        return cxpb, mutpb

    # 注册遗传操作，自定义变异操作处理max_depth
    def custom_mutate(individual):
        for i in range(len(individual)):
            if rng.random() < MUTPB_INIT:  # 使用随机数生成器实例
                if i == 1:  # 针对max_depth的处理
                    individual[i] = rng.choice([None, 5, 10, 15, 20])
                elif i == 0:  # n_estimators
                    individual[i] = rng.randint(50, 300)
                elif i == 2:  # min_samples_split
                    individual[i] = rng.randint(2, 10)
                elif i == 3:  # min_samples_leaf
                    individual[i] = rng.randint(1, 4)
                elif i == 4:  # max_features
                    individual[i] = rng.choice(['sqrt', 'log2', None])
        return individual,

    toolbox.register("evaluate", evalRF)
    toolbox.register("mate", tools.cxTwoPoint)  # 两点交叉
    toolbox.register("mutate", custom_mutate)  # 自定义变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择策略

    # 开始遗传算法优化
    print(f"{feature_set_name}特征集 - 开始遗传算法优化随机森林参数...")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)  # 保存最优个体

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 运行遗传算法，手动实现自适应交叉和变异概率
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    # 开始迭代
    for gen in range(1, NGEN + 1):
        # 自适应调整交叉和变异概率
        cxpb, mutpb = adaptive_crossover_mutation(gen)

        # 选择下一代个体
        offspring = toolbox.select(pop, len(pop))
        # 克隆所选个体
        offspring = list(map(toolbox.clone, offspring))

        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < cxpb:  # 使用随机数生成器实例
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if rng.random() < mutpb:  # 使用随机数生成器实例
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估没有适应度值的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新种群
        pop[:] = offspring

        # 更新名人堂和日志
        hof.update(pop)
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    # 获取最优参数
    best_params = hof[0]
    print(f"{feature_set_name}特征集 - 最优参数: n_estimators={best_params[0]}, max_depth={best_params[1]}, "
          f"min_samples_split={best_params[2]}, min_samples_leaf={best_params[3]}, "
          f"max_features={best_params[4]}")

    # 使用最优参数创建并训练随机森林模型（使用当前种子）
    ga_rf = RandomForestRegressor(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
        min_samples_leaf=best_params[3],
        max_features=best_params[4],
        random_state=seed,
        n_jobs=-1
    )

    # 在训练集上训练最终模型
    ga_rf.fit(X_train, y_train)

    return ga_rf, best_params


def run_ga_mediation_experiment(seed, data):
    """运行单次GA-RF中介效应实验"""
    # 设置当前实验种子
    np.random.seed(seed)
    random.seed(seed)


    # 数据集划分：基于机械模量的分层抽样，确保每个分箱至少有4个样本
    min_samples_per_bin = 4
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

    # 1.1 训练宽高预测模型（中介模型第一层：3个输入变量[T, V, F]）
    model_width, params_width = train_ga_rf(
        train_data[predictors], train_data['Width'],
        val_data[predictors], val_data['Width'],
        "Width_Predictor",
        seed  # 传入当前种子
    )

    model_height, params_height = train_ga_rf(
        train_data[predictors], train_data['Height'],
        val_data[predictors], val_data['Height'],
        "Height_Predictor",
        seed  # 传入当前种子
    )

    # 生成预测的宽高特征，并与原始3个打印参数合并（中介模型第二层：5个输入变量[T, V, F, Ŵ, Ĥ]）
    # 训练集特征
    train_pred_width = model_width.predict(train_data[predictors])
    train_pred_height = model_height.predict(train_data[predictors])
    train_mediation_features = train_data[predictors].copy()
    train_mediation_features['predicted_width'] = train_pred_width  # Ŵ（预测宽）
    train_mediation_features['predicted_height'] = train_pred_height  # Ĥ（预测高）

    # 验证特征数量是否为5
    assert train_mediation_features.shape[1] == 5, \
        f"中介模型第二层训练特征数量错误: 应为5，实际为{train_mediation_features.shape[1]}"

    # 验证集特征
    val_pred_width = model_width.predict(val_data[predictors])
    val_pred_height = model_height.predict(val_data[predictors])
    val_mediation_features = val_data[predictors].copy()
    val_mediation_features['predicted_width'] = val_pred_width
    val_mediation_features['predicted_height'] = val_pred_height

    # 验证特征数量是否为5
    assert val_mediation_features.shape[1] == 5, \
        f"中介模型第二层验证特征数量错误: 应为5，实际为{val_mediation_features.shape[1]}"

    # 测试集特征
    test_pred_width = model_width.predict(test_data[predictors])
    test_pred_height = model_height.predict(test_data[predictors])
    test_mediation_features = test_data[predictors].copy()
    test_mediation_features['predicted_width'] = test_pred_width
    test_mediation_features['predicted_height'] = test_pred_height

    # 验证特征数量是否为5
    assert test_mediation_features.shape[1] == 5, \
        f"中介模型第二层测试特征数量错误: 应为5，实际为{test_mediation_features.shape[1]}"

    # 1.2 训练中介模型（嵌套模型：5个特征[T, V, F, Ŵ, Ĥ]）
    model_mediation, params_mediation = train_ga_rf(
        train_mediation_features, train_data[target],
        val_mediation_features, val_data[target],
        "Mediation_Model",
        seed  # 传入当前种子
    )

    # 2. 训练直接模型（输入参数：3个打印参数[T, V, F]）
    model_direct, params_direct = train_ga_rf(
        train_data[predictors], train_data[target],
        val_data[predictors], val_data[target],
        "Direct_Model",
        seed  # 传入当前种子
    )

    # 3. 训练混合模型（输入参数：3个打印参数+2个实际宽高[T, V, F, W_true, H_true]）
    hybrid_features = predictors + mediators
    model_hybrid, params_hybrid = train_ga_rf(
        train_data[hybrid_features], train_data[target],
        val_data[hybrid_features], val_data[target],
        "Hybrid_Model",
        seed  # 传入当前种子
    )

    # 模型预测
    y_pred_val_mediation = model_mediation.predict(val_mediation_features)
    y_pred_test_mediation = model_mediation.predict(test_mediation_features)

    y_pred_val_direct = model_direct.predict(val_data[predictors])
    y_pred_test_direct = model_direct.predict(test_data[predictors])

    y_pred_val_hybrid = model_hybrid.predict(val_data[hybrid_features])
    y_pred_test_hybrid = model_hybrid.predict(test_data[hybrid_features])

    # 计算评估指标
    results = {
        'mediation': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_mediation),
                'R2': r2_score(val_data[target], y_pred_val_mediation)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_mediation),
                'R2': r2_score(test_data[target], y_pred_test_mediation),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_mediation),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_mediation)
            }
        },
        'direct': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_direct),
                'R2': r2_score(val_data[target], y_pred_val_direct)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_direct),
                'R2': r2_score(test_data[target], y_pred_test_direct),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_direct),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_direct)
            }
        },
        'hybrid': {
            'val': {
                'MSE': mean_squared_error(val_data[target], y_pred_val_hybrid),
                'R2': r2_score(val_data[target], y_pred_val_hybrid)
            },
            'test': {
                'MSE': mean_squared_error(test_data[target], y_pred_test_hybrid),
                'R2': r2_score(test_data[target], y_pred_test_hybrid),
                'MAE': mean_absolute_error(test_data[target], y_pred_test_hybrid),
                'MedAE': median_absolute_error(test_data[target], y_pred_test_hybrid)
            }
        }
    }

    # 保存预测结果和模型
    predictions = {
        'mediation': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_mediation},
                      'test': {'y_true': test_data[target], 'y_pred': y_pred_test_mediation}},
        'direct': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_direct},
                   'test': {'y_true': test_data[target], 'y_pred': y_pred_test_direct}},
        'hybrid': {'val': {'y_true': val_data[target], 'y_pred': y_pred_val_hybrid},
                   'test': {'y_true': test_data[target], 'y_pred': y_pred_test_hybrid}}
    }

    models = {
        'width': model_width,
        'height': model_height,
        'mediation': model_mediation,
        'direct': model_direct,
        'hybrid': model_hybrid,
        'features': {
            'predictors': predictors,
            'mediators': mediators,
            'hybrid': hybrid_features,
            'mediation': list(train_mediation_features.columns)  # 明确中介模型的5个特征
        }
    }

    return results, predictions, models, test_data


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "GA_RF_Intermediary(multi seeds)_CV_model_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始中介效应GA-RF模型分析，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据准备完成:")
print(f"- 包含的特征: {list(data.columns)}")
print(f"- 打印参数(自变量): {predictors} [T（温度）、V（速度）、F（进给率）]")
print(f"- 中介变量: {mediators} [H（高度）、W（宽度）]")
print(f"- 目标变量: 'Experiment_mean(MPa)' [机械模量]")
print("\n模型结构详情:")
print("1. 直接模型: [T、V、F] → 机械模量 (快速初步预测性能)")
print("2. 中介模型(嵌套): [T、V、F] → [Ŵ、Ĥ] → 机械模量 (高精度预测+解析中介机制)")
print("3. 混合模型: [T、V、F、W_true、H_true] → 机械模量 (验证几何特征的增益潜力)")

# 存储所有实验结果
all_results = []
all_predictions = []
final_models = None
test_data = None

# 运行多次实验（与RF版本保持一致的9次实验）
for i, seed in enumerate(seeds):
    print(f"\n{'=' * 30}")
    print(f"开始第 {i + 1}/{len(seeds)} 次实验，种子值: {seed}")
    print(f"{'=' * 30}")

    results, predictions, models, test = run_ga_mediation_experiment(seed, data)
    all_results.append(results)
    all_predictions.append(predictions)

    # 保存最后一次实验的模型和测试数据用于可视化
    if i == len(seeds) - 1:
        final_models = models
        test_data = test

    # 输出本次实验的评估结果
    print(f"\n第 {i + 1} 次实验评估结果（保留4位小数）:")
    print(f"中介模型 - 测试集R²: {results['mediation']['test']['R2']:.4f}")
    print(f"直接模型 - 测试集R²: {results['direct']['test']['R2']:.4f}")
    print(f"混合模型 - 测试集R²: {results['hybrid']['test']['R2']:.4f}")


# 计算所有实验的平均值、标准差和变异系数
def calculate_stats(results_list):
    stats_results = {
        'mediation': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        },
        'direct': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        },
        'hybrid': {
            'val': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0}},
            'test': {'MSE': {'mean': 0, 'std': 0, 'cv': 0}, 'R2': {'mean': 0, 'std': 0, 'cv': 0},
                     'MAE': {'mean': 0, 'std': 0, 'cv': 0}, 'MedAE': {'mean': 0, 'std': 0, 'cv': 0}}
        }
    }

    # 收集所有种子的指标值
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in stats_results[model_type][dataset_type]:
                values = [res[model_type][dataset_type][metric] for res in results_list]
                mean_val = np.mean(values)
                std_val = np.std(values)
                # 变异系数 = 标准差 / 平均值（处理平均值为0的情况）
                cv_val = std_val / mean_val if mean_val != 0 else 0

                stats_results[model_type][dataset_type][metric]['mean'] = mean_val
                stats_results[model_type][dataset_type][metric]['std'] = std_val
                stats_results[model_type][dataset_type][metric]['cv'] = cv_val

    return stats_results


# 计算统计结果
stats_results = calculate_stats(all_results)

# 输出统计结果
print('\n' + '=' * 50)
print("多次实验的统计评估结果（保留4位小数）:")
print("格式：平均值 ± 标准差 (变异系数)")
print('=' * 50)

model_info = [
    ('mediation', '中介模型', '打印参数→宽高→机械模量', '科研优化、临床个性化定制'),
    ('direct', '直接模型', '打印参数直接→机械模量', '低精度需求、快速筛选参数'),
    ('hybrid', '混合模型', '打印参数+宽高→机械模量', '模型性能上限参照')
]

for model_type, model_name, param_desc, scenario in model_info:
    print(f"\n{model_name}:")
    print(f"  参数组合: {param_desc}")
    print(f"  适用场景: {scenario}")
    print("  验证集评估:")
    print(
        f"    均方误差 (MSE): {stats_results[model_type]['val']['MSE']['mean']:.4f} ± {stats_results[model_type]['val']['MSE']['std']:.4f} ({stats_results[model_type]['val']['MSE']['cv']:.2%})")
    print(
        f"    决定系数 (R2): {stats_results[model_type]['val']['R2']['mean']:.4f} ± {stats_results[model_type]['val']['R2']['std']:.4f} ({stats_results[model_type]['val']['R2']['cv']:.2%})")
    print("  测试集评估:")
    print(
        f"    均方误差 (MSE): {stats_results[model_type]['test']['MSE']['mean']:.4f} ± {stats_results[model_type]['test']['MSE']['std']:.4f} ({stats_results[model_type]['test']['MSE']['cv']:.2%})")
    print(
        f"    决定系数 (R2): {stats_results[model_type]['test']['R2']['mean']:.4f} ± {stats_results[model_type]['test']['R2']['std']:.4f} ({stats_results[model_type]['test']['R2']['cv']:.2%})")
    print(
        f"    平均绝对误差 (MAE): {stats_results[model_type]['test']['MAE']['mean']:.4f} ± {stats_results[model_type]['test']['MAE']['std']:.4f} ({stats_results[model_type]['test']['MAE']['cv']:.2%})")
    print(
        f"    中位数绝对误差 (MedAE): {stats_results[model_type]['test']['MedAE']['mean']:.4f} ± {stats_results[model_type]['test']['MedAE']['std']:.4f} ({stats_results[model_type]['test']['MedAE']['cv']:.2%})")


# 中介效应分析
def calculate_mediation_effect(stats_results):
    total_effect = stats_results['direct']['test']['R2']['mean']
    direct_effect = stats_results['hybrid']['test']['R2']['mean'] - stats_results['mediation']['test']['R2']['mean']
    mediation_effect = total_effect - direct_effect
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


mediation_stats = calculate_mediation_effect(stats_results)

print('\n' + '=' * 50)
print("中介效应分析结果:")
print('=' * 50)
print(f"总效应 (直接模型R²): {mediation_stats['total_effect']:.4f}")
print(f"直接效应 (控制宽高后): {mediation_stats['direct_effect']:.4f}")
print(f"中介效应 (通过宽高): {mediation_stats['mediation_effect']:.4f}")
print(f"中介比例 (中介效应/总效应): {mediation_stats['mediation_ratio']:.2%}")

# 绘制特征重要性分析
plt.figure(figsize=(18, 10))

# 1. 宽度预测模型特征重要性
plt.subplot(2, 3, 1)
feature_importance_width = final_models['width'].feature_importances_
feature_names = final_models['features']['predictors']
plt.bar(feature_names, feature_importance_width)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Width Predictor - Feature Importance')
plt.xticks(rotation=45)

# 2. 高度预测模型特征重要性
plt.subplot(2, 3, 2)
feature_importance_height = final_models['height'].feature_importances_
plt.bar(feature_names, feature_importance_height)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Height Predictor - Feature Importance')
plt.xticks(rotation=45)

# 3. 中介模型特征重要性
plt.subplot(2, 3, 3)
feature_importance_mediation = final_models['mediation'].feature_importances_
mediation_feature_names = final_models['features']['mediation']
plt.bar(mediation_feature_names, feature_importance_mediation)
plt.xlabel('Mediated Features')
plt.ylabel('Importance')
plt.title('Mediation Model - Feature Importance')
plt.xticks(rotation=45)

# 4. 直接模型特征重要性
plt.subplot(2, 3, 4)
feature_importance_direct = final_models['direct'].feature_importances_
plt.bar(feature_names, feature_importance_direct)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Direct Model - Feature Importance')
plt.xticks(rotation=45)

# 5. 混合模型特征重要性
plt.subplot(2, 3, 5)
feature_importance_hybrid = final_models['hybrid'].feature_importances_
hybrid_feature_names = final_models['features']['hybrid']
plt.bar(hybrid_feature_names, feature_importance_hybrid)
plt.xlabel('Hybrid Features')
plt.ylabel('Importance')
plt.title('Hybrid Model - Feature Importance')
plt.xticks(rotation=45)

# 调整布局并保存图像
plt.tight_layout()
feature_importance_plot_path = get_unique_filename(os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_CV_feature_importance_comparison.png"))
plt.savefig(feature_importance_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n特征重要性比较图已保存至: {feature_importance_plot_path}")

# 绘制预测值与真实值对比图
plt.figure(figsize=(18, 6))

# 中介模型
plt.subplot(1, 3, 1)
plt.scatter(test_data[target], all_predictions[-1]['mediation']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Mediation Model (R² = {stats_results["mediation"]["test"]["R2"]["mean"]:.4f})')

# 直接模型
plt.subplot(1, 3, 2)
plt.scatter(test_data[target], all_predictions[-1]['direct']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Direct Model (R² = {stats_results["direct"]["test"]["R2"]["mean"]:.4f})')

# 混合模型
plt.subplot(1, 3, 3)
plt.scatter(test_data[target], all_predictions[-1]['hybrid']['test']['y_pred'], alpha=0.6)
plt.plot([test_data[target].min(), test_data[target].max()],
         [test_data[target].min(), test_data[target].max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Hybrid Model (R² = {stats_results["hybrid"]["test"]["R2"]["mean"]:.4f})')

plt.tight_layout()
prediction_comparison_plot_path = get_unique_filename(os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_CV_prediction_vs_actual_comparison.png"))
plt.savefig(prediction_comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"预测值与真实值对比图已保存至: {prediction_comparison_plot_path}")

# 计算并输出总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f"\n{'=' * 50}")
print(f"所有实验完成!")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {str(timedelta(seconds=total_time))}")
print(f"日志文件保存至: {log_file}")
print("=" * 50)

# 保存最终模型
model_paths = {
    'width': get_unique_filename(os.path.join("models", "GA_RF_Intermediary(multi seeds)_CV_width_predictor_model.pkl")),
    'height': get_unique_filename(os.path.join("models", "GA_RF_Intermediary(multi seeds)_CV_height_predictor_model.pkl")),
    'mediation': get_unique_filename(os.path.join("models", "GA_RF_Intermediary(multi seeds)_CV_mediation_model.pkl")),
    'direct': get_unique_filename(os.path.join("models", "GA_RF_Intermediary(multi seeds)_CV_direct_model.pkl")),
    'hybrid': get_unique_filename(os.path.join("models", "GA_RF_Intermediary(multi seeds)_CV_hybrid_model.pkl"))
}

joblib.dump(final_models['width'], model_paths['width'])
joblib.dump(final_models['height'], model_paths['height'])
joblib.dump(final_models['mediation'], model_paths['mediation'])
joblib.dump(final_models['direct'], model_paths['direct'])
joblib.dump(final_models['hybrid'], model_paths['hybrid'])

print("\n模型已保存至:")
for model_name, path in model_paths.items():
    print(f"- {model_name}: {path}")

# 保存统计结果到Excel
stats_excel_path = get_unique_filename(os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_CV_model_statistics.xlsx"))
with pd.ExcelWriter(stats_excel_path, engine='openpyxl') as writer:
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            df = pd.DataFrame(stats_results[model_type][dataset_type]).transpose()
            df.to_excel(writer, sheet_name=f"{model_type}_{dataset_type}")

print(f"\n统计结果已保存至Excel: {stats_excel_path}")

# 恢复标准输出
sys.stdout = sys.stdout.terminal
