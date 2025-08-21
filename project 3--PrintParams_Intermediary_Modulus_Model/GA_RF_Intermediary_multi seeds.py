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

    # 定义变量
    predictors = ['printing_temperature', 'feed_rate', 'printing_speed']  # 打印参数
    mediators = ['Height', 'Width']  # 中介变量
    target = 'Experiment_mean(MPa)'  # 最终目标变量

    # 数据集划分：训练集(70%)、验证集(15%)、测试集(15%)
    train_val, test = train_test_split(data, test_size=0.3, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.5, random_state=seed)

    # 1.1 训练宽高预测模型
    model_width, params_width = train_ga_rf(
        train[predictors], train['Width'],
        val[predictors], val['Width'],
        "Width_Predictor",
        seed  # 传入当前种子
    )

    model_height, params_height = train_ga_rf(
        train[predictors], train['Height'],
        val[predictors], val['Height'],
        "Height_Predictor",
        seed  # 传入当前种子
    )

    # 生成预测的宽高特征
    train_pred_width = model_width.predict(train[predictors])
    train_pred_height = model_height.predict(train[predictors])
    train_mediation_features = pd.DataFrame({
        'predicted_width': train_pred_width,
        'predicted_height': train_pred_height
    })

    val_pred_width = model_width.predict(val[predictors])
    val_pred_height = model_height.predict(val[predictors])
    val_mediation_features = pd.DataFrame({
        'predicted_width': val_pred_width,
        'predicted_height': val_pred_height
    })

    test_pred_width = model_width.predict(test[predictors])
    test_pred_height = model_height.predict(test[predictors])
    test_mediation_features = pd.DataFrame({
        'predicted_width': test_pred_width,
        'predicted_height': test_pred_height
    })

    # 1.2 训练中介模型
    model_mediation, params_mediation = train_ga_rf(
        train_mediation_features, train[target],
        val_mediation_features, val[target],
        "Mediation_Model",
        seed  # 传入当前种子
    )

    # 2. 训练直接模型
    model_direct, params_direct = train_ga_rf(
        train[predictors], train[target],
        val[predictors], val[target],
        "Direct_Model",
        seed  # 传入当前种子
    )

    # 3. 训练混合模型
    hybrid_features = predictors + mediators
    model_hybrid, params_hybrid = train_ga_rf(
        train[hybrid_features], train[target],
        val[hybrid_features], val[target],
        "Hybrid_Model",
        seed  # 传入当前种子
    )

    # 模型预测
    y_pred_val_mediation = model_mediation.predict(val_mediation_features)
    y_pred_test_mediation = model_mediation.predict(test_mediation_features)

    y_pred_val_direct = model_direct.predict(val[predictors])
    y_pred_test_direct = model_direct.predict(test[predictors])

    y_pred_val_hybrid = model_hybrid.predict(val[hybrid_features])
    y_pred_test_hybrid = model_hybrid.predict(test[hybrid_features])

    # 计算评估指标
    results = {
        'mediation': {
            'val': {
                'MSE': mean_squared_error(val[target], y_pred_val_mediation),
                'R2': r2_score(val[target], y_pred_val_mediation)
            },
            'test': {
                'MSE': mean_squared_error(test[target], y_pred_test_mediation),
                'R2': r2_score(test[target], y_pred_test_mediation),
                'MAE': mean_absolute_error(test[target], y_pred_test_mediation),
                'MedAE': median_absolute_error(test[target], y_pred_test_mediation)
            }
        },
        'direct': {
            'val': {
                'MSE': mean_squared_error(val[target], y_pred_val_direct),
                'R2': r2_score(val[target], y_pred_val_direct)
            },
            'test': {
                'MSE': mean_squared_error(test[target], y_pred_test_direct),
                'R2': r2_score(test[target], y_pred_test_direct),
                'MAE': mean_absolute_error(test[target], y_pred_test_direct),
                'MedAE': median_absolute_error(test[target], y_pred_test_direct)
            }
        },
        'hybrid': {
            'val': {
                'MSE': mean_squared_error(val[target], y_pred_val_hybrid),
                'R2': r2_score(val[target], y_pred_val_hybrid)
            },
            'test': {
                'MSE': mean_squared_error(test[target], y_pred_test_hybrid),
                'R2': r2_score(test[target], y_pred_test_hybrid),
                'MAE': mean_absolute_error(test[target], y_pred_test_hybrid),
                'MedAE': median_absolute_error(test[target], y_pred_test_hybrid)
            }
        }
    }

    # 保存预测结果和模型
    predictions = {
        'mediation': {'val': {'y_true': val[target], 'y_pred': y_pred_val_mediation},
                      'test': {'y_true': test[target], 'y_pred': y_pred_test_mediation}},
        'direct': {'val': {'y_true': val[target], 'y_pred': y_pred_val_direct},
                   'test': {'y_true': test[target], 'y_pred': y_pred_test_direct}},
        'hybrid': {'val': {'y_true': val[target], 'y_pred': y_pred_val_hybrid},
                   'test': {'y_true': test[target], 'y_pred': y_pred_test_hybrid}}
    }

    models = {
        'width': model_width,
        'height': model_height,
        'mediation': model_mediation,
        'direct': model_direct,
        'hybrid': model_hybrid,
        'features': {'predictors': predictors, 'mediators': mediators, 'hybrid': hybrid_features}
    }

    return results, predictions, models, test


# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("prediction_results", exist_ok=True)

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "GA_RF_Intermediary(multi seeds)_model_log.txt"
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
print(f"- 打印参数(自变量): ['printing_temperature', 'feed_rate', 'printing_speed']")
print(f"- 中介变量: ['Height', 'Width']")
print(f"- 目标变量: 'Experiment_mean(MPa)'")

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


# 计算所有实验的平均值（复用RF版本的计算逻辑）
def calculate_averages(results_list):
    avg_results = {
        'mediation': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        },
        'direct': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        },
        'hybrid': {
            'val': {'MSE': [], 'R2': []},
            'test': {'MSE': [], 'R2': [], 'MAE': [], 'MedAE': []}
        }
    }

    for res in results_list:
        for model_type in ['mediation', 'direct', 'hybrid']:
            for dataset_type in ['val', 'test']:
                for metric in avg_results[model_type][dataset_type]:
                    if metric in res[model_type][dataset_type]:
                        avg_results[model_type][dataset_type][metric].append(
                            res[model_type][dataset_type][metric])

    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset_type in ['val', 'test']:
            for metric in avg_results[model_type][dataset_type]:
                avg_results[model_type][dataset_type][metric] = np.mean(
                    avg_results[model_type][dataset_type][metric])

    return avg_results


average_results = calculate_averages(all_results)

# 输出平均值结果
print('\n' + '=' * 50)
print("多次实验的平均评估结果（保留4位小数）:")
print('=' * 50)

for model_type, model_name in [
    ('mediation', '中介模型 (打印参数→宽高→机械模量)'),
    ('direct', '直接模型 (打印参数直接→机械模量)'),
    ('hybrid', '混合模型 (打印参数+宽高→机械模量)')
]:
    print(f"\n{model_name}:")
    print("验证集平均评估:")
    print(f"  均方误差 (MSE): {average_results[model_type]['val']['MSE']:.4f}")
    print(f"  决定系数 (R2): {average_results[model_type]['val']['R2']:.4f}")
    print("测试集平均评估:")
    print(f"  均方误差 (MSE): {average_results[model_type]['test']['MSE']:.4f}")
    print(f"  决定系数 (R2): {average_results[model_type]['test']['R2']:.4f}")
    print(f"  平均绝对误差 (MAE): {average_results[model_type]['test']['MAE']:.4f}")
    print(f"  中位数绝对误差 (MedAE): {average_results[model_type]['test']['MedAE']:.4f}")


# 中介效应分析（复用RF版本的计算逻辑）
def calculate_mediation_effect(average_results):
    total_effect = average_results['direct']['test']['R2']
    direct_effect = average_results['hybrid']['test']['R2'] - average_results['mediation']['test']['R2']
    mediation_effect = total_effect - direct_effect
    mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'mediation_effect': mediation_effect,
        'mediation_ratio': mediation_ratio
    }


mediation_stats = calculate_mediation_effect(average_results)

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
mediation_feature_names = ['predicted_width', 'predicted_height']
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
plt.bar(final_models['features']['hybrid'], feature_importance_hybrid)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Hybrid Model - Feature Importance')
plt.xticks(rotation=45)

plt.tight_layout()
importance_img_path = os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_feature_importance.png")
plt.savefig(importance_img_path, dpi=300)
plt.show()
print(f"特征重要性图已保存至: {importance_img_path}")

# 绘制预测值与真实值的散点图
plt.figure(figsize=(18, 5))

model_types = ['mediation', 'direct', 'hybrid']
model_names = ['Mediated Model', 'Direct Model', 'Hybrid Model']

for i, (model_type, name) in enumerate(zip(model_types, model_names), 1):
    plt.subplot(1, 3, i)
    y_true = all_predictions[-1][model_type]['test']['y_true']
    y_pred = all_predictions[-1][model_type]['test']['y_pred']
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('True Values (MPa)')
    plt.ylabel('Predicted values (MPa)')
    r2 = all_results[-1][model_type]['test']['R2']
    plt.title(f'{name} - R²={r2:.4f}')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
scatter_img_path = os.path.join("outputs", "GA_RF_Intermediary(multi seeds)_predictions_scatter.png")
plt.savefig(scatter_img_path, dpi=300)
plt.show()
print(f"预测散点图已保存至: {scatter_img_path}")

# 导出所有预测结果和平均值到Excel
output_file = get_unique_filename(
    os.path.join("prediction_results", "GA_RF_Intermediary(multi seeds)_analysis_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 导出每次实验的预测结果
    for exp_idx, predictions in enumerate(all_predictions):
        for model_type, model_name in zip(model_types, model_names):
            # 验证集结果
            df_val = pd.DataFrame({
                'True Values': predictions[model_type]['val']['y_true'],
                'Predicted Values': predictions[model_type]['val']['y_pred'].round(4),
                'Error': (predictions[model_type]['val']['y_true'] - predictions[model_type]['val']['y_pred']).round(4)
            })
            df_val.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_{model_type}_val', index=False)

            # 测试集结果
            df_test = pd.DataFrame({
                'True Values': predictions[model_type]['test']['y_true'],
                'Predicted Values': predictions[model_type]['test']['y_pred'].round(4),
                'Error': (predictions[model_type]['test']['y_true'] - predictions[model_type]['test']['y_pred']).round(
                    4)
            })
            df_test.to_excel(writer, sheet_name=f'exp_{exp_idx + 1}_{model_type}_test', index=False)

    # 导出平均结果
    avg_df = pd.DataFrame()
    for model_type in ['mediation', 'direct', 'hybrid']:
        for dataset in ['val', 'test']:
            for metric, value in average_results[model_type][dataset].items():
                avg_df.loc[f'{model_type}_{dataset}', metric] = round(value, 4)
    avg_df.to_excel(writer, sheet_name='average_results')

print(f"预测结果已导出至: {output_file}")

# 保存最后一次实验的模型
if final_models:
    model_width_path = os.path.join("models", "ga_rf(multi seeds)_width_predictor_final.joblib")
    model_height_path = os.path.join("models", "ga_rf(multi seeds)_height_predictor_final.joblib")
    model_mediation_path = os.path.join("models", "ga_rf(multi seeds)_mediation_model_final.joblib")
    model_direct_path = os.path.join("models", "ga_rf(multi seeds)_direct_model_final.joblib")
    model_hybrid_path = os.path.join("models", "ga_rf(multi seeds)_hybrid_model_final.joblib")

    joblib.dump(final_models['width'], model_width_path)
    joblib.dump(final_models['height'], model_height_path)
    joblib.dump(final_models['mediation'], model_mediation_path)
    joblib.dump(final_models['direct'], model_direct_path)
    joblib.dump(final_models['hybrid'], model_hybrid_path)

    print(f"\n最终模型已保存至:")
    print(f"- 宽度预测模型: {model_width_path}")
    print(f"- 高度预测模型: {model_height_path}")
    print(f"- 中介模型: {model_mediation_path}")
    print(f"- 直接模型: {model_direct_path}")
    print(f"- 混合模型: {model_hybrid_path}")

# 计算并输出总运行时间
end_time = time.time()
total_time = timedelta(seconds=int(end_time - start_time))
print(f"\n实验完成，总运行时间: {total_time}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
