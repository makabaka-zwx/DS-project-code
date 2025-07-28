import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from datetime import timedelta
import os
import random
from deap import base, creator, tools, algorithms
import joblib  # 用于模型序列化
import openpyxl  # 用于Excel文件操作

seed = 2520157  # 随机种子
np.random.seed(seed)
random.seed(seed)


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


def train_ga_rf(X_train, y_train, X_val, y_val, feature_set_name):
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

    # 定义遗传算法
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化适应度
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 定义参数范围和类型
    toolbox = base.Toolbox()
    toolbox.register("n_estimators", random.randint, 50, 200)  # 决策树数量
    toolbox.register("max_depth", lambda: random.choice([None, 5, 10, 15, 20]))  # max_depth的可能取值
    toolbox.register("min_samples_split", random.randint, 2, 10)  # 最小分割样本数
    toolbox.register("min_samples_leaf", random.randint, 1, 4)  # 最小叶子节点样本数
    toolbox.register("max_features", lambda: random.choice(['sqrt', 'log2', None]))  # 最大特征数

    # 创建个体和种群
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n_estimators, toolbox.max_depth,
                      toolbox.min_samples_split, toolbox.min_samples_leaf,
                      toolbox.max_features), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 定义适应度函数
    def evalRF(individual):
        n_est, max_d, min_split, min_leaf, max_feat = individual

        # 创建创建随机森林模型
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
            if random.random() < MUTPB_INIT:  # 使用初始变异概率作为基准
                if i == 1:  # 针对max_depth的处理
                    individual[i] = random.choice([None, 5, 10, 15, 20])
                elif i == 0:  # n_estimators
                    individual[i] = random.randint(50, 300)
                elif i == 2:  # min_samples_split
                    individual[i] = random.randint(2, 10)
                elif i == 3:  # min_samples_leaf
                    individual[i] = random.randint(1, 4)
                elif i == 4:  # max_features
                    individual[i] = random.choice(['sqrt', 'log2', None])
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
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
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

    # 使用最优参数创建并训练随机森林模型
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


# 创建输出目录
os.makedirs("outputs", exist_ok=True)  # 日志与图片的保存目录
os.makedirs("models", exist_ok=True)  # 模型保存目录
os.makedirs("prediction_results", exist_ok=True)  # 预测结果保存目录

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "model_training_log—GA_RF3.0.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始GA-RF模型对比实验，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 准备两种特征集
# 特征集1: 使用aspect_ratio
data_with_ratio = data.copy()
data_with_ratio['aspect_ratio'] = data_with_ratio['Height'] / data_with_ratio['Width']
selected_columns_ratio = ['printing_temperature', 'feed_rate', 'printing_speed', 'aspect_ratio', 'Experiment_mean(MPa)']
data_with_ratio = data_with_ratio[selected_columns_ratio]

# 特征集2: 使用Height和Width
selected_columns_dim = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width',
                        'Experiment_mean(MPa)']
data_with_dim = data[selected_columns_dim]

print("数据准备完成:")
print(f"- 特征集1 (aspect_ratio): {list(data_with_ratio.columns)}")
print(f"- 特征集2 (Height+Width): {list(data_with_dim.columns)}")

# 为两种特征集分别划分训练集、验证集和测试集 (7:1.5:1.5)
# 特征集1: aspect_ratio
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

# 特征集2: Height+Width
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

print("\n数据集划分完成:")
print(f"- aspect_ratio特征集: 训练集={X_train_ratio.shape}, 验证集={X_val_ratio.shape}, 测试集={X_test_ratio.shape}")
print(f"- Height+Width特征集: 训练集={X_train_dim.shape}, 验证集={X_val_dim.shape}, 测试集={X_test_dim.shape}")

# 分别对两个特征集运行GA-RF模型
model_ratio, params_ratio = train_ga_rf(
    X_train_ratio, y_train_ratio, X_val_ratio, y_val_ratio, "aspect_ratio"
)
model_dim, params_dim = train_ga_rf(
    X_train_dim, y_train_dim, X_val_dim, y_val_dim, "Height+Width"
)

# 保存模型
model_ratio_path = os.path.join("models", "ga_rf_ratio_model3.0.joblib")
model_dim_path = os.path.join("models", "ga_rf_dim_model3.0.joblib")
joblib.dump(model_ratio, model_ratio_path)
joblib.dump(model_dim, model_dim_path)
print(f"\n模型已保存至:")
print(f"- aspect_ratio特征集模型: {model_ratio_path}")
print(f"- Height+Width特征集模型: {model_dim_path}")

# 在验证集和测试集上进行预测
# aspect_ratio特征集
y_pred_val_ratio = model_ratio.predict(X_val_ratio)
y_pred_test_ratio = model_ratio.predict(X_test_ratio)

# Height+Width特征集
y_pred_val_dim = model_dim.predict(X_val_dim)
y_pred_test_dim = model_dim.predict(X_test_dim)

# 评估两个模型
# aspect_ratio特征集评估
results_ratio = {
    'val': {
        'MSE': mean_squared_error(y_val_ratio, y_pred_val_ratio),
        'R2': r2_score(y_val_ratio, y_pred_val_ratio)
    },
    'test': {
        'MSE': mean_squared_error(y_test_ratio, y_pred_test_ratio),
        'R2': r2_score(y_test_ratio, y_pred_test_ratio),
        'MAE': mean_absolute_error(y_test_ratio, y_pred_test_ratio),
        'MedAE': median_absolute_error(y_test_ratio, y_pred_test_ratio)
    }
}

# Height+Width特征集评估
results_dim = {
    'val': {
        'MSE': mean_squared_error(y_val_dim, y_pred_val_dim),
        'R2': r2_score(y_val_dim, y_pred_val_dim)
    },
    'test': {
        'MSE': mean_squared_error(y_test_dim, y_pred_test_dim),
        'R2': r2_score(y_test_dim, y_pred_test_dim),
        'MAE': mean_absolute_error(y_test_dim, y_pred_test_dim),
        'MedAE': median_absolute_error(y_test_dim, y_pred_test_dim)
    }
}

# 输出评估结果
print('\n' + '=' * 50)
print("模型评估结果（保留4位小数）:")
print('=' * 50)

print("\n使用aspect_ratio特征的GA-RF模型:")
print("验证集评估:")
print(f"  均方误差 (MSE): {results_ratio['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {results_ratio['val']['R2']:.4f}")
print("测试集评估:")
print(f"  均方误差 (MSE): {results_ratio['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {results_ratio['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results_ratio['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results_ratio['test']['MedAE']:.4f}")

print("\n使用Height+Width特征的GA-RF模型:")
print("验证集评估:")
print(f"  均方误差 (MSE): {results_dim['val']['MSE']:.4f}")
print(f"  决定系数 (R2): {results_dim['val']['R2']:.4f}")
print("测试集评估:")
print(f"  均方误差 (MSE): {results_dim['test']['MSE']:.4f}")
print(f"  决定系数 (R2): {results_dim['test']['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results_dim['test']['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results_dim['test']['MedAE']:.4f}")

# 特征重要性分析
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
feature_importance_ratio = model_ratio.feature_importances_
feature_names_ratio = X_ratio.columns
plt.bar(feature_names_ratio, feature_importance_ratio)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('GA-RF Feature Importance (aspect_ratio)')
plt.xticks(rotation=45)

# Height+Width特征集
plt.subplot(1, 2, 2)
feature_importance_dim = model_dim.feature_importances_
feature_names_dim = X_dim.columns
plt.bar(feature_names_dim, feature_importance_dim)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('GA-RF Feature Importance (Height+Width)')
plt.xticks(rotation=45)

plt.tight_layout()
importance_img_path = os.path.join("outputs", "ga_rf_feature_importance.png")
plt.savefig(importance_img_path, dpi=300)
plt.show()
print(f"特征重要性图已保存至: {importance_img_path}")

# 绘制预测值与真实值的散点图
plt.figure(figsize=(15, 6))

# aspect_ratio特征集
plt.subplot(1, 2, 1)
plt.scatter(y_test_ratio, y_pred_test_ratio, alpha=0.7)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'GA-RF (aspect_ratio) - R²={results_ratio["test"]["R2"]:.4f}')
plt.plot([y_test_ratio.min(), y_test_ratio.max()],
         [y_test_ratio.min(), y_test_ratio.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.7)

# Height+Width特征集
plt.subplot(1, 2, 2)
plt.scatter(y_test_dim, y_pred_test_dim, alpha=0.7)
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title(f'GA-RF (Height+Width) - R²={results_dim["test"]["R2"]:.4f}')
plt.plot([y_test_dim.min(), y_test_dim.max()],
         [y_test_dim.min(), y_test_dim.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
prediction_img_path = os.path.join("outputs", "ga_rf_prediction_scatter.png")
plt.savefig(prediction_img_path, dpi=300)
plt.show()
print(f"预测散点图已保存至: {prediction_img_path}")

# 导出预测结果到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "ga_rf_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # aspect_ratio特征集的验证集结果
    df_ratio_val = pd.DataFrame({
        'True Values': y_val_ratio,
        'Predicted Values': y_pred_val_ratio.round(4),
        'Error': (y_val_ratio - y_pred_val_ratio).round(4)
    })
    df_ratio_val.to_excel(writer, sheet_name='ratio_validation', index=False)

    # aspect_ratio特征集的测试集结果
    df_ratio_test = pd.DataFrame({
        'True Values': y_test_ratio,
        'Predicted Values': y_pred_test_ratio.round(4),
        'Error': (y_test_ratio - y_pred_test_ratio).round(4)
    })
    df_ratio_test.to_excel(writer, sheet_name='ratio_test', index=False)

    # Height+Width特征集的验证集结果
    df_dim_val = pd.DataFrame({
        'True Values': y_val_dim,
        'Predicted Values': y_pred_val_dim.round(4),
        'Error': (y_val_dim - y_pred_val_dim).round(4)
    })
    df_dim_val.to_excel(writer, sheet_name='dim_validation', index=False)

    # Height+Width特征集的测试集结果
    df_dim_test = pd.DataFrame({
        'True Values': y_test_dim,
        'Predicted Values': y_pred_test_dim.round(4),
        'Error': (y_test_dim - y_pred_test_dim).round(4)
    })
    df_dim_test.to_excel(writer, sheet_name='dim_test', index=False)

    # 模型参数对比
    params_data = {
        '参数': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'aspect_ratio模型': params_ratio,
        'Height+Width模型': params_dim
    }
    df_params = pd.DataFrame(params_data)
    df_params.to_excel(writer, sheet_name='model_parameters', index=False)

    # 模型评估指标汇总
    metrics_data = {
        'Metric': ['MSE', 'R2', 'MAE', 'MedAE'],
        'aspect_ratio (validation)': [
            round(results_ratio['val']['MSE'], 4),
            round(results_ratio['val']['R2'], 4),
            None,
            None
        ],
        'aspect_ratio (test)': [
            round(results_ratio['test']['MSE'], 4),
            round(results_ratio['test']['R2'], 4),
            round(results_ratio['test']['MAE'], 4),
            round(results_ratio['test']['MedAE'], 4)
        ],
        'Height+Width (validation)': [
            round(results_dim['val']['MSE'], 4),
            round(results_dim['val']['R2'], 4),
            None,
            None
        ],
        'Height+Width (test)': [
            round(results_dim['test']['MSE'], 4),
            round(results_dim['test']['R2'], 4),
            round(results_dim['test']['MAE'], 4),
            round(results_dim['test']['MedAE'], 4)
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_excel(writer, sheet_name='metrics_summary', index=False)

print(f"\n所有预测结果已导出至: {output_file}")

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__
