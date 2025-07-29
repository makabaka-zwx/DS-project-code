import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
base_log_file = "GA_RF_Intermediary_model_log.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始中介效应GA-RF模型分析，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载数据 - 不考虑aspect_ratio
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

print("数据准备完成:")
print(f"- 包含的特征: {list(data.columns)}")
print(f"- 打印参数(自变量): ['printing_temperature', 'feed_rate', 'printing_speed']")
print(f"- 中介变量(同时作为因变量): ['Height', 'Width']")
print(f"- 目标变量: 'Experiment_mean(MPa)'")

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

# --------------------------
# 1. 中介模型：打印参数 → 宽高 → 机械模量
# --------------------------

# 1.1 第一步：用打印参数预测宽和高（将宽和高作为因变量）
print("\n" + "=" * 50)
print("1. 训练中介变量预测模型 (打印参数 → 宽高)")
print("-" * 50)

# 预测宽度的模型
width_predictor_name = "Width_Predictor"
model_width, params_width = train_ga_rf(
    train_data[predictors], train_data['Width'],
    val_data[predictors], val_data['Width'],
    width_predictor_name
)

# 预测高度的模型
height_predictor_name = "Height_Predictor"
model_height, params_height = train_ga_rf(
    train_data[predictors], train_data['Height'],
    val_data[predictors], val_data['Height'],
    height_predictor_name
)

# 1.2 第二步：用预测的宽高预测机械模量（中介模型）
print("\n" + "=" * 50)
print("2. 训练中介模型 (预测的宽高 → 机械模量)")
print("-" * 50)

# 使用预测的宽高作为特征来训练最终模型
# 生成训练集的预测宽高
train_pred_width = model_width.predict(train_data[predictors])
train_pred_height = model_height.predict(train_data[predictors])

train_mediation_features = pd.DataFrame({
    'predicted_width': train_pred_width,
    'predicted_height': train_pred_height
})

# 生成验证集的预测宽高
val_pred_width = model_width.predict(val_data[predictors])
val_pred_height = model_height.predict(val_data[predictors])

val_mediation_features = pd.DataFrame({
    'predicted_width': val_pred_width,
    'predicted_height': val_pred_height
})

# 训练中介模型
mediation_model_name = "Mediation_Model"
model_mediation, params_mediation = train_ga_rf(
    train_mediation_features, train_data[target],
    val_mediation_features, val_data[target],
    mediation_model_name
)

# --------------------------
# 2. 直接模型：打印参数直接预测机械模量
# --------------------------
print("\n" + "=" * 50)
print("3. 训练直接模型 (打印参数 → 机械模量)")
print("-" * 50)

direct_model_name = "Direct_Model"
model_direct, params_direct = train_ga_rf(
    train_data[predictors], train_data[target],
    val_data[predictors], val_data[target],
    direct_model_name
)

# --------------------------
# 3. 混合模型：打印参数 + 实际宽高预测机械模量（作为参考）
# --------------------------
print("\n" + "=" * 50)
print("4. 训练混合模型 (打印参数 + 实际宽高 → 机械模量)")
print("-" * 50)

hybrid_features = predictors + mediators
hybrid_model_name = "Hybrid_Model"
model_hybrid, params_hybrid = train_ga_rf(
    train_data[hybrid_features], train_data[target],
    val_data[hybrid_features], val_data[target],
    hybrid_model_name
)

# 保存所有模型
model_width_path = os.path.join("models", "ga_rf_width_predictor.joblib")
model_height_path = os.path.join("models", "ga_rf_height_predictor.joblib")
model_mediation_path = os.path.join("models", "ga_rf_mediation_model.joblib")
model_direct_path = os.path.join("models", "ga_rf_direct_model.joblib")
model_hybrid_path = os.path.join("models", "ga_rf_hybrid_model.joblib")

joblib.dump(model_width, model_width_path)
joblib.dump(model_height, model_height_path)
joblib.dump(model_mediation, model_mediation_path)
joblib.dump(model_direct, model_direct_path)
joblib.dump(model_hybrid, model_hybrid_path)

print(f"\n模型已保存至:")
print(f"- 宽度预测模型: {model_width_path}")
print(f"- 高度预测模型: {model_height_path}")
print(f"- 中介模型: {model_mediation_path}")
print(f"- 直接模型: {model_direct_path}")
print(f"- 混合模型: {model_hybrid_path}")

# 在测试集上评估所有模型
print("\n" + "=" * 50)
print("5. 模型测试集评估结果")
print("-" * 50)

# 生成测试集的预测宽高（用于中介模型）
test_pred_width = model_width.predict(test_data[predictors])
test_pred_height = model_height.predict(test_data[predictors])

test_mediation_features = pd.DataFrame({
    'predicted_width': test_pred_width,
    'predicted_height': test_pred_height
})

# 各模型预测
y_pred_mediation = model_mediation.predict(test_mediation_features)
y_pred_direct = model_direct.predict(test_data[predictors])
y_pred_hybrid = model_hybrid.predict(test_data[hybrid_features])

y_true = test_data[target]

# 评估中介模型
results_mediation = {
    'MSE': mean_squared_error(y_true, y_pred_mediation),
    'R2': r2_score(y_true, y_pred_mediation),
    'MAE': mean_absolute_error(y_true, y_pred_mediation),
    'MedAE': median_absolute_error(y_true, y_pred_mediation)
}

# 评估直接模型
results_direct = {
    'MSE': mean_squared_error(y_true, y_pred_direct),
    'R2': r2_score(y_true, y_pred_direct),
    'MAE': mean_absolute_error(y_true, y_pred_direct),
    'MedAE': median_absolute_error(y_true, y_pred_direct)
}

# 评估混合模型
results_hybrid = {
    'MSE': mean_squared_error(y_true, y_pred_hybrid),
    'R2': r2_score(y_true, y_pred_hybrid),
    'MAE': mean_absolute_error(y_true, y_pred_hybrid),
    'MedAE': median_absolute_error(y_true, y_pred_hybrid)
}

# 输出评估结果
print("\n中介模型评估:")
print(f"  均方误差 (MSE): {results_mediation['MSE']:.4f}")
print(f"  决定系数 (R2): {results_mediation['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results_mediation['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results_mediation['MedAE']:.4f}")

print("\n直接模型评估:")
print(f"  均方误差 (MSE): {results_direct['MSE']:.4f}")
print(f"  决定系数 (R2): {results_direct['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results_direct['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results_direct['MedAE']:.4f}")

print("\n混合模型评估:")
print(f"  均方误差 (MSE): {results_hybrid['MSE']:.4f}")
print(f"  决定系数 (R2): {results_hybrid['R2']:.4f}")
print(f"  平均绝对误差 (MAE): {results_hybrid['MAE']:.4f}")
print(f"  中位数绝对误差 (MedAE): {results_hybrid['MedAE']:.4f}")

# 中介效应分析
print("\n" + "=" * 50)
print("6. 中介效应分析结果")
print("-" * 50)

# 提取关键模型的R²值
med_r2 = results_mediation['R2']  # 中介模型
dir_r2 = results_direct['R2']  # 直接模型
hyb_r2 = results_hybrid['R2']  # 混合模型

# 计算中介效应比例
total_effect = dir_r2
direct_effect = hyb_r2 - med_r2  # 控制中介变量后的直接效应
mediation_effect = total_effect - direct_effect  # 中介效应 = 总效应 - 直接效应
mediation_ratio = mediation_effect / total_effect if total_effect != 0 else 0

print(f"总效应 (直接模型R²): {total_effect:.4f}")
print(f"直接效应 (控制宽高后): {direct_effect:.4f}")
print(f"中介效应 (通过宽高): {mediation_effect:.4f}")
print(f"中介比例 (中介效应/总效应): {mediation_ratio:.2%}")

# 特征重要性分析
plt.figure(figsize=(18, 10))

# 1. 宽度预测模型特征重要性
plt.subplot(2, 3, 1)
feature_importance_width = model_width.feature_importances_
feature_names = predictors
plt.bar(feature_names, feature_importance_width)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Width Predictor - Feature Importance')
plt.xticks(rotation=45)

# 2. 高度预测模型特征重要性
plt.subplot(2, 3, 2)
feature_importance_height = model_height.feature_importances_
plt.bar(feature_names, feature_importance_height)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Height Predictor - Feature Importance')
plt.xticks(rotation=45)

# 3. 中介模型特征重要性
plt.subplot(2, 3, 3)
feature_importance_mediation = model_mediation.feature_importances_
mediation_feature_names = ['predicted_width', 'predicted_height']
plt.bar(mediation_feature_names, feature_importance_mediation)
plt.xlabel('Mediated Features')
plt.ylabel('Importance')
plt.title('Mediation Model - Feature Importance')
plt.xticks(rotation=45)

# 4. 直接模型特征重要性
plt.subplot(2, 3, 4)
feature_importance_direct = model_direct.feature_importances_
plt.bar(feature_names, feature_importance_direct)
plt.xlabel('Printing Parameters')
plt.ylabel('Importance')
plt.title('Direct Model - Feature Importance')
plt.xticks(rotation=45)

# 5. 混合模型特征重要性
plt.subplot(2, 3, 5)
feature_importance_hybrid = model_hybrid.feature_importances_
plt.bar(hybrid_features, feature_importance_hybrid)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Hybrid Model - Feature Importance')
plt.xticks(rotation=45)

plt.tight_layout()
importance_img_path = os.path.join("outputs", "GA_RF_Intermediary_feature_importance.png")
plt.savefig(importance_img_path, dpi=300)
plt.show()
print(f"特征重要性图已保存至: {importance_img_path}")

# 绘制预测值与真实值的散点图
plt.figure(figsize=(18, 5))

# 中介模型
plt.subplot(1, 3, 1)
plt.scatter(y_true, y_pred_mediation, alpha=0.7)
plt.xlabel('True Values (MPa)')
plt.ylabel('Predicted values (MPa)')
plt.title(f'Mediation Model - R²={results_mediation["R2"]:.4f}')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.7)

# 直接模型
plt.subplot(1, 3, 2)
plt.scatter(y_true, y_pred_direct, alpha=0.7)
plt.xlabel('True Values (MPa)')
plt.ylabel('Predicted values (MPa)')
plt.title(f'Direct Model - R²={results_direct["R2"]:.4f}')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.7)

# 混合模型
plt.subplot(1, 3, 3)
plt.scatter(y_true, y_pred_hybrid, alpha=0.7)
plt.xlabel('True Values (MPa)')
plt.ylabel('Predicted values (MPa)')
plt.title(f'Hybrid Model - R²={results_hybrid["R2"]:.4f}')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
prediction_img_path = os.path.join("outputs", "GA_RF_Intermediary_prediction_scatter.png")
plt.savefig(prediction_img_path, dpi=300)
plt.show()
print(f"预测散点图已保存至: {prediction_img_path}")

# 绘制评估指标对比图
metrics = ['MSE', 'R2', 'MAE', 'MedAE']
models = ['Mediation Model', 'Direct Model', 'Hybrid Model']
results = [results_mediation, results_direct, results_hybrid]

plt.figure(figsize=(16, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)

    values = [result[metric] for result in results]
    x_pos = np.arange(len(models))

    plt.bar(x_pos, values, color=colors, alpha=0.7)
    plt.title(f'Model {metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(x_pos, models, rotation=15)

    # 添加数值标签
    for j, v in enumerate(values):
        plt.text(j, v + 0.01, f'{v:.4f}', ha='center', rotation=45, fontsize=8)

    # 设置R2的y轴范围
    if metric == 'R2':
        plt.ylim(0, 1)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
metrics_img_path = os.path.join("outputs", "GA_RF_Intermediary_metrics_comparison.png")
plt.savefig(metrics_img_path, dpi=300)
plt.show()
print(f"评估指标对比图已保存至: {metrics_img_path}")

# 导出预测结果到Excel
output_file = get_unique_filename(os.path.join("prediction_results", "GA_RF_Intermediary_predictions.xlsx"))
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 测试集预测结果
    df_test_results = pd.DataFrame({
        'True Values': y_true,
        'Mediation Model Predictions': y_pred_mediation.round(4),
        'Direct Model Predictions': y_pred_direct.round(4),
        'Hybrid Model Predictions': y_pred_hybrid.round(4)
    })
    df_test_results.to_excel(writer, sheet_name='test_predictions', index=False)

    # 模型参数对比
    params_data = {
        '参数': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        '宽度预测模型': params_width,
        '高度预测模型': params_height,
        '中介模型': params_mediation,
        '直接模型': params_direct,
        '混合模型': params_hybrid
    }
    df_params = pd.DataFrame(params_data)
    df_params.to_excel(writer, sheet_name='model_parameters', index=False)

    # 模型评估指标汇总
    metrics_data = {
        'Metric': ['MSE', 'R2', 'MAE', 'MedAE'],
        '中介模型': [
            round(results_mediation['MSE'], 4),
            round(results_mediation['R2'], 4),
            round(results_mediation['MAE'], 4),
            round(results_mediation['MedAE'], 4)
        ],
        '直接模型': [
            round(results_direct['MSE'], 4),
            round(results_direct['R2'], 4),
            round(results_direct['MAE'], 4),
            round(results_direct['MedAE'], 4)
        ],
        '混合模型': [
            round(results_hybrid['MSE'], 4),
            round(results_hybrid['R2'], 4),
            round(results_hybrid['MAE'], 4),
            round(results_hybrid['MedAE'], 4)
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_excel(writer, sheet_name='metrics_summary', index=False)

    # 中介效应分析结果
    mediation_analysis = {
        '指标': ['总效应 (直接模型R²)', '直接效应 (控制宽高后)', '中介效应 (通过宽高)', '中介比例'],
        '值': [
            f"{total_effect:.4f}",
            f"{direct_effect:.4f}",
            f"{mediation_effect:.4f}",
            f"{mediation_ratio:.2%}"
        ]
    }
    df_mediation = pd.DataFrame(mediation_analysis)
    df_mediation.to_excel(writer, sheet_name='mediation_analysis', index=False)

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
