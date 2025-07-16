# 本模型探索了标准化之后的RF模型的效果
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 新增：数据预处理工具
import seaborn as sns


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


# 遗传算法参数设置
POPULATION_SIZE = 50  # 种群大小
NGEN = 10  # 迭代代数
CXPB_INIT = 0.6  # 初始交叉概率
CXPB_FINAL = 0.95  # 最终交叉概率
MUTPB_INIT = 0.4  # 初始变异概率
MUTPB_FINAL = 0.05  # 最终变异概率

# 数据预处理配置（可修改）
PREPROCESSING_METHOD = 'minmax'  # 可选: 'minmax', 'standard', 'none'

# 创建输出目录
os.makedirs("outputs", exist_ok=True)  # 日志与图片的保存目录
os.makedirs("models", exist_ok=True)  # 模型保存目录
os.makedirs("data explore", exist_ok=True)  # 图表保存目录

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
preprocessing_tag = {
    'minmax': 'with_minmax',
    'standard': 'with_standard',
    'none': 'without_preprocessing'
}[PREPROCESSING_METHOD]

base_log_file = f"model_training_log—GA_RF_{preprocessing_tag}.txt"
log_file = get_unique_filename(os.path.join("outputs", base_log_file))

# 重定向输出流
sys.stdout = Logger(log_file)

print(f"开始模型训练，日志将保存到 {log_file}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
print("遗传算法参数:")
print(f"种群大小: {POPULATION_SIZE}")
print(f"迭代代数: {NGEN}")
print(f"初始交叉概率: {CXPB_INIT}")
print(f"最终交叉概率: {CXPB_FINAL}")
print(f"初始变异概率: {MUTPB_INIT}")
print(f"最终变异概率: {MUTPB_FINAL}")
print(f"数据预处理方法: {PREPROCESSING_METHOD}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 选择需要的列
selected_columns = ['printing_temperature', 'feed_rate', 'printing_speed', 'Height', 'Width', 'Experiment_mean(MPa)']
data = data[selected_columns]

# 准备特征和目标变量
X = data.drop('Experiment_mean(MPa)', axis=1)
y = data['Experiment_mean(MPa)']
feature_names = X.columns  # 保存特征名称

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


# 数据预处理函数
def preprocess_data(X_train, X_test, method='minmax'):
    """
    对数据进行预处理（标准化或归一化）

    参数:
    - X_train: 训练集特征
    - X_test: 测试集特征
    - method: 预处理方法，可选 'minmax'（Min-Max归一化）, 'standard'（Z-score标准化）, 'none'（不处理）

    返回:
    - X_train_processed: 处理后的训练集
    - X_test_processed: 处理后的测试集
    - processor: 预处理对象（用于后续预测）
    """
    if method == 'minmax':
        processor = MinMaxScaler(feature_range=(0, 1))
        X_train_processed = processor.fit_transform(X_train)
        X_test_processed = processor.transform(X_test)
        print("数据已使用Min-Max归一化处理")

    elif method == 'standard':
        processor = StandardScaler()
        X_train_processed = processor.fit_transform(X_train)
        X_test_processed = processor.transform(X_test)
        print("数据已使用Z-score标准化处理")

    else:  # 'none'
        processor = None
        X_train_processed = X_train.values
        X_test_processed = X_test.values
        print("数据未进行预处理")

    # 转换回DataFrame格式
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    # 可视化预处理前后的数据分布
    visualize_data_distribution(X_train, X_train_processed, method)

    return X_train_processed, X_test_processed, processor


# 可视化数据分布
def visualize_data_distribution(X_before, X_after, method):
    """可视化预处理前后的数据分布"""
    plt.figure(figsize=(15, 10))

    for i, col in enumerate(feature_names):
        plt.subplot(2, len(feature_names), i + 1)
        sns.histplot(X_before[col], kde=True)
        plt.title(f'Before: {col}')

        plt.subplot(2, len(feature_names), i + 1 + len(feature_names))
        sns.histplot(X_after[col], kde=True)
        plt.title(f'After: {col}')

    plt.tight_layout()
    distribution_img = os.path.join("data explore", f"data_distribution_{method}.png")
    plt.savefig(distribution_img, dpi=300)
    print(f"数据分布对比图已保存至: {distribution_img}")


# 执行数据预处理
X_train_processed, X_test_processed, processor = preprocess_data(X_train, X_test, PREPROCESSING_METHOD)

# 保存预处理对象
if processor is not None:
    processor_filename = f"{PREPROCESSING_METHOD}_processor.joblib"
    processor_path = os.path.join("models", processor_filename)
    joblib.dump(processor, processor_path)
    print(f"预处理对象已保存至: {processor_path}")

# 定义遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化适应度
creator.create("Individual", list, fitness=creator.FitnessMax)

# 定义参数范围和类型，对max_depth单独处理
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

    # 创建随机森林模型
    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        random_state=seed,
        n_jobs=-1
    )

    # 使用交叉验证评估模型
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in kf.split(X_train_processed):
        X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_val)
        # 使用负MSE作为适应度（因为要最大化）
        scores.append(-mean_squared_error(y_val, y_pred))

    return np.mean(scores),


# 自适应遗传算子：根据迭代代数动态调整交叉和变异概率
def adaptive_crossover_mutation(generation):
    """
    自适应调整交叉和变异概率：
    - 线性从初始值过渡到最终值
    """
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
print("开始遗传算法优化随机森林参数...")
pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)  # 保存最优个体

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# 运行遗传算法
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB_INIT, mutpb=MUTPB_INIT, ngen=NGEN,
                               stats=stats, halloffame=hof, verbose=True)

# 获取最优参数
best_params = hof[0]
print(f"最优参数: n_estimators={best_params[0]}, max_depth={best_params[1]}, "
      f"min_samples_split={best_params[2]}, min_samples_leaf={best_params[3]}, "
      f"max_features={best_params[4]}")

# 使用最优参数创建随机森林模型
ga_rf = RandomForestRegressor(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    min_samples_split=best_params[2],
    min_samples_leaf=best_params[3],
    max_features=best_params[4],
    random_state=seed,
    n_jobs=-1
)

# 训练模型
print("开始训练基于遗传算法优化的随机森林模型...")
ga_rf.fit(X_train_processed, y_train)
print("遗传算法优化的随机森林模型训练完成！")

# 保存模型到文件
model_filename = f"ga_optimized_rf_model_{preprocessing_tag}.joblib"
model_path = os.path.join("models", model_filename)
joblib.dump(ga_rf, model_path)
print(f"模型已保存至: {model_path}")

# 使用遗传算法优化的随机森林进行预测
y_pred_ga_rf = ga_rf.predict(X_test_processed)

# 评估遗传算法优化的随机森林模型
mse_ga_rf = mean_squared_error(y_test, y_pred_ga_rf)
r2_ga_rf = r2_score(y_test, y_pred_ga_rf)
mae_ga_rf = mean_absolute_error(y_test, y_pred_ga_rf)
medae_ga_rf = median_absolute_error(y_test, y_pred_ga_rf)

print('\n遗传算法优化的随机森林模型评估：')
print(f'均方误差 (MSE): {mse_ga_rf:.4f}')
print(f'决定系数 (R2): {r2_ga_rf:.4f}')
print(f'平均绝对误差 (MAE): {mae_ga_rf:.4f}')
print(f'中位数绝对误差 (MedAE): {medae_ga_rf:.4f}')

# 计算特征重要性
importances = ga_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n特征重要性排序:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# 计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 50)
print(f"模型训练完成！总运行时间: {timedelta(seconds=int(total_time))}")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"详细日志已保存到: {log_file}")

# 恢复标准输出
sys.stdout = sys.__stdout__

# 绘制遗传算法优化的随机森林模型预测值与真实值的散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_ga_rf, alpha=0.7)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted values', fontsize=12)
plt.title(f'GA Optimized RF Prediction ({preprocessing_tag})', fontsize=14)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加R²和MAE信息到图中
plt.text(y_test.min() + 0.05 * (y_test.max() - y_test.min()),
         y_test.max() - 0.1 * (y_test.max() - y_test.min()),
         f'R² = {r2_ga_rf:.4f}\nMAE = {mae_ga_rf:.4f}',
         bbox=dict(facecolor='white', alpha=0.8))

# 保存散点图到outputs文件夹
base_img_name = f"GA_RF_Prediction_{preprocessing_tag}"
img_extension = ".png"
base_img_path = os.path.join("outputs", base_img_name)
counter = 1.0

while True:
    img_filename = f"{base_img_path}{counter}{img_extension}"
    if not os.path.exists(img_filename):
        plt.savefig(img_filename, dpi=300, bbox_inches='tight')
        print(f"遗传算法优化随机森林散点图已保存至: {img_filename}")
        break
    counter += 0.1

plt.show()


# 新增：模型加载函数（可在其他脚本中调用）
def load_saved_rf_model():
    """加载保存的RF模型，用于后续预测"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"模型文件不存在，请先训练模型并保存。路径: {model_path}")


# 示例：在其他脚本中调用模型（新增）
if __name__ == "__main__":
    # 加载模型和预处理对象
    loaded_model = load_saved_rf_model()

    if processor is not None:
        loaded_processor = joblib.load(processor_path)
    else:
        loaded_processor = None

    # 准备新数据（打印参数）进行预测
    new_prediction_data = pd.DataFrame({
        'printing_temperature': [210, 190],
        'feed_rate': [100, 80],
        'printing_speed': [50, 70],
        'Height': [10, 12],
        'Width': [8, 9]
    })

    # 对新数据进行预处理
    if loaded_processor is not None:
        new_data_processed = loaded_processor.transform(new_prediction_data)
    else:
        new_data_processed = new_prediction_data.values

    new_data_df = pd.DataFrame(new_data_processed, columns=feature_names)

    # 预测弹性模量
    predictions = loaded_model.predict(new_data_df)
    print("\n新参数组合的弹性模量预测结果:")
    for i, params in new_prediction_data.iterrows():
        print(f"参数组合 {i + 1}:")
        print(
            f"  温度: {params['printing_temperature']}°C, 进料速率: {params['feed_rate']}%, 打印速度: {params['printing_speed']}mm/s")
        print(f"  预测弹性模量: {predictions[i]:.2f} MPa")