# 本代码的目前作用是进行数据探索，来修改输入数据的特征数目。
# 采用的是RF-GA模型与特征：'Height', 'feed_rate', 'Width'。

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
import seaborn as sns  # 新增：用于数据可视化
from sklearn.feature_selection import mutual_info_regression  # 新增：互信息计算
from sklearn.preprocessing import StandardScaler  # 新增：数据标准化

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
NGEN = 15  # 迭代代数
CXPB_INIT = 0.6  # 初始交叉概率
CXPB_FINAL = 0.95  # 最终交叉概率
MUTPB_INIT = 0.4  # 初始变异概率
MUTPB_FINAL = 0.05  # 最终变异概率

# 特征筛选阈值设置（新增）
CORR_THRESHOLD = 0.2  # 相关系数阈值
MI_THRESHOLD = 0.1  # 互信息阈值

# 创建输出目录
os.makedirs("outputs", exist_ok=True)   # 日志与图片的保存目录
os.makedirs("models", exist_ok=True)  # 模型保存目录
os.makedirs("plots", exist_ok=True)   # 新增：图表保存目录

# 开始计时
start_time = time.time()

# 生成唯一的日志文件名
base_log_file = "model_training_log—GA_RF_with_feature_engineering.txt"
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
print(f"特征筛选 - 相关系数阈值: {CORR_THRESHOLD}")
print(f"特征筛选 - 互信息阈值: {MI_THRESHOLD}")
print("=" * 50)

# 加载数据
data = pd.read_csv('./data_FEA_ANN_FEA-ANN.csv')

# 筛选保留指定列
selected_columns = [
    'printing_temperature', 'feed_rate', 'printing_speed',
    'Height', 'Width', 'Experiment_mean(MPa)', 'Experiment_std(MPa)'
]
data = data[selected_columns]

# 数据探索与特征工程（新增）
print("开始数据探索与特征工程...")

# 1. 查看数据集行数和列数
rows, columns = data.shape

# 2. 描述性统计
print("\n数据描述性统计:")
print(data.describe())

# 3. 绘制特征分布图（新增）
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns):
    plt.subplot(3, 4, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
feat_dist_img = os.path.join("plots", "feature_distributions.png")
plt.savefig(feat_dist_img, dpi=300)
print(f"特征分布图已保存至: {feat_dist_img}")

# 4. 计算相关系数矩阵（新增）
correlation = data.corr()
print("\n特征与目标变量相关系数:")
target_corr = correlation['Experiment_mean(MPa)'].sort_values(ascending=False)
print(target_corr)

# 5. 绘制相关系数热图（新增）
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
corr_heatmap = os.path.join("plots", "correlation_heatmap.png")
plt.savefig(corr_heatmap, dpi=300)
print(f"相关系数热图已保存至: {corr_heatmap}")

# 6. 计算互信息（新增）
X = data.drop('Experiment_mean(MPa)', axis=1)
y = data['Experiment_mean(MPa)']
mutual_info = mutual_info_regression(X, y)
mutual_info = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)
print("\n特征与目标变量互信息:")
print(mutual_info)

# 7. 特征重要性可视化（新增）
plt.figure(figsize=(10, 6))
plt.barh(mutual_info.index, mutual_info)
plt.xlabel('Mutual Information Value')
plt.title('Feature Importance by Mutual Information')
plt.grid(True, alpha=0.3)
mi_importance = os.path.join("plots", "mutual_info_importance.png")
plt.savefig(mi_importance, dpi=300)
print(f"互信息重要性图已保存至: {mi_importance}")

# 8. 特征筛选（新增）
# 方法1：基于相关系数筛选
corr_filtered_features = target_corr[abs(target_corr) > CORR_THRESHOLD].index.tolist()
# 排除目标变量自身
if 'Experiment_mean(MPa)' in corr_filtered_features:
    corr_filtered_features.remove('Experiment_mean(MPa)')
print(f"\n相关系数筛选后保留特征: {corr_filtered_features}")

# 方法2：基于互信息筛选
mi_filtered_features = mutual_info[mutual_info > MI_THRESHOLD].index.tolist()
print(f"互信息筛选后保留特征: {mi_filtered_features}")

# 综合筛选（取交集）
final_features = list(set(corr_filtered_features) & set(mi_filtered_features))
print(f"综合筛选后保留特征: {final_features}")

if not final_features:
    print("警告：没有特征满足筛选阈值，使用所有特征进行建模")
    final_features = X.columns.tolist()

# 9. 分析筛选后特征的分布（新增）
plt.figure(figsize=(15, 10))
for i, col in enumerate(final_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=data[col])
    plt.title(f'{col} Boxplot')
plt.tight_layout()
filtered_boxplot = os.path.join("plots", "filtered_features_boxplot.png")
plt.savefig(filtered_boxplot, dpi=300)
print(f"筛选后特征箱线图已保存至: {filtered_boxplot}")

# 10. 检查缺失值（新增）
missing_values = data[final_features].isnull().sum()
print("\n缺失值统计:")
print(missing_values)

# 准备特征和目标变量
X = data[final_features].drop('Experiment_std(MPa)', axis=1)
y = data['Experiment_mean(MPa)']
feature_names = X.columns  # 保存特征名称

# 11. 数据标准化（新增）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=feature_names)

# 12. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 13. 可视化训练集和测试集分布（新增）
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True)
plt.title('Target Variable Distribution - Training Set')
plt.subplot(1, 2, 2)
sns.histplot(y_test, kde=True)
plt.title('Target Variable Distribution - Testing Set')
plt.tight_layout()
train_test_dist = os.path.join("plots", "train_test_distribution.png")
plt.savefig(train_test_dist, dpi=300)
print(f"训练集和测试集分布已保存至: {train_test_dist}")

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

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
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
ga_rf.fit(X_train, y_train)
print("遗传算法优化的随机森林模型训练完成！")

# 保存模型到文件
model_filename = "ga_optimized_rf_model_with_feature_engineering.joblib"
model_path = os.path.join("models", model_filename)
joblib.dump(ga_rf, model_path)
print(f"模型已保存至: {model_path}")

# 使用遗传算法优化的随机森林进行预测
y_pred_ga_rf = ga_rf.predict(X_test)

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

# 绘制特征重要性图（新增）
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('随机森林特征重要性')
plt.grid(True, alpha=0.3)
rf_importance = os.path.join("plots", "rf_feature_importance.png")
plt.savefig(rf_importance, dpi=300)
print(f"随机森林特征重要性图已保存至: {rf_importance}")

# 14. 绘制预测值与真实值的残差图（新增）
residuals = y_test - y_pred_ga_rf
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_ga_rf, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Value')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# 15. 绘制残差分布（新增）
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
residuals_plot = os.path.join("plots", "residuals_plot.png")
plt.savefig(residuals_plot, dpi=300)
print(f"残差图已保存至: {residuals_plot}")

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
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('基于特征工程和GA优化的RF模型预测结果', fontsize=14)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加R²和MAE信息到图中
plt.text(y_test.min() + 0.05 * (y_test.max() - y_test.min()),
         y_test.max() - 0.1 * (y_test.max() - y_test.min()),
         f'R² = {r2_ga_rf:.4f}\nMAE = {mae_ga_rf:.4f}',
         bbox=dict(facecolor='white', alpha=0.8))

# 保存散点图到outputs文件夹
base_img_name = "GA_RF_Prediction_with_FE"
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
    # 加载模型
    loaded_model = load_saved_rf_model()

    # 准备新数据（打印参数）进行预测
    new_prediction_data = pd.DataFrame({
        'printing_temperature': [210, 190],
        'feed_rate': [100, 80],
        'printing_speed': [50, 70],
        'Height': [10, 12],
        'Width': [8, 9]
    })

    # 对新数据进行标准化处理
    new_data_scaled = scaler.transform(new_prediction_data[final_features])
    new_data_df = pd.DataFrame(new_data_scaled, columns=final_features)

    # 预测弹性模量
    predictions = loaded_model.predict(new_data_df)
    print("\n新参数组合的弹性模量预测结果:")
    for i, params in new_prediction_data.iterrows():
        print(f"参数组合 {i + 1}:")
        print(
            f"  温度: {params['printing_temperature']}°C, 进料速率: {params['feed_rate']}%, 打印速度: {params['printing_speed']}mm/s")
        print(f"  预测弹性模量: {predictions[i]:.2f} MPa")