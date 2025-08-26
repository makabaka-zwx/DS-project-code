from flask import Flask, render_template_string, request, jsonify
import numpy as np
import os
import joblib  # 用于加载 joblib 模型

app = Flask(__name__)

# 加载 joblib 模型
model_ratio_path = os.path.join("../../models", "ga_rf_ratio_model3.0.joblib")
model_dim_path = os.path.join("../../models", "ga_rf_dim_model3.0.joblib")

# 加载模型
try:
    model_ratio = joblib.load(model_ratio_path)
    model_dim = joblib.load(model_dim_path)
except FileNotFoundError:
    print("错误: 模型文件未找到，请确保模型文件所在文件夹models，与app.py所在文件夹，在同一目录下。")
    model_ratio = None
    model_dim = None

# 所有模型的元数据（替换为 joblib 模型）
models = {
    "model_ratio": {
        "name": "GA-RF 模型 (aspect_ratio 特征)",
        "function": model_ratio,
        "x_ranges": {
            "x1": (180, 250),  # 打印温度（°C）
            "x2": (100, 5100), # 打印速度（mm/min）
            "x3": (15, 42),    # 进给率（%）
            "x4": (0, 2)       # aspect_ratio（保持原逻辑）
        }
    },
    "model_dim": {
        "name": "GA-RF 模型 (Height+Width 特征)",
        "function": model_dim,
        "x_ranges": {
            "x1": (180, 250),  # 打印温度（°C）
            "x2": (100, 5100), # 打印速度（mm/min）
            "x3": (15, 42),    # 进给率（%）
            "x4": (0, 100),    # Height（保持原逻辑）
            "x5": (0, 100)     # Width（保持原逻辑）
        }
    }
}


def find_possible_x_values(target_y, model_key, tolerance=0.5, num_samples=1000):
    """
    根据目标y值和模型，寻找可能的x取值范围（适配 joblib 模型）

    参数:
        target_y: 目标y值
        model_key: 模型名称
        tolerance: 可接受的误差范围
        num_samples: 采样数量

    返回:
        包含可能的x值范围和示例组合的字典
    """
    model = models[model_key]
    x_ranges = model["x_ranges"]
    model_func = model["function"]

    if model_func is None:
        return {
            "found": False,
            "message": "模型未正确加载，请检查模型文件。"
        }

    # 生成随机采样点（根据模型的输入特征数量调整）
    num_features = len(x_ranges)
    samples = []
    for feature in x_ranges.values():
        samples.append(np.random.uniform(low=feature[0], high=feature[1], size=num_samples))

    # 转换为 numpy 数组并转置，以便按样本访问
    samples = np.array(samples).T

    # 计算每个采样点的y值，并筛选出符合条件的点
    valid_points = []
    for sample in samples:
        # 模型预测（根据特征数量调整）
        if num_features == 4:
            y = model_func.predict([[sample[0], sample[1], sample[2], sample[3]]])[0]
        elif num_features == 5:
            y = model_func.predict([[sample[0], sample[1], sample[2], sample[3], sample[4]]])[0]
        else:
            return {
                "found": False,
                "message": "模型特征数量不匹配，请检查配置。"
            }

        if abs(y - target_y) <= tolerance:
            valid_points.append(sample.tolist() + [y])

    if not valid_points:
        return {
            "found": False,
            "message": f"未找到符合条件的x值组合，请尝试调整目标y值或增大误差容忍度。"
        }

    # 计算每个x的取值范围
    feature_values = {}
    for i in range(num_features):
        feature_values[f"x{i + 1}"] = [point[i] for point in valid_points]

    # 提取前5个示例组合
    examples = []
    for point in valid_points[:5]:
        example = {f"x{i + 1}": round(point[i], 2) for i in range(num_features)}  # 保留2位小数
        example["y"] = round(point[-1], 2)
        examples.append(example)

    return {
        "found": True,
        "ranges": {
            f"x{i + 1}": {
                "min": round(min(feature_values[f"x{i + 1}"]), 2),  # 保留2位小数
                "max": round(max(feature_values[f"x{i + 1}"]), 2),  # 保留2位小数
                "count": len(feature_values[f"x{i + 1}"])
            } for i in range(num_features)
        },
        "examples": examples,
        "total_points": len(valid_points),
        "sampled_points": num_samples
    }


@app.route('/')
def index():
    """渲染主页面"""
    # 读取HTML模板
    with open('UI.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 传递模型信息到前端
    model_options = []
    for key, model in models.items():
        model_options.append(f'<option value="{key}">{model["name"]}</option>')

    return html_content.replace('{{ model_options }}', ''.join(model_options))


@app.route('/calculate', methods=['POST'])
def calculate():
    """处理计算请求"""
    data = request.json
    try:
        target_y = float(data.get('target_y'))
        model_key = data.get('model')
        tolerance = float(data.get('tolerance', 0.5))

        if not model_key or model_key not in models:
            return jsonify({"error": "请选择有效的模型"}), 400

        result = find_possible_x_values(target_y, model_key, tolerance)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # 确保UI.html存在
    if not os.path.exists('UI.html'):
        print("错误: UI.html文件，请确保该文件与app.py在同一目录下。")
    else:
        # 启动服务器，允许局域网内其他设备访问
        app.run(host='0.0.0.0', port=5000, debug=True)