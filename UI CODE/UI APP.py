from flask import Flask, render_template_string, request, jsonify
import numpy as np
import os

app = Flask(__name__)


# 示例模型 - 用户可以替换为自己的实际模型
# 模型1: y = x1 + 2*x2 + 3*x3
def model1(x1, x2, x3):
    return x1 + 2 * x2 + 3 * x3


# 模型2: y = x1*x2 + x3
def model2(x1, x2, x3):
    return x1 * x2 + x3


# 模型3: y = sin(x1) + cos(x2) + x3
def model3(x1, x2, x3):
    return np.sin(x1) + np.cos(x2) + x3


# 所有模型的元数据
models = {
    "model1": {
        "name": "线性模型 (y = x1 + 2*x2 + 3*x3)",
        "function": model1,
        "x_ranges": {
            "x1": (-10, 10),
            "x2": (-5, 5),
            "x3": (-2, 8)
        }
    },
    "model2": {
        "name": "非线性模型 (y = x1*x2 + x3)",
        "function": model2,
        "x_ranges": {
            "x1": (-5, 5),
            "x2": (-5, 5),
            "x3": (-10, 10)
        }
    },
    "model3": {
        "name": "三角函数模型 (y = sin(x1) + cos(x2) + x3)",
        "function": model3,
        "x_ranges": {
            "x1": (-np.pi, np.pi),
            "x2": (-np.pi, np.pi),
            "x3": (-2, 2)
        }
    }
}


def find_possible_x_values(target_y, model_key, tolerance=0.5, num_samples=1000):
    """
    根据目标y值和模型，寻找可能的x1, x2, x3取值范围

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

    # 生成随机采样点
    x1_samples = np.random.uniform(
        low=x_ranges["x1"][0],
        high=x_ranges["x1"][1],
        size=num_samples
    )
    x2_samples = np.random.uniform(
        low=x_ranges["x2"][0],
        high=x_ranges["x2"][1],
        size=num_samples
    )
    x3_samples = np.random.uniform(
        low=x_ranges["x3"][0],
        high=x_ranges["x3"][1],
        size=num_samples
    )

    # 计算每个采样点的y值，并筛选出符合条件的点
    valid_points = []
    for x1, x2, x3 in zip(x1_samples, x2_samples, x3_samples):
        y = model["function"](x1, x2, x3)
        if abs(y - target_y) <= tolerance:
            valid_points.append((x1, x2, x3, y))

    if not valid_points:
        return {
            "found": False,
            "message": f"未找到符合条件的x值组合，请尝试调整目标y值或增大误差容忍度。"
        }

    # 计算每个x的取值范围
    x1_values = [p[0] for p in valid_points]
    x2_values = [p[1] for p in valid_points]
    x3_values = [p[2] for p in valid_points]

    # 提取前5个示例组合
    examples = []
    for p in valid_points[:5]:
        examples.append({
            "x1": round(p[0], 4),
            "x2": round(p[1], 4),
            "x3": round(p[2], 4),
            "y": round(p[3], 4)
        })

    return {
        "found": True,
        "ranges": {
            "x1": {
                "min": round(min(x1_values), 4),
                "max": round(max(x1_values), 4),
                "count": len(x1_values)
            },
            "x2": {
                "min": round(min(x2_values), 4),
                "max": round(max(x2_values), 4),
                "count": len(x2_values)
            },
            "x3": {
                "min": round(min(x3_values), 4),
                "max": round(max(x3_values), 4),
                "count": len(x3_values)
            }
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
    # 确保index.html存在
    if not os.path.exists('UI.html'):
        print("错误: UI.html文件，请确保该文件与app.py在同一目录下。")
    else:
        # 启动服务器，允许局域网内其他设备访问
        app.run(host='0.0.0.0', port=5000, debug=True)
