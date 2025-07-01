import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import meshio
import os
from joblib import load  # 用于加载RF模型


# 加载预训练的RF模型（从文件加载）
def load_rf_model(model_path="models/ga_optimized_rf_model.joblib"):
    """加载遗传算法优化的RF模型"""
    if os.path.exists(model_path):
        return load(model_path)
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")


# FEA仿真类：实现3D打印骨支架的力学分析
class BoneScaffoldFEA:
    def __init__(self, rf_model=None, material_params=None):
        """初始化FEA仿真器，可传入RF模型预测材料属性"""
        self.rf_model = rf_model
        self.material_params = material_params or {
            'E': 3.5e9,  # 弹性模量(Pa)，默认PLA值
            'nu': 0.35  # 泊松比
        }
        self.mesh = None
        self.displacement = None
        self.stress = None

    def generate_mesh_from_print_params(self, print_params):
        """根据3D打印参数生成FEA网格（简化示例）"""
        # 从RF模型预测弹性模量
        if self.rf_model:
            rf_input = np.array([
                print_params['printing_temperature'],
                print_params['feed_rate'],
                print_params['printing_speed'],
                print_params['Height'],
                print_params['Width']
            ]).reshape(1, -1)
            self.material_params['E'] = self.rf_model.predict(rf_input)[0] * 1e6  # MPa转Pa
            print(f"RF预测弹性模量: {self.material_params['E'] / 1e9:.2f} GPa")

        # 实际应用中需根据打印参数生成复杂多孔网格，此处为简化立方体
        # 示例：根据层高和填充率调整网格密度
        layer_height = print_params.get('layer_height', 0.2)  # 层高(mm)
        infill_rate = print_params.get('infill_rate', 0.4)  # 填充率

        # 计算网格密度（填充率越高，网格越密）
        mesh_resolution = int(10 + infill_rate * 10)  # 10-20的网格分辨率
        self.mesh = BoxMesh(Point(0, 0, 0), Point(10, 10, 10),  # 10x10x10mm支架
                            mesh_resolution, mesh_resolution, mesh_resolution)
        print(f"生成网格: {mesh_resolution}x{mesh_resolution}x{mesh_resolution} 单元")
        return self.mesh

    def setup_mechanical_analysis(self, load_magnitude=1e4):
        """设置力学分析：边界条件与载荷"""
        if self.mesh is None:
            raise ValueError("请先生成网格")

        # 定义函数空间（3D位移场）
        V = VectorFunctionSpace(self.mesh, 'P', 1)

        # 边界条件：底部固定
        def bottom_boundary(x, on_boundary):
            return on_boundary and near(x[2], 0)

        bc = DirichletBC(V, Constant((0, 0, 0)), bottom_boundary)

        # 定义载荷：顶部均布压力（向下）
        load = Expression(('0', '0', f'-{load_magnitude}'), degree=0)  # N/m²
        ds = Measure('ds', domain=self.mesh)
        load_form = dot(load, TestFunction(V)) * ds(3)  # 顶部面为3号面

        # 定义弹性矩阵（线弹性材料）
        E = self.material_params['E']
        nu = self.material_params['nu']
        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # 定义应变和应力
        def epsilon(u):
            return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

        def sigma(u):
            return lambda_ * tr(epsilon(u)) * Identity(3) + 2 * mu * epsilon(u)

        # 变分问题
        u = TrialFunction(V)
        v = TestFunction(V)
        F = inner(sigma(u), epsilon(v)) * dx - load_form
        a, L = lhs(F), rhs(F)

        # 求解位移场
        u = Function(V)
        solve(a == L, u, bc)
        self.displacement = u

        # 计算应力
        self.stress = sigma(u)
        return u, self.stress

    def calculate_mechanical_properties(self):
        """计算力学性能指标（弹性模量、应力分布等）"""
        if self.displacement is None:
            raise ValueError("请先求解位移场")

        # 提取顶部平均位移（z方向）
        top_disp = 0
        top_nodes = 0
        for i, x in enumerate(self.mesh.coordinates()):
            if near(x[2], 10):  # 顶部边界（z=10mm）
                top_disp += self.displacement.vector()[3 * i + 2]
                top_nodes += 1

        if top_nodes > 0:
            avg_disp = top_disp / top_nodes
            strain = avg_disp / 10  # 10mm高度的应变
            stress_magnitude = 1e4  # 施加的压力(Pa)

            # 计算弹性模量（E = stress/strain）
            calculated_E = stress_magnitude / strain
            print(f"FEA计算弹性模量: {calculated_E / 1e6:.2f} MPa")
            print(f"与RF预测值的相对误差: "
                  f"{abs(calculated_E - self.material_params['E']) / self.material_params['E'] * 100:.2f}%")

            return {
                'elastic_modulus': calculated_E,
                'strain': strain,
                'stress': stress_magnitude,
                'displacement': self.displacement
            }
        return None

    def visualize_results(self, results, save_path=None):
        """可视化位移和应力分布"""
        if results is None or 'displacement' not in results:
            raise ValueError("无结果可可视化")

        plt.figure(figsize=(12, 10))

        # 1. 位移云图
        plt.subplot(2, 1, 1)
        u = results['displacement']
        u_mag = sqrt(sum(u ** 2 for u in u))
        plot(u_mag, title='Displacement Magnitude (m)', mode='color')
        plt.colorbar(label='Displacement')

        # 2. 应力云图（von Mises应力）
        plt.subplot(2, 1, 2)
        stress = self.stress
        von_mises = sqrt(0.5 * ((stress[0, 0] - stress[1, 1]) ** 2 + (stress[1, 1] - stress[2, 2]) ** 2
                                + (stress[2, 2] - stress[0, 0]) ** 2 + 6 * (
                                            stress[0, 1] ** 2 + stress[1, 2] ** 2 + stress[0, 2] ** 2)))
        plot(von_mises, title='von Mises Stress (Pa)', mode='color')
        plt.colorbar(label='Stress')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"结果图已保存至: {save_path}")
        plt.show()


# RF+FEA集成主函数
def run_rf_fea_integration(print_params, rf_model_path="models/ga_optimized_rf_model.joblib"):
    """运行RF预测+FEA仿真的完整流程"""
    # 1. 加载RF模型
    try:
        rf_model = load_rf_model(rf_model_path)
        print("成功加载RF模型")
    except Exception as e:
        print(f"加载RF模型失败: {e}")
        rf_model = None

    # 2. 初始化FEA仿真器
    fea = BoneScaffoldFEA(rf_model=rf_model)

    # 3. 生成网格（基于打印参数和RF预测）
    fea.generate_mesh_from_print_params(print_params)

    # 4. 运行力学分析
    fea.setup_mechanical_analysis()

    # 5. 计算力学性能
    results = fea.calculate_mechanical_properties()

    # 6. 可视化结果
    fea.visualize_results(results, "outputs/fea_results.png")

    return results


# 示例调用
if __name__ == "__main__":
    # 设置3D打印参数
    sample_params = {
        'printing_temperature': 210,  # 打印温度(°C)
        'feed_rate': 100,  # 进料速率(%)
        'printing_speed': 50,  # 打印速度(mm/s)
        'Height': 10,  # 支架高度(mm)
        'Width': 10,  # 支架宽度(mm)
        'layer_height': 0.2,  # 层高(mm)
        'infill_rate': 0.4  # 填充率
    }

    # 运行RF+FEA集成
    results = run_rf_fea_integration(sample_params)

    # 输出关键结果
    if results:
        print(f"\n最终结果:")
        print(f"弹性模量: {results['elastic_modulus'] / 1e6:.2f} MPa")
        print(f"应变: {results['strain'] * 100:.4f}%")
        print(f"应力: {results['stress'] / 1e3:.2f} kPa")