from fenics import *
import matplotlib.pyplot as plt

print("开始测试FEniCS是否安装成功......")
# 测试1：创建单位正方形网格
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# 测试2：求解泊松方程
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

u = Function(V)
solve(a == L, u)

# 测试3：可视化解
plot(u)
plt.show()
print("FEniCS安装成功！")