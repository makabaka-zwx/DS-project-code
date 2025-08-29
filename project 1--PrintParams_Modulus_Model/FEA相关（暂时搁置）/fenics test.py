import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from ufl import (TrialFunction, TestFunction, dx, grad, dot)
import pyvista as pv

# ========================
# 1. 初始化 MPI 通信（新版必须）
# ========================
comm = MPI.COMM_WORLD

# ========================
# 2. 创建网格（新版 API）
# ========================
# 示例：创建 2D 单位正方形网格
# nx, ny 为网格分段数
nx, ny = 8, 8
domain = mesh.create_unit_square(comm, nx, ny, mesh.CellType.triangle)

# ========================
# 3. 定义函数空间（新版 API）
# ========================
# 这里使用 Lagrange 一阶元（P1）
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# ========================
# 4. 定义边值条件（可选，示例中未用，但实际问题常用）
# ========================
# 示例：给边界加 Dirichlet 条件（这里简化，实际需定义边界标记）
# 如需复杂边界，建议用 mesh 标记或 gmsh 生成带标记的网格
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: 0.0 * x[0])  # 边界值设为 0（仅示例）

# （可选）用 DirichletBCMetaClass 封装边界条件
# from dolfinx.fem import dirichletbc
# bc = dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0)))

# ========================
# 5. 定义变分问题（泊松方程）
# ========================
u = TrialFunction(V)
v = TestFunction(V)

# 右端项 f
f = fem.Constant(domain, -6.0)

# 变分形式：-Δu = f → ∇u·∇v dx = f·v dx
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# ========================
# 6. 组装并求解线性系统
# ========================
# 组装矩阵和向量
problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()

# ========================
# 7. 结果可视化（结合 pyvista）
# ========================
# 仅在 rank=0 进程上可视化（避免并行冲突）
if comm.rank == 0:
    # 提取网格和解数据
    topology, cell_types, geometry = mesh.create_vtk_mesh(domain, domain.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # 将解数据映射到网格
    grid.point_data["u"] = u_sol.x.array

    # 绘制结果
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=True)
    plotter.add_title("FEniCSx 泊松方程求解结果")
    plotter.show()

# ========================
# 8. 输出验证信息
# ========================
print("FEniCSx 测试完成！")
print(f"解的最大值: {u_sol.x.array.max():.4f}")
print(f"解的最小值: {u_sol.x.array.min():.4f}")