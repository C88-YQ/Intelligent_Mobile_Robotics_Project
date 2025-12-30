from flight_environment import FlightEnvironment
import numpy as np
from path_planner import plan_flight_path
from trajectory_generator import generate_and_plot_flight_trajectory

# 初始化飞行环境
env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)

# --------------------------------------------------------------------------------------------------- #
# 路径规划：生成离散碰撞-free路径
path = plan_flight_path(env, start, goal)
path = np.array(path, dtype=np.float64)

# --------------------------------------------------------------------------------------------------- #
# 轨迹生成：生成光滑轨迹，并获取其3D坐标（用于同图对比）
smooth_trajectory = generate_and_plot_flight_trajectory(path)

# --------------------------------------------------------------------------------------------------- #
# 3D可视化：同一张图中展示障碍物、离散路径、光滑轨迹
env.plot_cylinders(path=path, trajectory=smooth_trajectory)

# --------------------------------------------------------------------------------------------------- #
