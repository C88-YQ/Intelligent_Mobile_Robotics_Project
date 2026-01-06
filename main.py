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
path_thetastar, path_astar = plan_flight_path(env, start, goal)
path_thetastar = np.array(path_thetastar, dtype=np.float64)
path_astar = np.array(path_astar, dtype=np.float64)

# --------------------------------------------------------------------------------------------------- #
# 轨迹生成：生成光滑轨迹，并获取其3D坐标（用于同图对比）
smooth_trajectory_thetastar = generate_and_plot_flight_trajectory(path_thetastar, "Theta*")
smooth_trajectory_astar = generate_and_plot_flight_trajectory(path_astar, "A*")

# --------------------------------------------------------------------------------------------------- #
# 3D可视化：同一张图中展示障碍物、离散路径、光滑轨迹
env.plot_cylinders(
    path_thetastar=path_thetastar,
    path_astar=path_astar,
    trajectory_thetastar=smooth_trajectory_thetastar,
    trajectory_astar=smooth_trajectory_astar
)

# --------------------------------------------------------------------------------------------------- #
