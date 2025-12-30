from flight_environment import FlightEnvironment
import numpy as np
from path_planner import plan_flight_path
from trajectory_generator import generate_and_plot_flight_trajectory

# 初始化飞行环境
env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# 调用自主实现的路径规划算法生成碰撞-free路径
path = plan_flight_path(env, start, goal)

# 确保path是N×3的numpy数组（符合要求）
path = np.array(path, dtype=np.float64)

# --------------------------------------------------------------------------------------------------- #

# 可视化3D路径
env.plot_cylinders(path)

# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

# 调用自主实现的轨迹生成算法并可视化
generate_and_plot_flight_trajectory(path)

# --------------------------------------------------------------------------------------------------- #
