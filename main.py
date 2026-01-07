from flight_environment import FlightEnvironment
import numpy as np
from path_planner import plan_flight_path
from trajectory_generator import generate_and_plot_flight_trajectory

# --------------------------------------------------------------------------------------------------- #
# Initialize Flight Environment
start = (1, 2, 0)
goal = (18, 18, 3)
env = FlightEnvironment(
    obs_num=50,
    start=start,
    goal=goal,
    safe_radius=0.4
)

# --------------------------------------------------------------------------------------------------- #
# Path Planning：使用Theta*和A*算法规划离散路径
path_thetastar, path_astar = plan_flight_path(env, start, goal)
path_thetastar = np.array(path_thetastar, dtype=np.float64)
path_astar = np.array(path_astar, dtype=np.float64)

# --------------------------------------------------------------------------------------------------- #
# Trajectory Generation：生成并可视化光滑轨迹
smooth_trajectory_thetastar = generate_and_plot_flight_trajectory(path_thetastar, "Theta*", max_segment_dist=3.5)
smooth_trajectory_astar = generate_and_plot_flight_trajectory(path_astar, "A*", max_segment_dist=3.5)

# --------------------------------------------------------------------------------------------------- #
# Visualization：显示障碍物、离散路径、光滑轨迹
env.plot_cylinders(
    path_thetastar=path_thetastar,
    path_astar=path_astar,
    trajectory_thetastar=smooth_trajectory_thetastar,
    trajectory_astar=smooth_trajectory_astar
)

# --------------------------------------------------------------------------------------------------- #
