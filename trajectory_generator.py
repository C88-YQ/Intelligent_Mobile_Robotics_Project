"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class CubicPolynomialTrajectoryGenerator:
    """自主实现的三次多项式插值轨迹生成器"""
    def __init__(self, path, total_time=None):
        self.path = path  # 离散路径点（N×3 numpy数组）
        self.n_points = path.shape[0]  # 路径点数量
        self.total_time = total_time if total_time is not None else self.n_points * 2.0  # 总轨迹时间
        self.time_nodes = self._generate_time_nodes()
        self.trajectory_t = None
        self.trajectory_x = None
        self.trajectory_y = None
        self.trajectory_z = None
    
    def _generate_time_nodes(self):
        """生成每个路径点对应的时间节点（按路径点距离分配时间）"""
        distances = [0.0]
        for i in range(1, self.n_points):
            dist = np.linalg.norm(self.path[i] - self.path[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance < 1e-6:
            return np.linspace(0, self.total_time, self.n_points)
        time_nodes = np.array(distances) / total_distance * self.total_time
        
        return time_nodes
    
    def _cubic_polynomial_interpolation(self, x_nodes, t_nodes, t_trajectory):
        """三次多项式插值（生成单轴光滑轨迹）"""
        cubic_spline = interpolate.CubicSpline(t_nodes, x_nodes, bc_type='natural')
        return cubic_spline(t_trajectory)
    
    def generate_trajectory(self, time_resolution=0.01):
        """生成3D光滑轨迹，返回（时间序列，M×3轨迹坐标数组）"""
        # 生成密集时间序列
        self.trajectory_t = np.arange(0, self.total_time + time_resolution, time_resolution)
        
        # 对x、y、z三轴分别进行三次多项式插值
        self.trajectory_x = self._cubic_polynomial_interpolation(
            self.path[:, 0], self.time_nodes, self.trajectory_t
        )
        self.trajectory_y = self._cubic_polynomial_interpolation(
            self.path[:, 1], self.time_nodes, self.trajectory_t
        )
        self.trajectory_z = self._cubic_polynomial_interpolation(
            self.path[:, 2], self.time_nodes, self.trajectory_t
        )
        
        # 构造M×3的光滑轨迹坐标数组（用于3D同图绘制）
        trajectory_3d = np.vstack((self.trajectory_x, self.trajectory_y, self.trajectory_z)).T
        
        return self.trajectory_t, trajectory_3d
    
    def plot_trajectory(self, algorithm_name="Theta*"):
        """可视化轨迹时间历史"""
        if self.trajectory_t is None:
            self.generate_trajectory()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        ax1.plot(self.trajectory_t, self.trajectory_x, 'b-', linewidth=2, label='Smooth Trajectory(x)')
        ax1.scatter(self.time_nodes, self.path[:, 0], c='r', s=50, label='Waypoints(x)')
        ax1.set_ylabel('x / m')
        ax1.set_title('Trajectory Time History - {}'.format(algorithm_name))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(self.trajectory_t, self.trajectory_y, 'g-', linewidth=2, label='Smooth Trajectory(y)')
        ax2.scatter(self.time_nodes, self.path[:, 1], c='r', s=50, label='Waypoints(y)')
        ax2.set_ylabel('y / m')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(self.trajectory_t, self.trajectory_z, 'm-', linewidth=2, label='Smooth Trajectory(z)')
        ax3.scatter(self.time_nodes, self.path[:, 2], c='r', s=50, label='Waypoints(z)')
        ax3.set_xlabel('t / s')
        ax3.set_ylabel('z / m')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

def generate_and_plot_flight_trajectory(path, algorithm_name="Theta*"):
    path_np = np.array(path, dtype=np.float64)
    if path_np.ndim != 2 or path_np.shape[1] != 3:
        raise ValueError("Path must be a Nx3 numpy array.")
    
    trajectory_generator = CubicPolynomialTrajectoryGenerator(path_np)
    traj_t, traj_3d = trajectory_generator.generate_trajectory()
    trajectory_generator.plot_trajectory(algorithm_name)
    
    return traj_3d
