"""
In this file, you should implement your own trajectory generation class or function.
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
        self.time_nodes = self._generate_time_nodes()  # 路径点对应的时间节点
        self.trajectory_t = None  # 轨迹的时间序列
        self.trajectory_x = None  # x方向轨迹
        self.trajectory_y = None  # y方向轨迹
        self.trajectory_z = None  # z方向轨迹
    
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
        
        return self.trajectory_t, trajectory_3d  # 返回轨迹3D坐标，方便传入plot_cylinders
    
    def plot_trajectory(self, algorithm_name="Theta*"):
        """可视化轨迹时间历史"""
        if self.trajectory_t is None:
            self.generate_trajectory()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 绘制x轴轨迹
        ax1.plot(self.trajectory_t, self.trajectory_x, 'b-', linewidth=2, label='Smooth Trajectory(x)')
        ax1.scatter(self.time_nodes, self.path[:, 0], c='r', s=50, label='Waypoints(x)')
        ax1.set_ylabel('x / m')
        ax1.set_title('Trajectory Time History - {}'.format(algorithm_name))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制y轴轨迹
        ax2.plot(self.trajectory_t, self.trajectory_y, 'g-', linewidth=2, label='Smooth Trajectory(y)')
        ax2.scatter(self.time_nodes, self.path[:, 1], c='r', s=50, label='Waypoints(y)')
        ax2.set_ylabel('y / m')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 绘制z轴轨迹
        ax3.plot(self.trajectory_t, self.trajectory_z, 'm-', linewidth=2, label='Smooth Trajectory(z)')
        ax3.scatter(self.time_nodes, self.path[:, 2], c='r', s=50, label='Waypoints(z)')
        ax3.set_xlabel('t / s')
        ax3.set_ylabel('z / m')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

def generate_and_plot_flight_trajectory(path, algorithm_name="Theta*"):
    """对外暴露的轨迹生成与可视化接口，返回光滑轨迹3D坐标"""
    path_np = np.array(path, dtype=np.float64)
    if path_np.ndim != 2 or path_np.shape[1] != 3:
        raise ValueError("路径必须是N×3的numpy数组或可转换为该格式的列表")
    
    trajectory_generator = CubicPolynomialTrajectoryGenerator(path_np)
    traj_t, traj_3d = trajectory_generator.generate_trajectory()  # 提取3D轨迹坐标
    trajectory_generator.plot_trajectory(algorithm_name)
    
    return traj_3d  # 返回光滑轨迹，用于3D同图对比
