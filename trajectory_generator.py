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
    def __init__(self, path, total_time=None, max_segment_dist=2.5):
        self.original_time_nodes = None  # 原始路径点对应的时间节点
        self.trajectory_t = None
        self.trajectory_x = None
        self.trajectory_y = None
        self.trajectory_z = None
        self.original_path = path  # 原始离散路径点（N×3 numpy数组）
        self.max_segment_dist = max_segment_dist  # 最大允许的路径段距离，超过则插入中间点
        self.n_original_points = path.shape[0]  # 原始路径点数量
        self.enhanced_path = self._enhance_path_with_midpoints()  # 增强后的路径（含中间点）
        self.n_enhanced_points = self.enhanced_path.shape[0]  # 增强后路径点数量
        self.total_time = total_time if total_time is not None else self.n_original_points * 2.0  # 总轨迹时间
        self.time_nodes = self._generate_time_nodes()  # 增强路径对应的时间节点
    
    def _enhance_path_with_midpoints(self):
        """在间距超过阈值的路径点之间插入均匀分布的中间点"""
        enhanced_path = [self.original_path[0]]
        
        # 遍历所有相邻原始路径点
        for i in range(1, self.n_original_points):
            p_prev = self.original_path[i-1]
            p_curr = self.original_path[i]
            segment_dist = np.linalg.norm(p_curr - p_prev)

            # 如果当前段距离超过阈值，插入中间点
            if segment_dist > self.max_segment_dist:
                # 计算需要插入的中间点数量
                n_midpoints = int(np.ceil(segment_dist / self.max_segment_dist) - 1)
                # 生成均匀分布的中间点
                for j in range(1, n_midpoints + 1):
                    ratio = j / (n_midpoints + 1)
                    mid_point = p_prev + ratio * (p_curr - p_prev)
                    enhanced_path.append(mid_point)
            
            # 添加当前原始路径点
            enhanced_path.append(p_curr)
        
        return np.array(enhanced_path)
    
    def _generate_time_nodes(self):
        """生成每个路径点对应的时间节点（按路径点距离分配时间）"""
        # 计算增强路径的累积距离
        distances = [0.0]
        for i in range(1, self.n_enhanced_points):
            dist = np.linalg.norm(self.enhanced_path[i] - self.enhanced_path[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance < 1e-6:
            time_nodes = np.linspace(0, self.total_time, self.n_enhanced_points)
        else:
            time_nodes = np.array(distances) / total_distance * self.total_time
        
        # 记录原始路径点对应的时间节点（用于绘图对比）
        self._set_original_time_nodes(distances)
        
        return time_nodes
    
    def _set_original_time_nodes(self, enhanced_distances):
        """计算原始路径点在增强路径时间轴上对应的时间节点"""
        self.original_time_nodes = []
        current_idx = 0
        # 遍历原始路径点，找到其在增强路径中的位置
        for orig_point in self.original_path:
            # 找到增强路径中匹配原始点的索引
            for i in range(current_idx, len(self.enhanced_path)):
                if np.allclose(self.enhanced_path[i], orig_point, atol=1e-6):
                    self.original_time_nodes.append(enhanced_distances[i] / enhanced_distances[-1] * self.total_time)
                    current_idx = i + 1
                    break
        
        self.original_time_nodes = np.array(self.original_time_nodes)
    
    def _cubic_polynomial_interpolation(self, x_nodes, t_nodes, t_trajectory):
        """三次多项式插值（生成单轴光滑轨迹）"""
        cubic_spline = interpolate.CubicSpline(t_nodes, x_nodes, bc_type='clamped')
        return cubic_spline(t_trajectory)
    
    def generate_trajectory(self, time_resolution=0.01):
        """生成3D光滑轨迹，返回（时间序列，M×3轨迹坐标数组）"""
        # 生成密集时间序列
        self.trajectory_t = np.arange(0, self.total_time + time_resolution, time_resolution)
        
        # 对x、y、z三轴分别进行三次多项式插值（基于增强路径）
        self.trajectory_x = self._cubic_polynomial_interpolation(
            self.enhanced_path[:, 0], self.time_nodes, self.trajectory_t
        )
        self.trajectory_y = self._cubic_polynomial_interpolation(
            self.enhanced_path[:, 1], self.time_nodes, self.trajectory_t
        )
        self.trajectory_z = self._cubic_polynomial_interpolation(
            self.enhanced_path[:, 2], self.time_nodes, self.trajectory_t
        )
        
        # 构造M×3的光滑轨迹坐标数组（用于3D同图绘制）
        trajectory_3d = np.vstack((self.trajectory_x, self.trajectory_y, self.trajectory_z)).T
        
        return self.trajectory_t, trajectory_3d
    
    def plot_trajectory(self, algorithm_name="Theta*"):
        """可视化轨迹时间历史（区分原始路径点和插入的中间点）"""
        if self.trajectory_t is None:
            self.generate_trajectory()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 绘制光滑轨迹
        ax1.plot(self.trajectory_t, self.trajectory_x, 'b-', linewidth=2, label='Smooth Trajectory(x)')
        ax2.plot(self.trajectory_t, self.trajectory_y, 'g-', linewidth=2, label='Smooth Trajectory(y)')
        ax3.plot(self.trajectory_t, self.trajectory_z, 'm-', linewidth=2, label='Smooth Trajectory(z)')
        
        # 绘制原始路径点（红色大圆点）
        ax1.scatter(self.original_time_nodes, self.original_path[:, 0], c='r', s=80, label='Original Waypoints(x)')
        ax2.scatter(self.original_time_nodes, self.original_path[:, 1], c='r', s=80, label='Original Waypoints(y)')
        ax3.scatter(self.original_time_nodes, self.original_path[:, 2], c='r', s=80, label='Original Waypoints(z)')
        
        # 绘制插入的中间点（蓝色小圆点）
        # 筛选出非原始路径点的中间点
        midpoint_mask = np.ones(len(self.enhanced_path), dtype=bool)
        for orig_point in self.original_path:
            for i, enh_point in enumerate(self.enhanced_path):
                if np.allclose(enh_point, orig_point, atol=1e-6):
                    midpoint_mask[i] = False
                    break
        midpoint_times = self.time_nodes[midpoint_mask]
        midpoint_coords = self.enhanced_path[midpoint_mask]
        
        ax1.scatter(midpoint_times, midpoint_coords[:, 0], c='blue', s=30, alpha=0.5, label='Inserted Midpoints(x)')
        ax2.scatter(midpoint_times, midpoint_coords[:, 1], c='blue', s=30, alpha=0.5, label='Inserted Midpoints(y)')
        ax3.scatter(midpoint_times, midpoint_coords[:, 2], c='blue', s=30, alpha=0.5, label='Inserted Midpoints(z)')
        
        # 设置图表属性
        ax1.set_ylabel('x / m')
        ax1.set_title('Trajectory Time History - {}'.format(algorithm_name))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_ylabel('y / m')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_xlabel('t / s')
        ax3.set_ylabel('z / m')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 额外绘制3D轨迹对比图
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        # 绘制光滑轨迹
        ax3d.plot(self.trajectory_x, self.trajectory_y, self.trajectory_z, 'b-', linewidth=2, label='Smooth Trajectory')
        # 绘制原始路径点连线
        ax3d.plot(self.original_path[:, 0], self.original_path[:, 1], self.original_path[:, 2], 'r--', linewidth=1, alpha=0.7, label='Original Path Line')
        # 绘制原始路径点
        ax3d.scatter(self.original_path[:, 0], self.original_path[:, 1], self.original_path[:, 2], c='r', s=80, label='Original Waypoints')
        # 绘制插入的中间点
        ax3d.scatter(midpoint_coords[:, 0], midpoint_coords[:, 1], midpoint_coords[:, 2], c='blue', s=30, alpha=0.5, label='Inserted Midpoints')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D Trajectory - {}'.format(algorithm_name))
        ax3d.legend()
        plt.show()

def generate_and_plot_flight_trajectory(path, algorithm_name="Theta*", max_segment_dist=3.5):
    path_np = np.array(path, dtype=np.float64)
    if path_np.ndim != 2 or path_np.shape[1] != 3:
        raise ValueError("Path must be a Nx3 numpy array.")
    
    # 传入最大段距离参数，控制中间点插入阈值
    trajectory_generator = CubicPolynomialTrajectoryGenerator(path_np, max_segment_dist=max_segment_dist)
    traj_t, traj_3d = trajectory_generator.generate_trajectory()
    trajectory_generator.plot_trajectory(algorithm_name)
    
    return traj_3d