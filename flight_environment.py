import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题（配置matplotlib字体，消除中文警告）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 优先英文，备选黑体（有中文环境则生效）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示问题

class FlightEnvironment:
    def __init__(self,obs_num):
        self.env_width = 20.0
        self.env_length = 20.0
        self.env_height = 5
        self.space_size = (self.env_width,self.env_length,self.env_height)
        self._obs_num = obs_num

        self.cylinders = self.generate_random_cylinders(self.space_size,self._obs_num,0.1,0.3,5,5)

    def generate_random_cylinders(self,space_size, N,
                              min_radius, max_radius,
                              min_height, max_height,
                              max_tries=100000):

        X, Y, Z = space_size
        cylinders = []
        tries = 0

        while len(cylinders) < N and tries < max_tries:
            tries += 1

            r = np.random.uniform(min_radius, max_radius)
            h = np.random.uniform(min_height, min(max_height, Z))

            x = np.random.uniform(r, X - r)
            y = np.random.uniform(r, Y - r)

            candidate = np.array([x, y, h, r])

            no_overlapping = True
            for c in cylinders:
                dx = x - c[0]
                dy = y - c[1]
                dist = np.hypot(dx, dy)
                if dist < (r + c[3]):  
                    no_overlapping = False
                    break

            if no_overlapping:
                cylinders.append(candidate)

        if len(cylinders) < N:
            raise RuntimeError("Unable to generate a sufficient number of non-overlapping cylinders with the given parameters. Please reduce N or decrease the radius range.")

        return np.vstack(cylinders)
    
    def is_outside(self,point):
        """
        Check whether a 3D point lies outside the environment boundary.
        """
        x,y,z = point
        if (0 <= x <= self.env_width and
                0 <= y <= self.env_length and
                0 <= z <= self.env_height):
            outside_env = False
        else:
            outside_env = True
        return outside_env
    
    def is_collide(self, point, epsilon=0.2):
        """
        Check whether a point in 3D space collides with a given set of cylinders.
        """
        cylinders = self.cylinders
        px, py, pz = point

        for cx, cy, h, r in cylinders:
            if not (0 <= pz <= h):
                continue 
            dist_xy = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist_xy <= (r + epsilon):
                return True   
        
        return False
    
    def plot_cylinders(self, path=None, trajectory=None):
        """
        可视化障碍物、离散路径、光滑轨迹（同一张3D图对比，修复图例和中文问题）
        参数:
            path: 离散路径点（N×3数组）
            trajectory: 光滑轨迹（M×3数组，连续坐标）
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cylinders = self.cylinders
        space_size = self.space_size
        Xmax, Ymax, Zmax = space_size
        
        # 1. 绘制障碍物圆柱（移除label，3D曲面不参与图例，避免报错）
        for cx, cy, h, r in cylinders:
            z = np.linspace(0, h, 30)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta, z = np.meshgrid(theta, z)

            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)

            # 移除label='Obstacles'，避免图例处理异常
            ax.plot_surface(x, y, z, color='skyblue', alpha=0.5)
            theta2 = np.linspace(0, 2*np.pi, 30)
            x_top = cx + r * np.cos(theta2)
            y_top = cy + r * np.sin(theta2)
            z_top = np.ones_like(theta2) * h
            ax.plot_trisurf(x_top, y_top, z_top, color='steelblue', alpha=0.6)

        # 2. 设置坐标轴
        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_length)
        ax.set_zlim(0, self.env_height)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # 改用英文标题，彻底消除中文字体问题（如需中文，需确保环境有对应字体）
        ax.set_title('3D Flight Environment: Discrete Path vs Smooth Trajectory')

        # 3. 绘制离散路径（仅给有效元素加label，避免重复）
        discrete_path_label = True  # 控制图例只添加一次
        if path is not None:
            path = np.array(path, dtype=np.float64)
            if path.ndim == 2 and path.shape[1] == 3:
                xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
                
                # 离散路径连线（粗红，虚线，添加有效label）
                ax.plot(xs, ys, zs, linewidth=2.5, color='darkred', linestyle='--', 
                        alpha=0.8, label='Discrete Path' if discrete_path_label else "")
                
                # 离散路径点（小巧，金色带黑边，散点不重复加label）
                n_points = len(xs)
                if n_points >= 1:
                    # 首尾点（仅给起始点加label，避免图例重复）
                    ax.scatter(xs[0], ys[0], zs[0], s=80, color='gold', marker='*', 
                               edgecolors='black', linewidth=0.5, label='Start Point' if discrete_path_label else "")
                    ax.scatter(xs[-1], ys[-1], zs[-1], s=80, color='limegreen', marker='*', 
                               edgecolors='black', linewidth=0.5)
                    # 中间点（无label，避免图例冗余）
                    if n_points > 2:
                        ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], s=25, color='royalblue', 
                                   marker='o', alpha=0.9)
                    discrete_path_label = False  # 关闭label开关，避免重复

        # 4. 绘制光滑轨迹（添加有效label，与离散路径区分）
        if trajectory is not None:
            trajectory = np.array(trajectory, dtype=np.float64)
            if trajectory.ndim == 2 and trajectory.shape[1] == 3:
                traj_x, traj_y, traj_z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
                
                # 光滑轨迹连线（细蓝，实线，添加label）
                ax.plot(traj_x, traj_y, traj_z, linewidth=1.8, color='cornflowerblue', 
                        alpha=0.9, label='Smooth Trajectory')

        # 5. 显示图例（仅包含有效元素，无3D曲面，避免报错）
        ax.legend(loc='best', fontsize=8)
        self.set_axes_equal(ax)
        plt.show()

    def set_axes_equal(self,ax):
        """Make axes of 3D plot have equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max([x_range, y_range, z_range]) / 2.0

        mid_x = (x_limits[0] + x_limits[1]) * 0.5
        mid_y = (y_limits[0] + y_limits[1]) * 0.5
        mid_z = (z_limits[0] + z_limits[1]) * 0.5

        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
