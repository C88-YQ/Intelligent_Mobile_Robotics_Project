"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write your own path planning algorithm. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
import numpy as np
import math
from heapq import heappush, heappop
import time

class OccupiedMap3D:
    def __init__(self, env, grid_resolution=0.05, safety_distance=0.2):
        self.env = env
        self.grid_resolution = grid_resolution
        self.safety_distance = safety_distance

        self.x_min_world = 0.0
        self.y_min_world = 0.0
        self.z_min_world = 0.0
        self.x_max_world = env.env_width
        self.y_max_world = env.env_length
        self.z_max_world = env.env_height

        # 创建 occupied_map 0 = free, 1 = occupied (obstacle + safety_distance)
        self.occupied_map = None

        start_time = time.time()
        self._build_occupied_map()
        end_time = time.time()
        print(f"构建3D占用栅格地图耗时：{end_time - start_time:.4f}秒")

    def _build_occupied_map(self):
        self.grid_size_x = int(np.ceil((self.x_max_world - self.x_min_world) / self.grid_resolution))
        self.grid_size_y = int(np.ceil((self.y_max_world - self.y_min_world) / self.grid_resolution))
        self.grid_size_z = int(np.ceil((self.z_max_world - self.z_min_world) / self.grid_resolution))

        self.occupied_map = np.zeros((self.grid_size_x, self.grid_size_y, self.grid_size_z), dtype=np.uint8)

        for cx, cy, h, r in self.env.cylinders:
            r_inflated = r + self.safety_distance

            # ceil 向上取整
            x_grid_min = int(max(0, np.floor((cx - r_inflated) / self.grid_resolution)))
            x_grid_max = int(min(self.grid_size_x - 1, np.ceil((cx + r_inflated) / self.grid_resolution)))
            y_grid_min = int(max(0, np.floor((cy - r_inflated) / self.grid_resolution)))
            y_grid_max = int(min(self.grid_size_y - 1, np.ceil((cy + r_inflated) / self.grid_resolution)))

            # 所有圆柱障碍物的高度均从0开始
            z_grid_max = int(min(self.grid_size_z - 1, np.ceil(h / self.grid_resolution)))

            for x_grid in range(x_grid_min, x_grid_max):
                for y_grid in range(y_grid_min, y_grid_max):
                    # 栅格坐标转换为世界坐标
                    x_world = x_grid * self.grid_resolution + self.grid_resolution / 2.0
                    y_world = y_grid * self.grid_resolution + self.grid_resolution / 2.0

                    dist = np.sqrt((x_world - cx) ** 2 + (y_world - cy) ** 2)
                    if dist <= r_inflated:
                        for z_grid in range(0, z_grid_max + 1):
                            self.occupied_map[x_grid, y_grid, z_grid] = 1  # 标记为占用

    def world_to_grid(self, point):
        x, y, z = point

        if (x < self.x_min_world or x >= self.x_max_world or
            y < self.y_min_world or y >= self.y_max_world or
            z < self.z_min_world or z >= self.z_max_world):
            return None  # 超出环境边界
        

        x_grid = int(round(x / self.grid_resolution))
        y_grid = int(round(y / self.grid_resolution))
        z_grid = int(round(z / self.grid_resolution))

        if (x_grid < 0 or x_grid >= self.grid_size_x or
            y_grid < 0 or y_grid >= self.grid_size_y or
            z_grid < 0 or z_grid >= self.grid_size_z):
            return None  # 超出栅格边界
        
        return (x_grid, y_grid, z_grid)
    
    def is_occupied(self, point):
        grid_coords = self.world_to_grid(point)
        if grid_coords is None:
            return True  # 超出边界视为占用
        x_grid, y_grid, z_grid = grid_coords
        return self.occupied_map[x_grid, y_grid, z_grid] == 1
    
class AStar3DPathPlanner:
    """自主实现的3D A*路径规划器"""
    def __init__(self, env):
        self.env = env  # 飞行环境实例
        self.movement_deltas = self._get_3d_movement_deltas()  # 3D移动方向（26邻域）
        self.safety_distance = 0.2  # 与障碍物保持的最小安全距离（可按需调整，单位：m）

        self.use_occupied_map = False

        if self.use_occupied_map:
            self.occupied_map = OccupiedMap3D(
                env=env,
                grid_resolution=0.05,
                safety_distance=self.safety_distance
            )

    def _get_3d_movement_deltas(self):
        """生成3D空间中的26个移动方向（包含上下左右前后及对角线）"""
        deltas = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # 排除原地不动
                    deltas.append((dx, dy, dz))
        return deltas
    
    def _heuristic(self, current, goal):
        """计算启发函数（欧几里得距离，保证A*的最优性）"""
        x1, y1, z1 = current
        x2, y2, z2 = goal
        return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    def _distance_to_nearest_obstacle(self, point):
        """
        计算当前点到最近障碍物的距离
        遍历所有圆柱障碍物，计算点到圆柱表面的最短距离
        """
        x, y, z = point
        min_distance = float('inf')  # 初始化最近距离为无穷大
        
        # 遍历环境中所有圆柱障碍物
        for cx, cy, h, r in self.env.cylinders:
            # 1. 先判断点是否在圆柱的高度范围内（z轴）
            if z < 0 or z > h:
                # 点在圆柱高度外，最短距离为点到圆柱上下底面中心的距离 - 圆柱半径
                dist_to_cylinder_center_xy = math.sqrt((x - cx)**2 + (y - cy)**2)
                dist_to_cylinder_surface = dist_to_cylinder_center_xy - r
            else:
                # 点在圆柱高度内，最短距离为点到圆柱中心（xy平面）的距离 - 圆柱半径
                dist_to_cylinder_center_xy = math.sqrt((x - cx)**2 + (y - cy)**2)
                dist_to_cylinder_surface = dist_to_cylinder_center_xy - r
            
            # 更新到最近障碍物的距离（取非负，避免点在圆柱内部时距离为负）
            current_dist = max(0.0, dist_to_cylinder_surface)
            if current_dist < min_distance:
                min_distance = current_dist
        
        return min_distance

    def _is_valid_point(self, point):
        """验证点是否有效（不越界、不碰撞）"""
        if self.use_occupied_map:
            if self.occupied_map.is_occupied(point):
                return False
        else:
            # 验证是否越界
            if self.env.is_outside(point):
                return False

            # 验证是否与障碍物碰撞
            if self.env.is_collide(point):
                return False

            # 与最近障碍物保持足够的安全距离
            dist_to_nearest_obstacle = self._distance_to_nearest_obstacle(point)
            if dist_to_nearest_obstacle < self.safety_distance:
                return False
        
        # 所有条件满足，点有效
        return True
    
    def plan_path(self, start, goal, step_size=1.0):
        """
        核心路径规划方法
        参数:
            start: 起始点 (x, y, z)
            goal: 目标点 (x, y, z)
            step_size: 移动步长
        返回:
            N×3的numpy数组，包含碰撞-free路径
        """
        # 验证起始点和目标点的有效性
        if not self._is_valid_point(start):
            raise ValueError("起始点无效（越界或碰撞障碍物）")
        if not self._is_valid_point(goal):
            raise ValueError("目标点无效（越界或碰撞障碍物）")
        
        # 初始化A*算法的核心数据结构
        open_heap = []  # 优先队列（小顶堆）
        closed_set = set()  # 已访问节点集合
        came_from = {}  # 路径回溯字典
        g_score = {}  # 从起始点到当前点的实际代价
        f_score = {}  # 总代价（g_score + heuristic）
        
        # 起始点初始化
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        g_score[start_tuple] = 0.0
        f_score[start_tuple] = self._heuristic(start_tuple, goal_tuple)
        heappush(open_heap, (f_score[start_tuple], start_tuple))
        
        # A*主循环
        while open_heap:
            # 取出f_score最小的节点
            current_f, current_point = heappop(open_heap)
            
            # 到达目标点，回溯路径
            if self._heuristic(current_point, goal_tuple) < step_size:
                return self._reconstruct_path(came_from, current_point, goal_tuple)
            
            # 标记当前节点为已访问
            if current_point in closed_set:
                continue
            closed_set.add(current_point)
            
            # 遍历所有可能的移动方向
            for delta in self.movement_deltas:
                # 计算下一步节点
                dx, dy, dz = delta
                next_x = current_point[0] + dx * step_size
                next_y = current_point[1] + dy * step_size
                next_z = current_point[2] + dz * step_size
                next_point = (next_x, next_y, next_z)
                next_tuple = tuple(next_point)
                
                # 跳过无效节点
                if not self._is_valid_point(next_point):
                    continue
                if next_tuple in closed_set:
                    continue
                
                # 计算当前路径的g代价
                tentative_g_score = g_score[current_point] + math.sqrt(dx**2 + dy**2 + dz**2) * step_size
                
                # 更新节点信息（如果找到更优路径）
                if next_tuple not in g_score or tentative_g_score < g_score[next_tuple]:
                    came_from[next_tuple] = current_point
                    g_score[next_tuple] = tentative_g_score
                    f_score[next_tuple] = g_score[next_tuple] + self._heuristic(next_tuple, goal_tuple)
                    heappush(open_heap, (f_score[next_tuple], next_tuple))
        
        # 未找到路径时抛出异常
        raise RuntimeError("无法找到从起始点到目标点的有效路径")
    
    def _reconstruct_path(self, came_from, current, goal):
        """回溯重建路径，并转换为numpy数组格式"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # 反转路径（从起始点到目标点），并添加最终目标点
        path.reverse()
        path.append(goal)
        
        # 转换为N×3的numpy数组，去重并优化
        path_np = np.array(path, dtype=np.float64)
        # 去除连续重复的点
        path_np = path_np[np.unique(np.round(path_np, 3), axis=0, return_index=True)[1]]
        path_np = path_np[np.argsort(np.arange(len(path_np)))]
        
        return path_np
    
class ThetaStar3DPathPlanner:
    def __init__(self, env):
        self.env = env
        self.movement_deltas = self._get_3d_movement_deltas()  # 3D移动方向（26邻域）
        self.los_sample_step = 0.2  # LOS直线检测采样步长（越小越精确，兼顾效率）
        self.safety_distance = 0.4  # 与障碍物保持的最小安全距离（可按需调整，单位：m）

        self.use_occupied_map = True  # 启用占用地图

        if self.use_occupied_map:
            self.occupied_map = OccupiedMap3D(
                env=env,
                grid_resolution=0.05,
                safety_distance=self.safety_distance
            )

    def _get_3d_movement_deltas(self):
        """生成3D空间中的26个移动方向（包含上下左右前后及对角线）"""
        deltas = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # 排除原地不动
                    deltas.append((dx, dy, dz))
        return deltas
    
    def _heuristic(self, current, goal):
        """计算启发函数（欧几里得距离，保证搜索最优性）"""
        x1, y1, z1 = current
        x2, y2, z2 = goal
        return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    def _distance_to_nearest_obstacle(self, point):
        """
        计算当前点到最近障碍物的距离
        遍历所有圆柱障碍物，计算点到圆柱表面的最短距离
        """
        x, y, z = point
        min_distance = float('inf')  # 初始化最近距离为无穷大
        
        # 遍历环境中所有圆柱障碍物
        for cx, cy, h, r in self.env.cylinders:
            # 1. 先判断点是否在圆柱的高度范围内（z轴）
            if z < 0 or z > h:
                # 点在圆柱高度外，最短距离为点到圆柱上下底面中心的距离 - 圆柱半径
                dist_to_cylinder_center_xy = math.sqrt((x - cx)**2 + (y - cy)**2)
                dist_to_cylinder_surface = dist_to_cylinder_center_xy - r
            else:
                # 点在圆柱高度内，最短距离为点到圆柱中心（xy平面）的距离 - 圆柱半径
                dist_to_cylinder_center_xy = math.sqrt((x - cx)**2 + (y - cy)**2)
                dist_to_cylinder_surface = dist_to_cylinder_center_xy - r
            
            # 更新到最近障碍物的距离（取非负，避免点在圆柱内部时距离为负）
            current_dist = max(0.0, dist_to_cylinder_surface)
            if current_dist < min_distance:
                min_distance = current_dist
        
        return min_distance
    
    def _is_valid_point(self, point):

        if self.use_occupied_map:
            if self.occupied_map.is_occupied(point):
                return False
        else:
            # 验证是否越界
            if self.env.is_outside(point):
                return False

            # 验证是否与障碍物碰撞
            if self.env.is_collide(point):
                return False

            # 与最近障碍物保持足够的安全距离
            dist_to_nearest_obstacle = self._distance_to_nearest_obstacle(point)
            if dist_to_nearest_obstacle < self.safety_distance:
                return False
        
        # 所有条件满足，点有效
        return True
    
    def _line_of_sight(self, point_a, point_b):
        """
        θ*核心：3D空间直线可视性检测（LOS）
        新增：直线上所有采样点均需与障碍物保持最小安全距离
        """
        x1, y1, z1 = point_a
        x2, y2, z2 = point_b
        
        # 两点重合，直接视为可视
        if self._heuristic(point_a, point_b) < 1e-6:
            return True
        
        # 计算直线上的采样点数量（保证采样密度，避免漏检障碍物）
        dist_between = self._heuristic(point_a, point_b)
        n_samples = max(5, int(dist_between / self.los_sample_step))
        t_values = np.linspace(0, 1, n_samples)
        
        # 遍历采样点，验证每个点的有效性（包含安全距离验证）
        for t in t_values:
            # 线性插值生成当前采样点
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            sample_point = (x, y, z)
            
            # 若存在无效点（含安全距离不满足），说明直线不可视
            if not self._is_valid_point(sample_point):
                return False
        
        # 所有采样点均有效，直线可视
        return True
    
    def _update_vertex(self, current_point, next_point, parent_point, came_from, g_score, step_size):
        """
        θ*核心：路径拉直优化
        尝试将next_point直接连接到current_point的父节点（parent_point），若LOS可达且代价更优，则更新父节点
        """
        # 情况1：current_point无父节点（即为起始点），直接返回常规代价（与A*一致）
        if parent_point is None:
            move_cost = self._heuristic(current_point, next_point) * step_size
            return g_score[current_point] + move_cost, current_point  # 返回代价+默认父节点
        
        # 情况2：计算常规路径代价（current_point → next_point）
        cost_current_to_next = self._heuristic(current_point, next_point) * step_size
        tentative_g_from_current = g_score[current_point] + cost_current_to_next
        
        # 情况3：检测parent_point → next_point的直线可视性，尝试路径拉直（含安全距离验证）
        if self._line_of_sight(parent_point, next_point):
            cost_parent_to_next = self._heuristic(parent_point, next_point) * step_size
            tentative_g_from_parent = g_score[parent_point] + cost_parent_to_next
            
            # 若拉直路径代价更低，更新next_point的父节点为parent_point（实现路径拉直）
            if tentative_g_from_parent < tentative_g_from_current:
                return tentative_g_from_parent, parent_point
        
        # 情况4：无更优拉直路径，保留常规父节点（current_point）和代价
        return tentative_g_from_current, current_point
    
    def plan_path(self, start, goal, step_size=1.0):
        """
        θ*核心路径规划方法（带障碍物安全距离，生成更安全、更平滑的路径）
        """
        # 验证起始点和目标点的有效性（包含安全距离验证）
        if not self._is_valid_point(start):
            raise ValueError("起始点无效（越界、碰撞障碍物或安全距离不足）")
        if not self._is_valid_point(goal):
            raise ValueError("目标点无效（越界、碰撞障碍物或安全距离不足）")
        
        # 初始化算法核心数据结构（与A*一致，保留父节点追溯支持）
        open_heap = []  # 优先队列（小顶堆）
        closed_set = set()  # 已访问节点集合
        came_from = {}  # 路径回溯字典（记录每个节点的父节点）
        g_score = {}  # 从起始点到当前点的实际代价
        f_score = {}  # 总代价（g_score + heuristic）
        
        # 起始点初始化
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        g_score[start_tuple] = 0.0
        f_score[start_tuple] = self._heuristic(start_tuple, goal_tuple)
        heappush(open_heap, (f_score[start_tuple], start_tuple))
        
        # θ*主循环（继承A*框架，优化节点更新逻辑）
        while open_heap:
            # 取出f_score最小的节点
            current_f, current_point = heappop(open_heap)
            
            # 到达目标点，回溯路径
            if self._heuristic(current_point, goal_tuple) < step_size:
                return self._reconstruct_path(came_from, current_point, goal_tuple)
            
            # 标记当前节点为已访问
            if current_point in closed_set:
                continue
            closed_set.add(current_point)
            
            # 获取当前节点的父节点（θ*关键：用于路径拉直）
            current_parent = came_from.get(current_point, None)
            
            # 遍历所有可能的移动方向
            for delta in self.movement_deltas:
                # 计算下一步节点
                dx, dy, dz = delta
                next_x = current_point[0] + dx * step_size
                next_y = current_point[1] + dy * step_size
                next_z = current_point[2] + dz * step_size
                next_point = (next_x, next_y, next_z)
                next_tuple = tuple(next_point)
                
                # 跳过无效节点（包含安全距离验证）
                if not self._is_valid_point(next_point):
                    continue
                if next_tuple in closed_set:
                    continue
                
                # θ*核心优化：更新节点（尝试路径拉直，获取最优代价和父节点）
                tentative_g_score, best_parent = self._update_vertex(
                    current_point, next_point, current_parent, came_from, g_score, step_size
                )
                
                # 更新节点信息（如果找到更优路径）
                if next_tuple not in g_score or tentative_g_score < g_score.get(next_tuple, float('inf')):
                    came_from[next_tuple] = best_parent  # 记录最优父节点（可能是current_point或current_parent）
                    g_score[next_tuple] = tentative_g_score
                    f_score[next_tuple] = g_score[next_tuple] + self._heuristic(next_tuple, goal_tuple)
                    heappush(open_heap, (f_score[next_tuple], next_tuple))
        
        # 未找到路径时抛出异常
        raise RuntimeError("无法找到从起始点到目标点的有效路径（可能无满足安全距离的可行路径）")
    
    def _reconstruct_path(self, came_from, current, goal):
        """回溯重建路径，并转换为numpy数组格式（与原A*保持一致，保证格式兼容）"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # 反转路径（从起始点到目标点），并添加最终目标点
        path.reverse()
        path.append(goal)
        
        # 转换为N×3的numpy数组，去重并优化
        path_np = np.array(path, dtype=np.float64)
        # 去除连续重复的点
        path_np = path_np[np.unique(np.round(path_np, 3), axis=0, return_index=True)[1]]
        path_np = path_np[np.argsort(np.arange(len(path_np)))]
        
        return path_np

def plan_flight_path(env, start, goal):
    """对外暴露的路径规划接口，方便main.py调用（无需修改，直接兼容）"""

    start_time = time.time()
    planner = ThetaStar3DPathPlanner(env)
    path = planner.plan_path(start, goal, step_size=1.0)
    end_time = time.time()
    print(f"Theta*路径规划耗时：{end_time - start_time:.4f}秒")

    start_time = time.time()
    planner_astar = AStar3DPathPlanner(env)
    path_astar = planner_astar.plan_path(start, goal, step_size=1.0)
    end_time = time.time()
    print(f"A*路径规划耗时：{end_time - start_time:.4f}秒")
    
    return path
