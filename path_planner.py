"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
import numpy as np
import math
from heapq import heappush, heappop

class AStar3DPathPlanner:
    """自主实现的3D A*路径规划器"""
    def __init__(self, env):
        self.env = env  # 飞行环境实例
        self.movement_deltas = self._get_3d_movement_deltas()  # 3D移动方向（26邻域）
    
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
    
    def _is_valid_point(self, point):
        """验证点是否有效（不越界、不碰撞）"""
        x, y, z = point
        # 转换为整数坐标（环境碰撞检测通常基于网格）
        int_point = (int(round(x)), int(round(y)), int(round(z)))
        if self.env.is_outside(int_point):
            return False
        if self.env.is_collide(int_point):
            return False
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

def plan_flight_path(env, start, goal):
    """对外暴露的路径规划接口，方便main.py调用"""
    planner = AStar3DPathPlanner(env)
    return planner.plan_path(start, goal, step_size=1.0)
