import numpy as np
from maze_matrix import maze_matrix

def is_path_exists(maze, start, end):
    """使用BFS检查从起点到终点是否存在路径"""
    from collections import deque
    
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([start])
    visited[start] = True
    
    # 四个方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x, y = queue.popleft()
        
        # 检查是否到达终点
        if (x, y) == end:
            return True
            
        # 探索四个方向
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 检查边界和访问状态
            if (0 <= nx < rows and 0 <= ny < cols and 
                not visited[nx, ny] and maze[nx, ny] == 0):
                visited[nx, ny] = True
                queue.append((nx, ny))
                
    return False

# 测试迷宫连通性
start_pos = (0, 0)
goal_pos = (15, 15)

print("Maze shape:", maze_matrix.shape)
print("Start position (0,0):", "Free" if maze_matrix[0,0] == 0 else "Blocked")
print("Goal position (15,15):", "Free" if maze_matrix[15,15] == 0 else "Blocked")

# 检查路径是否存在
path_exists = is_path_exists(maze_matrix, start_pos, goal_pos)
print(f"Path exists from {start_pos} to {goal_pos}: {path_exists}")

# 显示迷宫的一部分以进行视觉检查
print("\nTop-left corner of maze (5x5):")
print(maze_matrix[:5, :5])

print("\nBottom-right corner of maze (5x5):")
print(maze_matrix[-5:, -5:])
