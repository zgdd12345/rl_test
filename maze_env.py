
import gym
from gym import spaces
import numpy as np

from maze_matrix import maze_matrix  # 假设maze_matrix.py在同一目录下   

class MazeEnv(gym.Env):
    def __init__(self):
        # 16x16迷宫，0=可通行，1=障碍
        self.maze = maze_matrix
        self.start_pos = (0, 0)  # 起点
        self.goal_pos = (15, 15)   # 终点
        self.agent_pos = list(self.start_pos)
        
        # 动作空间：上下左右
        self.action_space = spaces.Discrete(4)
        # 观察空间：智能体坐标
        self.observation_space = spaces.Box(low=0, high=15, shape=(2,), dtype=np.int32)
        
        # 可视化参数
        self.viewer = None
        self.cell_size = 50
        
    def reset(self):
        self.agent_pos = list(self.start_pos)
        return np.array(self.agent_pos)
    
    def step(self, action):
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        # 执行动作
        if action == 0: new_x -= 1  # 上
        elif action == 1: new_x += 1  # 下
        elif action == 2: new_y -= 1  # 左
        elif action == 3: new_y += 1  # 右
        
        # 检查新位置是否有效
        if (0 <= new_x < 16 and 0 <= new_y < 16 and 
            self.maze[new_x, new_y] == 0):
            self.agent_pos = [new_x, new_y]
        
        # 检查是否到达目标
        done = (new_x, new_y) == self.goal_pos
        reward = 20 if done else -0.1  # 到达目标获得更大奖励
        
        return np.array(self.agent_pos), reward, done, {}
    
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            
            # 绘制迷宫
            for x in range(16):
                for y in range(16):
                    rect = rendering.FilledPolygon([
                        (y*self.cell_size, x*self.cell_size),
                        ((y+1)*self.cell_size, x*self.cell_size),
                        ((y+1)*self.cell_size, (x+1)*self.cell_size),
                        (y*self.cell_size, (x+1)*self.cell_size)
                    ])
                    if self.maze[x, y] == 1:  # 障碍
                        rect.set_color(0, 0, 0)
                    elif (x, y) == tuple(self.goal_pos):  # 目标
                        rect.set_color(0, 1, 0)
                    else:  # 可通行区域
                        rect.set_color(1, 1, 1)
                    self.viewer.add_geom(rect)
                    
            # 绘制智能体
            self.agent_trans = rendering.Transform()
            agent = rendering.make_circle(self.cell_size//3)
            agent.set_color(0, 0, 1)
            agent.add_attr(self.agent_trans)
            self.viewer.add_geom(agent)
        
        # 更新智能体位置
        x, y = self.agent_pos
        self.agent_trans.set_translation(
            (y+0.5)*self.cell_size, (x+0.5)*self.cell_size)
        
        return self.viewer.render(mode)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
