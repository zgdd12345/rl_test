
import pygame
import numpy as np
import time

# 初始化pygame
pygame.init()

# 环境参数
GRID_SIZE = 4
CELL_SIZE = 100
MARGIN = 2
WINDOW_SIZE = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# 创建窗口
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("GridWorld RL")

# 定义环境
class GridWorld:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.goal = (GRID_SIZE-1, GRID_SIZE-1)
        self.agent_pos = [0, 0]
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)
    
    def step(self, action):
        x, y = self.agent_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < GRID_SIZE-1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < GRID_SIZE-1:
            y += 1
            
        self.agent_pos = [x, y]
        done = (x, y) == self.goal
        reward = 1 if done else -0.1
        return (x, y), reward, done

# Q学习算法
class QLearning:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(env.actions)))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]
            
    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.env.actions.index(action)
        
        predict = self.q_table[x, y, action_idx]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action_idx] += self.alpha * (target - predict)

# 主循环
def main():
    env = GridWorld()
    agent = QLearning(env)
    
    running = True
    clock = pygame.time.Clock()
    episode = 0
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 训练一个episode
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 学习
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            steps += 1
            
            # 渲染
            screen.fill(WHITE)
            
            # 绘制网格
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    color = GRAY
                    if (row, col) == env.goal:
                        color = GREEN
                    elif [row, col] == env.agent_pos:
                        color = BLUE
                        
                    pygame.draw.rect(screen, color, 
                                   [(MARGIN + CELL_SIZE) * col + MARGIN,
                                    (MARGIN + CELL_SIZE) * row + MARGIN,
                                    CELL_SIZE, CELL_SIZE])
            
            # 显示信息
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"Episode: {episode}  Steps: {steps}", True, BLACK)
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(10)  # 控制渲染速度
            
        episode += 1
        
    pygame.quit()

if __name__ == "__main__":
    main()
