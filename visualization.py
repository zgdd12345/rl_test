import pygame
import time
from maze_env import MazeEnv
from rl_agent import QLearningAgent

# 初始化
pygame.init()
env = MazeEnv()
agent = QLearningAgent(env)
CELL_SIZE = 50
MAZE_WIDTH = 16*CELL_SIZE
MAZE_HEIGHT = 16*CELL_SIZE
INFO_HEIGHT = 150  # 用于显示信息的额外高度
WINDOW_SIZE = (MAZE_WIDTH, MAZE_HEIGHT + INFO_HEIGHT)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Maze RL")
font = pygame.font.Font(None, 36)

# 颜色定义
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,0,255)
GRAY = (200,200,200)

def draw_maze():
    for x in range(16):
        for y in range(16):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if env.maze[x,y] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)
    
    # 绘制目标
    goal_rect = pygame.Rect(env.goal_pos[1]*CELL_SIZE, 
                           env.goal_pos[0]*CELL_SIZE, 
                           CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, goal_rect)
    
    # 绘制智能体
    agent_rect = pygame.Rect(env.agent_pos[1]*CELL_SIZE+CELL_SIZE//4,
                            env.agent_pos[0]*CELL_SIZE+CELL_SIZE//4,
                            CELL_SIZE//2, CELL_SIZE//2)
    pygame.draw.ellipse(screen, BLUE, agent_rect)

def draw_info(elapsed_time, episode, all_time):
    # 在单独的信息区域绘制文本
    info_area = pygame.Rect(0, MAZE_HEIGHT, MAZE_WIDTH, INFO_HEIGHT)
    screen.fill(WHITE, info_area)  # 清除信息区域
    
    round_text = font.render(f"Round: {episode + 1}", True, BLACK)
    time_text = font.render(f"Time: {elapsed_time:.2f}s", True, BLACK)
    screen.blit(round_text, (10, MAZE_HEIGHT + 10))
    screen.blit(time_text, (10, MAZE_HEIGHT + 50))
    all_time_text = font.render(f"Total Time: {all_time:.2f}s", True, BLACK)
    screen.blit(all_time_text, (10, MAZE_HEIGHT + 90))
    pygame.draw.rect(screen, BLACK, info_area, 1)  # 绘制信息区域边框

# 主循环
running = True
clock = pygame.time.Clock()
s_time = time.time()
for episode in range(1000):
    state = env.reset()
    done = False
    start_time = time.time()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not running:
            break
            
        # RL步骤
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        
        # 渲染
        screen.fill(WHITE)
        draw_maze()
        
        # 显示轮次和运行时间
        elapsed_time = time.time() - start_time
        all_time = time.time() - s_time
        draw_info(elapsed_time, episode, all_time)
        
        pygame.display.flip()
        clock.tick(10000)  # 控制渲染速度
        
    if not running:
        break

pygame.quit()
