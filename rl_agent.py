import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((10, 10, 4))  # 10x10网格，4个动作
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state[0], state[1]])
    
    def learn(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.lr * (target - predict)
