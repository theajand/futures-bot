# src/rl_env.py - Gym env + Torch DQN for dynamic position sizing (big on low vol, small high)
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pandas as pd

# Custom Env (state: vol, equity, current position; reward: return - drawdown penalty)
class TradingEnv(gym.Env):
    def __init__(self, data):  # data = df with 'atr_14' vol proxy, 'Returns' = close.pct_change
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = spaces.Discrete(5)  # Sizes: 0.5, 0.75, 1.0, 1.5, 2.0
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)  # vol, equity, pos

    def reset(self):
        self.current_step = 0
        self.equity = 100000
        self.position = 0
        return np.array([self.data['atr_14'][0], self.equity, self.position])

    def step(self, action):
        size = [0.5, 0.75, 1.0, 1.5, 2.0][action]
        ret = self.data['Returns'][self.current_step] * size
        self.equity *= (1 + ret)
        reward = ret - 0.01 * (self.data['atr_14'][self.current_step] * size)**2  # Penalize high vol big size
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = np.array([self.data['atr_14'][self.current_step], self.equity, size])
        return obs, reward, done, {}

# DQN Agent
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Train agent
def train_rl(df, episodes=100, batch_size=64, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    env = TradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    memory = deque(maxlen=2000)
    eps = eps_start

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = agent(torch.tensor(state).float()).argmax().item()
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states).float()
                next_states = torch.tensor(next_states).float()
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards).float()
                dones = torch.tensor(dones).float()

                q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = agent(next_states).max(1)[0]
                target = rewards + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(eps_end, eps_decay * eps)
        print(f"Episode {e}: Reward {total_reward:.2f}, Equity {env.equity:.2f}")

    torch.save(agent.state_dict(), 'models/dqn_sizing.pth')
    return agent

# Example: Load df from features, add 'Returns' = close_spy.pct_change, train
df = pd.read_csv('data/features.csv')
df['Returns'] = df['close_spy'].pct_change().fillna(0)
train_rl(df)