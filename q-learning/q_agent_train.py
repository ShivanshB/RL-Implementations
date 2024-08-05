import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
from tqdm import tqdm
from ..maze_env import MazeEnv
from q_agent import QAgent

def train_agent(env, agent, episodes=100):
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

# initialize the environment and agent
env = MazeEnv(maze_height=20, maze_width=20, animation=True)
agent = QAgent(env.action_space, env.observation_space, learning_rate=1, epsilon_decay=0.1)

# train the agent in the environment
train_agent(env, agent)

# animate learning
env.animate('training')