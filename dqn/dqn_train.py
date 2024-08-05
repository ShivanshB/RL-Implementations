import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_env import MazeEnv
from dqn_agent import DQNAgent

def train_dqn(env, agent, num_episodes, max_steps, update_target_every=100):
    # training loop
    for episode in range(num_episodes):
        
        # copy over weights and refill replay buffer
        if episode % update_target_every == 0:
            update_replay_buffer(env, agent)
            agent.update_target_model()
        
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # continually add to the replay buffer
            agent.replay_buffer.append(state, action, reward, next_state, done)

            # train agent
            agent.train()

            # increment/update values
            state = next_state
            total_reward += reward

            # check if done
            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


def update_replay_buffer(env, agent):
    # warm up replay buffer, fill with varied training data
    state, _ = env.reset()
    for _ in range(agent.replay_buffer.max_samples):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.append(state, action, reward, next_state, done)
        
        # check if done
        if done:
            state, _ = env.reset()
        else:
            state = next_state

# create maze environment
env = MazeEnv(5, 5, animation=True)

# extract layer dimension
state_dim = env.observation_space.shape[0]

# create and train agent
agent = DQNAgent(state_dim, hidden_dim=128, action_space=env.action_space, buffer_capacity=50000, learning_rate=0.0005, epsilon_decay=0.999, epsilon_min=0.1, batch_size=128)
train_dqn(env, agent, num_episodes=100, max_steps=50000, update_target_every=10)

env.animate('training')

