import torch
import random
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, hidden_dim, action_space, learning_rate=1e-3, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64):
        
        # initalize params
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.num_actions = action_space.n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize q-network and target model
        self.q_network = DQN(self.state_dim, self.hidden_dim, self.num_actions)
        self.target_network = DQN(self.state_dim, self.hidden_dim, self.num_actions)

        # copy over weights from q-network to target model
        self.update_target_model()

        # set target model to eval b/c it will not be training
        self.target_network.eval()

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        # set optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)


    def select_action(self, state):
        # explore
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        # exploit
        with torch.no_grad():
            # convert input state to tensor 
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # generate q values for this state
            q_values = self.q_network(state)

            # return action corresponding to max value
            return q_values.argmax().item()
    
    def train(self):
        # check for enough data left to train
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # sample  and split training data
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # move values to tensors onto device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # inference
        q_values = self.q_network(states)[range(self.batch_size), actions]

        # turn off grad
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values

        # estimate q values using MDP property
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # calculate loss
        loss = self.loss_fn(q_values, target_q_values)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon for explore-exploit
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())