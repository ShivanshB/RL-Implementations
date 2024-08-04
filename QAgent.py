import numpy as np

class QAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.9):
        # initializing values
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # we will decay epsilon by a constant factor every episode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # q table
        self.q = {}

    def get_q_values(self, state, action):
        # return q value if recorded, else 0
        return self.q.get((state, action), 0.0)

    def choose_action(self, state):
        # explore
        if np.random.random() < self.epsilon:
            # choose a random action
            return self.action_space.sample()
        # exploit
        else:
            # choose the action with max value
            return max(range(self.action_space.n), key=lambda a: self.get_q_values(state, a))
        
    def learn(self, state, action, reward, next_state, done):
        # current q value
        current_q = self.get_q_values(state, action)

        # decay epsilon and do forecasing depending on if episode is complete
        if done:
            self.epsilon = 0.9 * self.epsilon_decay
            next_max_q = 0
        else:
            # maximum value in next state
            next_max_q = max([self.get_q_values(next_state, a) for a in range(self.action_space.n)])

        # q learning update equation
        self.q[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)