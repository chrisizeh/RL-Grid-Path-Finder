import numpy as np
import math
import random
import os 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import namedtuple, deque
from itertools import count
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment import Environment

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, layer, activation=nn.ReLU, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = self.create_model(layer, activation, dropout)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

    def create_model(self, layer, activation=nn.ReLU, dropout=0):
        layer_nn = []
        for i in range(len(layer) - 1):
            layer_nn.append(nn.Linear(layer[i], layer[i+1]))
            layer_nn.append(activation())

            if(dropout > 0):
                layer_nn.append(nn.Dropout(dropout))

        return nn.Sequential(*layer_nn)


class Agent():

    def __init__(self, env, nodes, batch_size=128, gamma=0.7, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=2, update_rate=0.005, learning_rate=1e-3) -> None:
        self.env = env
        self.n_actions = env.n_actions
        self.n_observations = env.n_states

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_rate = update_rate
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN([self.n_observations, nodes, self.n_actions]).to(self.device)
        self.target_net = DQN([self.n_observations, nodes, self.n_actions]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps = 0

    def select_action(self, state):
        sample = random.random()
        if sample > self.curr_epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.choice(range(self.n_actions))]], device=self.device, dtype=torch.long)


    def plot_rewards(self, show_result=False, path=None):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            if(path):
                plt.savefig(path)
            else:
                display.display(plt.gcf())



    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def training(self, episodes, plot_training=True, max_steps = 1000, path=None):
        plt.ion()
        self.episode_rewards = []
        # max_epsilon_step = episodes / self.epsilon_decay

        for i_episode in range(episodes):
            self.curr_epsilon = self.epsilon_end
            # self.curr_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (min(i_episode, max_epsilon_step) / max_epsilon_step)

            rewards = np.zeros(max_steps)
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                rewards[t] = reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.update_rate + target_net_state_dict[key] * (1 - self.update_rate)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(np.sum(rewards))

                    if plot_training: 
                        self.plot_rewards()
                    break
            
            print(f'Episode {i_episode} complete with epsilon {self.curr_epsilon}')

        print('Complete')
        self.plot_rewards(show_result=True, path=os.path.join(path, "rewards.png"))
        plt.ioff()
        plt.show()


    def get_test_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.target_net(state).max(1).indices.view(1, 1)


if __name__ == "__main__":
    path = os.path.join("plots", "deepQ")

    try:  
        os.mkdir("plots")
    except:  
        print("Path already exists")
    try:
        os.mkdir(path)
    except:  
        print("Path already exists")

    episodes = 50
    timelimit = 1000

    env_text = "grid_simple"
    env = Environment(f'./grids/{env_text}.txt', timelimit=timelimit)    
    path = os.path.join(path, env_text)
    try:  
        os.mkdir(path)
    except:  
        print("Path already exists")

    agent = Agent(env, 12, learning_rate=0.001, batch_size=24)
    agent.training(episodes, plot_training=True, max_steps=timelimit + 1, path=path)
    agent.env.test_start_positions(agent.get_test_action, path=path)