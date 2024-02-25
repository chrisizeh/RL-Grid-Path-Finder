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

'''
Store Transisitions for neural network optimization
'''
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


'''
Create pytorch neural network
'''
class DQN(nn.Module):

    '''
    Parameter:
        layer - list of node sizes to be used to build layer
        activation - activation function to use for each layer
        dropout - percentage of dropout to add between each layer
    '''
    def __init__(self, layer, activation=nn.ReLU, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = self.create_model(layer, activation, dropout)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

    '''
    Build sequential list of layer by using each node from the layer list
    [20, 40, 60, 2] -> (20, 40), (40, 60), (60, 2)
    '''
    def create_model(self, layer, activation=nn.ReLU, dropout=0):
        layer_nn = []
        for i in range(len(layer) - 1):
            layer_nn.append(nn.Linear(layer[i], layer[i+1]))
            layer_nn.append(activation())

            if(dropout > 0):
                layer_nn.append(nn.Dropout(dropout))

        return nn.Sequential(*layer_nn)


'''
Deep-Q Learner for Environment with OpenAI Gym Interface
'''
class Agent():

    '''
    Parameters:
        env             - Environment with OpenAI Gym Interface
        nodes           - Node size list to build neural network from
        batch_size      - Number of transitions to use for optimization step
        gamma           - Reward Decay
        epsilon         - Percentage of random action selection (Exploration)
        update_rate     - Learning rate from learning policy to target policy
        learning_rate   - learning rate of learning policy
    '''
    def __init__(self, env, nodes, batch_size=128, gamma=0.7, epsilon=0.9, update_rate=0.005, learning_rate=1e-3):
        self.env = env
        self.n_actions = env.n_actions
        self.n_observations = env.n_states

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_rate = update_rate
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN([self.n_observations, nodes, self.n_actions]).to(self.device)
        self.target_net = DQN([self.n_observations, nodes, self.n_actions]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps = 0


    '''
    Select action from learning policy with epsilon degree of randomness

    Parameter: state - current state to select action for
    Returns: action as integer
    '''
    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.choice(range(self.n_actions))]], device=self.device, dtype=torch.long)


    '''
    Plot Reward over episodes with moving average over 100 episodes

    Parameter:
        show_results - If training is finished and results are final
        path         - If provided plot is saved when show_results = True
    '''
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



    '''
    Optimize neural network using adam optimizer and random transitions sampled from memory
    '''
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


    '''
    Run multiple episodes, training the agent or just applying the policy

    Parameters:
        episodes    - Number of episodes to run
        plot        - If true, plot training progress or environment movement
        max_steps   - max possible steps of the environment before truncating
        path        - if provided plotted images are stored
        training    - True if policy should be optimized, False optimization is ommited (epsilon=0)
    '''
    def run(self, episodes, plot=True, max_steps = 1000, path=None, training=True):
        plt.ion()
        self.episode_rewards = []

        if (not training):
            self.epsilon = 0

        for i_episode in range(episodes):
            rewards = np.zeros(max_steps)
            state, _ = self.env.reset(i_episode % self.env.n_starts)
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

                if (training):
                    state = next_state
                    self.optimize_model()

                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * self.update_rate + target_net_state_dict[key] * (1 - self.update_rate)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(np.sum(rewards))

                    if plot: 
                        if (training):
                            self.plot_rewards()
                        else:
                            self.env.print(os.path.join(path, f"episode_{i_episode}.png"))
                    break
            
            print(f'Episode {i_episode} complete after {t} steps')

        print('Complete')
        if (training):
            self.plot_rewards(show_result=True, path=os.path.join(path, "rewards.png"))
        plt.ioff()



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

    episodes = 10000
    timelimit = 1000

    env_text = "grid_simple"
    env = Environment(f'./grids/{env_text}.txt', timelimit=timelimit)    
    path = os.path.join(path, env_text)
    try:  
        os.mkdir(path)
    except:  
        print("Path already exists")

    agent = Agent(env, 64, learning_rate=0.001, batch_size=24)
    agent.run(episodes, plot=False, max_steps=timelimit + 1, path=path)
    agent.run(env.n_starts, plot=True, max_steps=timelimit + 1, path=path, training=False)
