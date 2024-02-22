import gymnasium as gym
import numpy as np
import math
import random
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

class Environment():

	'''
	Grid:
	0 => border or obstacles
	1 => moveable positions
	2 => start
	3 => end

	Position: (y, x) for direct matrix checking
	'''
	def __init__(self, gridPath, timelimit= -1) -> None:
		self.timelimit = timelimit
		self.grid = np.loadtxt(gridPath)
		self.start_line = np.where(self.grid == 2)
		self.height, self.width = self.grid.shape

		end_points = np.where(self.grid == 3)
		if(len(end_points[0]) == 1):
			self.end = [end_points[0][0], end_points[1][0]]
		else:
			raise "Extacly one endpoint must be provided"

		self.history = []
		self.colormap = colors.ListedColormap(["black", "white", "blue", "green", "red", "orange"])

		self.actions = np.array([[1, -1], [1, 0], [1, 1], [0, -1], [0, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1]], dtype=np.int64)
		self.time = 0
		self.reset_pos()


	'''
	Actions numbered by Matrix for Speed Changes:
	(1, -1) (1, 0) (1, 1)
	(0, -1) (0, 0) (0, 1)
	(-1, -1) (-1, 0) (-1, 1)

	Returns:
	observation: Current position as state
	reward: -1 when end point is not reached in step, 0 else
	terminated: true if robot reached the target position
	truncated: true if environmant is stopped after timelimit reached
	info: absolute speeds for x and y axis
	'''
	def step(self, action:int) -> tuple[list[float], int, bool, bool, list[int]]:
		truncated = False
		terminated = False
		reward = -1

		if(self.timelimit != -1 and self.time > self.timelimit):
			truncated = True 

		speedChange = self.actions[action]
		self.speed[0] = max(0, min(3, self.speed[0] + speedChange[0]))
		self.speed[1] = max(0, min(3, self.speed[1] + speedChange[1]))

		if(self.speed[0] == 0 and self.speed[1] == 0):
			self.speed[random.choice([0, 1])] = 1
	
		if (not self.move_one_step()):
			self.history.append(self.pos.copy())
			self.reset_pos()

		# print(self.grid[self.pos[0] - self.speed[0]:self.pos[0] + 1, self.pos[1]:self.pos[1] + self.speed[1] + 1])

		return self.pos, reward, terminated, truncated, self.speed


	'''
	Reseting environment

	Returns:
	observation: Current position as state
	info: absolute speeds for x and y axis
	'''
	def reset(self) -> tuple[list[float], list[int]]:
		self.reset_pos()
		self.history = []
		self.time = 0

		return self.pos, self.speed
	

	'''
	Print the grid with the path until this timestep
	'''
	def print(self) -> None:
		print(self.history)
		print_grid = self.grid.copy()
		print_grid[self.pos[0], self.pos[1]] = 4

		for pos in self.history:
			print_grid[pos[0], pos[1]] = 5

		plt.figure(figsize=(self.width, self.height))
		plt.imshow(print_grid, cmap=self.colormap, interpolation='none')
		plt.show()
		return
	

	'''
	Reset position to random choice on starting line and speed to zero
	'''
	def reset_pos(self) -> None:
		index = random.choice(range(len(self.start_line[0])))
		self.pos = np.array([self.start_line[0][index], self.start_line[1][index]])
		self.speed = np.zeros(2, dtype=np.int64)


	'''
	Find a path with the defined x and y speed without obstacles on the grid
	'''
	def move_one_step(self) -> bool:
		new_pos = self.pos.copy()
		y = self.speed[0]
		x = self.speed[1]

		while (x != 0 or y != 0):
			self.history.append(new_pos.copy())
			try:
				if (y != 0 and self.grid[new_pos[0] - np.sign(y)][new_pos[1]] != 0):
					new_pos[0] -= np.sign(y)
					y -= np.sign(y)
				elif (x != 0 and self.grid[new_pos[0]][new_pos[1] + np.sign(x)] != 0):
					new_pos[1] += np.sign(x)
					x -= np.sign(x)
				else:
					print("Reset: Hit an Obstacle")
					return False
				
				self.pos = new_pos
			except:
				print("Reset: Hit the Border")
				return False
				
		return True

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

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = epsylon_end + (epsylon_start - epsylon_end) * \
        math.exp(-1. * steps_done / epsylon_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.actions.step()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def training(episodes):
    plt.ion()

    for i_episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*update_rate + target_net_state_dict[key]*(1-update_rate)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()



if __name__ == "__main__":
    env = Environment('./grids/grid_simple.txt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        
    batch_size = 128
    gamma = 0.99
    epsylon_start = 0.9
    epsylon_end = 0.05
    epsylon_decay = 1000
    update_rate = 0.005
    learning_rate = 1e-4

    n_actions = env.actions.size
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []
    episodes = 50
    
    training(episodes)