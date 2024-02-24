import numpy as np
import matplotlib.pyplot as plt
import random
import os

from IPython import display
from itertools import count
import torch

from environment import Environment

class Agent():

	def __init__(self, env, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=2, gamma=0.7) -> None:
		self.env = env
		self.n_actions = env.n_actions
		self.n_observations = env.n_states

		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		
		self.policy = np.full((self.env.height, self.env.width, 3, 3, self.n_actions), 1 / self.n_actions)

		self.steps = 0

	
	def get_test_action(self, state):
		return np.argmax(self.policy[state[0]][state[1]][state[2]][state[3]])


	def select_action(self, state):
		sample = random.random()
		
		if sample > self.curr_epsilon:
			return np.argmax(self.policy[state[0]][state[1]][state[2]][state[3]])
		else:
			return np.random.randint(0, self.n_actions)
		
	
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
 

	def training(self, episodes, plot_training=True, max_steps=1000, path=None):
		plt.ion()
		self.episode_rewards = []
		# max_epsilon_step = episodes / self.epsilon_decay

		for i_episode in range(episodes):
			self.curr_epsilon = self.epsilon_end
			# self.curr_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (min(i_episode, max_epsilon_step) / max_epsilon_step)

			rewards = np.zeros(max_steps)
			timeline = np.empty(max_steps, dtype=tuple)
			state, _ = self.env.reset()

			for t in count():
				action = self.select_action(state)
				timeline[t] = (state, action)
				state, reward, terminated, truncated, _ = self.env.step(action)
				rewards[t] = reward

				done = terminated or truncated
				if done or t > max_steps:
					self.episode_rewards.append(np.sum(rewards))
					timeline = timeline[:t]
					rewards = rewards[:t]
					break

			q = np.zeros(shape=(self.env.height, self.env.width, 3, 3, self.n_actions))
			returns = np.zeros(shape=(self.env.height, self.env.width, 3, 3, self.n_actions, 1))
			value = 0

			for i, step_info in enumerate(reversed(timeline)):
				reward = rewards[i]
				state = step_info[0]
				action = step_info[1]

				value = self.gamma * value + reward

				found = list(filter(lambda s: np.array_equal(step_info, s), timeline[:len(timeline) - i - 1]))
				if len(found) == 0:
					if returns[state[0]][state[1]][state[2]][state[3]][action] == 0:
						returns[state[0]][state[1]][state[2]][state[3]][action] = value
					else:
						np.append(returns[state[0]][state[1]][state[2]][state[3]][action], [value])
					q[state[0]][state[1]][state[2]][state[3]][action] = np.mean(returns[state[0]][state[1]][state[2]][state[3]][action])

					max_q = np.max(q[state[0]][state[1]][state[2]][state[3]])
					best_actions = np.where(q[state[0]][state[1]][state[2]][state[3]] == max_q)[0]
					best_action = np.random.choice(best_actions)
				
					for curr_action in range(self.n_actions):
						if curr_action == best_action:
							self.policy[state[0]][state[1]][state[2]][state[3]][curr_action] = 1 - self.curr_epsilon + self.curr_epsilon / self.n_actions
						else:
							self.policy[state[0]][state[1]][state[2]][state[3]][curr_action] = self.curr_epsilon / self.n_actions

			if plot_training: 
				self.plot_rewards()
			
			print(f'Episode {i_episode} complete with epsilon {self.curr_epsilon}')

		print('Complete')
		self.plot_rewards(show_result=True, path=os.path.join(path, "rewards.png"))


if __name__ == "__main__":
	path = os.path.join("plots", "Monte_Carlo")

	try:  
		os.mkdir("plots")
	except:  
		print("Path already exists")
	try:  
		os.mkdir(path)
	except:  
		print("Path already exists")

	episodes = 5000
	timelimit = 1000

	env_text = "grid_simple"
	env = Environment(f'./grids/{env_text}.txt', timelimit=timelimit)    
	path = os.path.join(path, env_text)
	try:  
		os.mkdir(path)
	except:  
		print("Path already exists")    
	agent = Agent(env)
	agent.training(episodes, plot_training=True, max_steps=timelimit + 1, path=path)
	agent.env.test_start_positions(agent.get_test_action, path=path)