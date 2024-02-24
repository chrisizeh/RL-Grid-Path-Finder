import numpy as np
import matplotlib.pyplot as plt
import random
import os

from IPython import display
from itertools import count
import torch

from environment import Environment

class Agent():

	def __init__(self, env, epsilon=0.05, gamma=0.99, max_steps=1000) -> None:
		self.env = env
		self.n_actions = env.n_actions
		self.n_observations = env.n_states

		self.epsilon = epsilon
		self.gamma = gamma
		self.max_steps = max_steps
		
		self.policy = np.full((self.env.height, self.env.width, 3, 3, self.n_actions), 1 / self.n_actions)
		self.times_counted = np.zeros((self.env.height, self.env.width, 3, 3, self.n_actions))
		self.q = np.full((self.env.height, self.env.width, 3, 3, self.n_actions), -self.max_steps, dtype=np.float32)

		self.steps = 0


	def select_action(self, state):
		sample = random.random()
		
		if sample > self.epsilon:
			max_value = np.max(self.policy[state[0]][state[1]][state[2]][state[3]])
			best_actions = np.where(self.policy[state[0]][state[1]][state[2]][state[3]] == max_value)[0]
			best_action = np.random.choice(best_actions)
			return best_action
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
 

	def run(self, episodes, plot=True, path=None, training=True):
		plt.ion()
		self.episode_rewards = []

		if (not training):
				self.epsilon = 0

		for i_episode in range(episodes):
			rewards = np.zeros(self.max_steps)
			timeline = np.empty(self.max_steps, dtype=tuple)
			state, _ = self.env.reset(i_episode % self.env.n_starts)

			for t in count():
				action = self.select_action(state)
				timeline[t] = (state, action)
				state, reward, terminated, truncated, _ = self.env.step(action)
				rewards[t] = reward

				done = terminated or truncated
				if done or t > self.max_steps:
					self.episode_rewards.append(np.sum(rewards))
					timeline = timeline[:t]
					rewards = rewards[:t]
					break

			if (training):
				value = 0
				for i, step_info in enumerate(reversed(timeline)):
					reward = rewards[i]
					state = step_info[0]
					action = step_info[1]

					value = self.gamma * value + reward

					found = list(filter(lambda s: np.array_equal(step_info, s), timeline[:len(timeline) - i - 1]))
					if len(found) == 0:
						self.times_counted[state[0]][state[1]][state[2]][state[3]][action] += 1
						self.q[state[0]][state[1]][state[2]][state[3]][action] += (value - self.q[state[0]][state[1]][state[2]][state[3]][action]) / self.times_counted[state[0]][state[1]][state[2]][state[3]][action]

						max_q = np.max(self.q[state[0]][state[1]][state[2]][state[3]])
						best_actions = np.where(self.q[state[0]][state[1]][state[2]][state[3]] == max_q)[0]
						best_action = np.random.choice(best_actions)
					
						for curr_action in range(self.n_actions):
							if curr_action == best_action:
								self.policy[state[0]][state[1]][state[2]][state[3]][curr_action] = 1 - self.epsilon + self.epsilon / self.n_actions
							else:
								self.policy[state[0]][state[1]][state[2]][state[3]][curr_action] = self.epsilon / self.n_actions

				if plot: 
					self.plot_rewards()
			
			else:
				if plot: 
					self.env.print(os.path.join(path, f"episode_{i_episode}.png"))

			print(f'Episode {i_episode} complete after {t} steps')

		print('Complete')
		if (training):
			self.plot_rewards(show_result=True, path=os.path.join(path, "rewards.png"))
		plt.ioff()


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

	episodes = 1000
	timelimit = 2000

	env_text = "grid_simple"
	env = Environment(f'./grids/{env_text}.txt', timelimit=timelimit)    
	path = os.path.join(path, env_text)
	try:  
		os.mkdir(path)
	except:  
		print("Path already exists")    
	agent = Agent(env, max_steps=timelimit + 1)
	agent.run(episodes, plot=True, path=path)
	agent.run(10, plot=True, path=path, training=False)
	plt.close()