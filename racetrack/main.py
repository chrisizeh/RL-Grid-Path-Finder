import numpy as np
from io import StringIO 
import time
import os

clear = lambda: os.system('clear')

def load_grid(txt):
	return np.loadtxt(txt)

def print_grid(position, action, grid, policy):
	clear()
	print(policy[position[0], position[1]])
	print(action)
	print_grid = grid.copy()
	print_grid[position[0], position[1]] = '9'
	print(print_grid)
	time.sleep(0.5)

def run_episode(start, grid, actions, policy):
	speed = np.zeros(2, dtype=np.int64)
	start_line = np.where(grid[0] == 1)[0]
	timeline = []

	end = False
	state = start
	steps = 100000
	for step in range(steps):

		if(np.random.rand() >= 0.1):
			action = np.random.choice(9, p=policy[state[0], state[1]])
		else:
			action = 4

		speed = np.add(speed, actions[action], dtype=np.int64)
		if speed[0] > 5 or speed[0] < 0:
			speed[0] -= actions[action][0]

		if speed[1] > 5 or speed[1] < 0:
			speed[1] -= actions[action][1]

		if speed[0] == 0 and speed[1] == 0:
			speed[0] -= actions[action][0]
			speed[1] -= actions[action][1]

		reward = -1

		if(-1 in grid[state[0]:state[0] + speed[0] + 1, state[1]:state[1] + speed[1] + 1]):
			print('FINISH')
			break	

		state = np.add(state, speed)  

		if state[0] < 0 or state[1] < 0 or state[0] >= len(grid) or state[1] >= len(grid[0]):
			state = [0, np.random.choice(start_line)]
			speed = [0, 0]
		elif grid[state[0], state[1]] == 0:
			state = [0, np.random.choice(start_line)]
			speed = [0, 0]
		elif grid[state[0], state[1]] == -1:
			print('FINISH')
			break

		timeline.append({'state': state, 'action': action, 'reward': reward})

		# uncomment to see trackmovement
		# print_grid(state, actions[action], grid, policy)

	return np.array(timeline)
		


actions = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)

grid = load_grid('grid.txt')

# uncomment to run existing policy
# policy = np.load('0_5_policy.npy')
policy = np.full((grid.shape + (9,)), 1/9)
print(policy)

q = np.zeros((grid.shape + (9,)))
returns = np.zeros((grid.shape + (9, 1)))

epochs = 10
episodes = 1000
epsilon = 0.1

for epoch in range(epochs):

	q = np.zeros((grid.shape + (9,)))
	returns = np.zeros((grid.shape + (9, 1)))
	for episode in range(episodes):
		print('episode: ', episode)

		# uncomment to use exploring starts
		# y = np.random.randint(len(grid) - 1)
		# x_line = np.where(grid[y] == 1)[0]
		# x = np.random.choice(x_line)

		# comment to use exploring starts
		y = 0
		x = 5

		print('start position:', [y,x])
		timeline = run_episode([y, x], grid, actions, policy)

		value = 0

		for i, step in enumerate(reversed(timeline)):
			value = 1 * value + step['reward']

			if step not in timeline[:len(timeline) - i - 1]:
				if returns[step['state'][0], step['state'][1], step['action']] == 0:
					returns[step['state'][0], step['state'][1], step['action']] = [value]
				else:
					np.append(returns[step['state'][0], step['state'][1], step['action']], [value])
				q[step['state'][0], step['state'][1], step['action']] = np.mean(returns[step['state'][0], step['state'][1], step['action']])

				max_q = np.max(q[step['state'][0], step['state'][1]])
				best_actions = np.where(q[step['state'][0], step['state'][1]] == max_q)[0]
				best_action = np.random.choice(best_actions)
			
				for action in range(len(actions)):
					if action == best_action:
						policy[step['state'][0], step['state'][1], action] = 1 - epsilon + epsilon / len(actions)
					else:
						policy[step['state'][0], step['state'][1], action] = epsilon / len(actions)

		print('racetime: ', len(timeline), '\n')


np.save('0_5_policy_4', policy)