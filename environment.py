import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import random
from math import ceil
import os

class Environment:

	'''
	Grid:
	0 => border or obstacles
	1 => moveable positions
	2 => start
	3 => end

	Position: (y, x) for direct matrix checking
	'''
	def __init__(self, gridPath, timelimit= -1, startIndex=None) -> None:
		self.n_actions = 9
		self.n_states = 4

		self.timelimit = timelimit
		self.grid = np.loadtxt(gridPath)
		self.start_line = np.where(self.grid == 2)
		self.height, self.width = self.grid.shape

		end_points = np.where(self.grid == 3)
		if(len(end_points[0]) < 1):
			raise "At least one endpoint must be provided"

		self.history = []
		self.colormap = colors.ListedColormap(["black", "white", "blue", "green", "turquoise", "red", "orange"])

		self.actions = np.array([[1, -1], [1, 0], [1, 1], [0, -1], [0, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1]], dtype=np.int64)
		self.time = 0
		self.reset_pos(startIndex)

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
		self.time += 1

		if(self.timelimit != -1 and self.time > self.timelimit):
			truncated = True 

		speedChange = self.actions[action]

		self.speed[0] = max(-2, min(2, self.speed[0] + speedChange[0]))
		self.speed[1] = max(-2, min(2, self.speed[1] + speedChange[1]))

		if(self.speed[0] == 0 and self.speed[1] == 0):
			self.speed[random.choice([0, 1])] = 1
	
		if (not self.move_one_step()):
			self.reset_pos()
		elif (self.grid[self.pos[0]][self.pos[1]] == 3):
			terminated = True
			reward = 0

		return [*self.pos, *self.speed], reward, terminated, truncated, self.speed


	'''
	Reseting environment

	Returns:
	observation: Current position as state
	info: absolute speeds for x and y axis
	'''
	def reset(self, startIndex=None) -> tuple[list[float], list[int]]:
		self.reset_pos(startIndex)
		self.history = []
		self.time = 0

		return [*self.pos, *self.speed], self.speed
	

	'''
	Print the grid with the path until this timestep
	'''
	def print(self, path=None) -> None:
		print_grid = self.grid.copy()

		for pos in range(1, len(self.history)):
			print_grid[self.history[pos][0], self.history[pos][1]] = 6

		print_grid[self.pos[0], self.pos[1]] = 5
		print_grid[self.history[0][0], self.history[0][1]] = 4
		plt.figure(figsize=(self.width, self.height))
		plt.imshow(print_grid, cmap=self.colormap, interpolation='none')

		if(path):
			plt.savefig(path)
			plt.clf()
		else:
			plt.show()	

	'''
	Reset position to random choice on starting line and speed to zero
	'''
	def reset_pos(self, startIndex=None) -> None:
		if startIndex != None:
			index = startIndex
		else:
			index = random.choice(range(len(self.start_line[0])))

		self.pos = np.array([self.start_line[0][index], self.start_line[1][index]])
		self.speed = np.zeros(2, dtype=np.int64)
		self.history.append(self.pos.copy())


	'''
	Find a path with the defined x and y speed without obstacles on the grid
	'''
	def move_one_step(self) -> bool:
		y = self.speed[0]
		x = self.speed[1]

		while (x != 0 or y != 0):
			self.history.append(self.pos.copy())

			max_speed = max(abs(y), abs(x))
			next_x = ceil(x / max_speed)
			next_y = ceil(y /max_speed)
			try:
				if ((self.pos[0] - next_y) > 0 and (self.pos[1] + next_x) > 0 and self.grid[self.pos[0] - next_y][self.pos[1] + next_x] != 0):
					self.pos[1] += next_x
					self.pos[0] -= next_y
					x -= next_x
					y -= next_y

					if(self.grid[self.pos[0]][self.pos[1]] == 3):
						return True
				else:
					return False
			except:
				return False
				
		return True
	

	def test_start_positions(self, get_action, path=None):
		for startIndex in range(len(self.start_line[0])):
			state, _ = self.reset(startIndex=startIndex)

			done = False
			while not done:
				action = get_action(state)
				observation, reward, terminated, truncated, _ = self.step(action)
				done = terminated or truncated

			print(f"Start Point {startIndex} took {self.time} steps")
			self.print(os.path.join(path, f"start_position_{startIndex}.png"))
		


if __name__ == "__main__":
	env = Environment('./grids/grid_simple.txt')
	env.reset(1)
	env.speed = [1, 2]
	print(env.grid)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	env.print()

	env.reset(1)
	env.speed = [2, 2]
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)

	pos, reward, terminated, truncated, info = env.step(6)
	print(pos, reward, terminated, truncated, info)

	pos, reward, terminated, truncated, info = env.step(3)
	print(pos, reward, terminated, truncated, info)

	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)

	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	env.print()