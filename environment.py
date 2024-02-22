import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import random
from math import ceil

class Environment:

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
		self.speed[0] = max(0, min(2, self.speed[0] + speedChange[0]))
		self.speed[1] = max(0, min(2, self.speed[1] + speedChange[1]))

		if(self.speed[0] == 0 and self.speed[1] == 0):
			self.speed[random.choice([0, 1])] = 1
	
		if (not self.move_one_step()):
			self.history.append(self.pos.copy())
			self.reset_pos()
		elif (self.pos == self.end):
			terminated = True
			reward = 0

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

		for pos in self.history:
			print_grid[pos[0], pos[1]] = 5

		print_grid[self.pos[0], self.pos[1]] = 4
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
		y = self.speed[0]
		x = self.speed[1]

		while (x != 0 or y != 0):
			self.history.append(self.pos.copy())

			max_speed = max(abs(y), abs(x))
			next_x = ceil(x / max_speed)
			next_y = ceil(y /max_speed)
			try:
				if (self.grid[self.pos[0] - next_y][self.pos[1] + next_x] != 0):
					self.pos[1] += next_x
					self.pos[0] -= next_y
					x -= next_x
					y -= next_y
				else:
					print("Reset: Hit an Obstacle")
					return False
			except:
				print("Reset: Hit the Border")
				return False
				
		return True
		


if __name__ == "__main__":
	env = Environment('./grids/grid_simple.txt')
	env.pos = [8, 1]
	env.speed = [1, 2]
	print(env.grid)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	pos, reward, terminated, truncated, info = env.step(4)
	print(pos, reward, terminated, truncated, info)
	env.print()

	env.reset()
	env.pos = [8, 1]
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