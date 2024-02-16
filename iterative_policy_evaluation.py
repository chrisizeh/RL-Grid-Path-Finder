import numpy as np

def prob(state, old_state, action):
	if old_state == 0:
		return 0

	elif old_state == 14 and action == "right":
		if state == 0:
			return 1
		return 0

	elif old_state == 13 and action == "down":
		if state == 15:
			return 1

	elif old_state == 11 and action == "down":
		if state == 0:
			return 1

	elif old_state == 15:
		if state == 12 and action == "left":
			return 1
		if state == 13 and action == "up":
			return 1
		if state == 14 and action == "right":
			return 1
		if state == 15 and action == "down":
			return 1

	elif old_state % 4 == 0 and action == "left":
		if old_state == state:
			return 1
		return 0

	elif old_state % 4 == 3 and action == "right":
		if old_state == state:
			return 1

	elif old_state < 4 and action == "up":
		if old_state == state:
			return 1

	elif old_state > 11 and action == "down":
		if old_state == state:
			return 1

	elif action == "left":
		if state == old_state - 1:
			return 1
	elif action == "up":
		if state == old_state - 4:
			return 1
	elif action == "right":
		if state == old_state + 1:
			return 1
	elif action == "down":
		if state == old_state + 4:
			return 1
	return 0

theta = 0.1

keys = range(16)
values = np.ones(16)	
v = dict(zip(keys, values))
v[0] = 0

delta = 1
while delta > theta:
	delta = 0
	for s in keys:
		value = v[s]
		sum = 0
		count = 0
		for a in ["left", "right", "up", "down"]:
			inner_sum = 0
			for new_s in keys:
				inner_sum += prob(new_s, s, a) * (-1 + v[new_s])
			sum += 0.25 * inner_sum
		v[s] = sum
		if abs(value - v[s]) > delta:
			delta = abs(value - v[s])
			
print(v)
