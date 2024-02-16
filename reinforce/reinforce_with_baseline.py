import numpy as np
import gym
import time
import os

def basis_vectors(state, action, theta, action_number):
	if np.linalg.norm(state) != 0:
		normed_state = np.append(np.transpose(state) / np.linalg.norm(state), [action / (action_number - 1)])
	else:
		normed_state = np.append(np.transpose(state), [action / (action_number - 1)])

	x = np.zeros(len(theta))
	c = np.ones(len(normed_state))


	for i in range(len(theta)):
		x[i] = np.cos(np.pi * np.dot(normed_state, c ** i))

	return x

def v_function(state, weigths, action_number):
	arr = []
	for i in range(action_number):
		arr.append(h(state, i, weigths, action_number))

	return np.sum(arr)


def h(state, action, theta, action_number):
	return np.dot(np.transpose(theta), basis_vectors(state, action, theta, action_number))

def policy(state, theta, action_number):
	arr = []
	for i in range(action_number):
		arr.append(h(state, i, theta, action_number))

	arr = np.exp(arr - max(arr))
	sum = np.sum(arr)
	arr /= sum

	return arr

def get_policy_action(state, theta, action_number):
	policy_array = policy(state, theta, action_number)
	return np.random.choice(np.flatnonzero(policy_array == policy_array.max()))

def eligibilty_vector(state, action, theta, action_number):
	sum = 0
	policy_array = policy(state, theta, action_number)
	for i in range(action_number):
		sum += policy_array[i] * basis_vectors(state, i, theta, action_number)

	return basis_vectors(state, action, theta, action_number) - sum

def run(env, alpha_theta, alpha_w, gamma, theta_length, weigths_length, episodes, max_steps, theta = None, weigths = None, logging = False, render = False):
	env = gym.make(env)
	steps = []

	if theta is None:
		if logging: print("Theta initialised")
		theta = np.zeros(theta_length)

	if weigths is None:
		if logging: print("Weigths initialised")
		weigths = np.zeros(weigths_length)	


	for episode in range(episodes):
		state = env.reset()

		rewards = []
		states = []
		actions = []

		for step in range(max_steps):

			if isinstance(state, list):
				action = get_policy_action(state, theta, env.action_space.n)
				states.append(state)
			else:
				action = get_policy_action([state], theta, env.action_space.n)
				states.append([state])

			state, reward, done, info = env.step(action)

			rewards.append(reward)
			actions.append(action)

			if render: env.render()

			if done:
				steps.append(step + 1)
				if logging: print("Episode {} finished after {} timesteps".format(episode, step + 1))	
				break

		for t in range(step):
			return_value = np.sum(rewards[t + 1:] * gamma ** np.arange(step - t))
			delta = return_value - v_function(states[t], weigths, env.action_space.n)

			weigths += alpha_w * delta * basis_vectors(states[t], actions[t], weigths, env.action_space.n) 
			
			policy_array = policy(states[t], theta, env.action_space.n)
			theta += alpha_theta * gamma ** t * delta * eligibilty_vector(states[t], actions[t], theta, env.action_space.n)
	env.close()		

	return steps, theta, weigths

if __name__ == "__main__":

	epochs = 500

	# Uncomment if existing weigths should be used
	# theta = np.load("reinforce_w_b_theta.npy")
	# weigths = np.load("reinforce_w_b_weigths.npy")

	theta = None
	weigths = None

	for i in range(epochs):
		# runn alghorithm for 100 episodes with alpha = 0.1, gamma = 0.3, theta and wheigts vector of 20, max step size 500
		# To enable the rendering set render = True
		# To enble the logging set logging = True
		# steps, theta, weigths = run('CartPole-v1', 0.1, 0.1, 0.3, 20, 20, 100, 500, theta = theta, weigths = weigths, logging = False, render = False)
		
		# Uncomment to test second szenario with best combination according to tests
		steps, theta, weigths = run('FrozenLake8x8-v0', 0.1, 0.1, 0.8, 20, 20, 100, 500, theta = theta, weigths = weigths, logging = False, render = False)

		print("Epoch: ", i)
		print("Last episode length: ", steps[-1])
		print("Average episode length: ", np.mean(steps))
		print("Variance of episode length: ", np.var(steps), "\n")

	# Uncomment if weights should be saved
	np.save("reinforce_w_b_frozen_weigths", weigths)
	np.save("reinforce_w_b_frozen_theta", theta)

