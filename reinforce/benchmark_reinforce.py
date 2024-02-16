import reinforce_algorithm
import reinforce_with_baseline
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

sns.set()

episodes = 10 #50
runs = 10 #100

alphas = [0.01, 0.1, 0.5, 0.7, 0.9]
gammas = [0.99, 0.8, 0.6, 0.3, 0.1]
theta_lengths = [2, 4, 8, 10, 20]

res = []

for alpha in alphas:
	for gamma in gammas:
		for theta_length in theta_lengths:
			print("alpha: %f, gamma: %f, theta_length: %d" % (alpha, gamma, theta_length))

			all_steps = []
			for run in range(runs):

				# Uncomment to benchmark reinforce algorithm with FrozenLake8x8
				steps, theta = reinforce_algorithm.run('FrozenLake8x8-v0', alpha, gamma, theta_length, episodes, 500)

				# Uncomment to benchmark reinforce algorithm with CartPole-v1
				# steps, theta, weigths = reinforce_with_baseline.run('FrozenLake8x8-v0', alpha, alpha, gamma, theta_length, theta_length, episodes, 500)
				
				all_steps.append(steps)
			res.append({"alpha": alpha, "gamma":gamma, "theta": theta_length, "steps_mean": np.mean(all_steps), "steps_var": np.var(all_steps)})

# plot the average steps
data = pd.DataFrame.from_records(res)

print(data.head)
plt.figure(1)
sns.lineplot(data=data, x="alpha", y="steps_mean", hue="gamma", style="theta")
plt.figure(2)
sns.lineplot(data=data, x="gamma", y="steps_mean", hue="theta", style="alpha")
plt.figure(3)
sns.lineplot(data=data, x="theta", y="steps_mean", hue="gamma", style="alpha")

plt.figure(4)
sns.lineplot(data=data, x="alpha", y="steps_var", hue="gamma", style="theta")
plt.figure(5)
sns.lineplot(data=data, x="gamma", y="steps_var", hue="theta", style="alpha")
plt.figure(6)
sns.lineplot(data=data, x="theta", y="steps_var", hue="gamma", style="alpha")

plt.show()