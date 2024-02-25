# Reinforcement Learning Algorithms for Grid Path Finding

This repository contains Python implementations of two reinforcement learning (RL) algorithms applied to a grid path finding environment. These implementations were developed as part of a university lecture in machine learning to explore the application of RL techniques in solving grid-based navigation problems.

## Environment Description

Consider a robot that starts at a designated starting line within a grid. The primary objective is for the robot to navigate to the target cell as quickly as possible, adhering to certain constraints:

* The robot's movement is constrained to a discrete set of grid positions.
* Velocity is also discrete, representing the number of grid cells moved horizontally and vertically per time step.
* Actions available to the robot are increments to the velocity components, allowing for movement in all directions.
* Each velocity component can be changed by +1, -1, or 0 in each step, resulting in a total of nine (3 x 3) possible actions.
* Both velocity components are restricted to be less than 3, and they cannot both be zero except at the starting line.

## Episode Dynamics

* Each episode begins with the robot positioned randomly along the starting line, with both velocity components initialized to zero.
* The episode terminates when the robot successfully reaches the target cell.
* The robot receives a reward of -1 for each step taken until reaching the target.
* If the robot collides with obstacles or walls, it is reset to a random position along the starting line, both velocity components are reduced to zero, and the episode continues.
* The episode also ends if the robot reaches the target cell.

## Implemented Algorithms

* First Visit Monte Carlo Control - Tabular algorithm
* Deep Q-Learner - Temporal Difference Off-Policy algorithm with neural network as approximator

## Usage

Grid problems are loaded using  `.txt` files in the `grids` folder. To run the a learning algorithm with a grid, set `env_text = "grid_simple"` to the filename.

## Results

Policies and training progress of already trained grids can be found in the `plots` folder.

<p>
  <img src="plots/Monte_Carlo/grid_race/episode_1.png?raw=true" width="29%" />
   <img src="plots/Monte_Carlo/grid_exercise/episode_7.png?raw=true" width="33%" />
  <img src="plots/Monte_Carlo/grid_maze_large/episode_2.png?raw=true" width="36%" />
</p>
