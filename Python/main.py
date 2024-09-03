# MIT License
# 
# Copyright (c) 2023 Roman Adámek
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Utility functions

def plot_obstacles(obstacles):
    for obs in obstacles:
        r = obs[2]
        pos = [obs[0] - r, obs[1] - r, 2 * r, 2 * r]
        rect = Rectangle((obs[0] - r, obs[1] - r), 2 * r, 2 * r, linewidth=0, edgecolor='none', facecolor='k', zorder=1)
        plt.gca().add_patch(rect)

def plot_state(state, style):
    x, y, phi = state[0], state[1], state[2]
    plt.plot(x, y, style)
    delta_x, delta_y = np.cos(phi) * 0.5, np.sin(phi) * 0.5
    plt.quiver(x, y, delta_x, delta_y)

# Placeholder classes for VehicleModel and MPPIController
class VehicleModel:
    def step(self, action, dt, car_state):
        # Dummy implementation - replace with actual vehicle model
        new_state = car_state + action * dt
        return new_state

class MPPIController:
    def __init__(self, lambda_, cov, nu, R, horizon, n_samples, car, dt, goal_state, obstacles):
        # Initialize with given parameters
        self.lambda_ = lambda_
        self.cov = cov
        self.nu = nu
        self.R = R
        self.horizon = horizon
        self.n_samples = n_samples
        self.car = car
        self.dt = dt
        self.goal_state = goal_state
        self.obstacles = obstacles

    def get_action(self, car_state):
        # Dummy action generation - replace with actual controller logic
        return np.random.randn(len(car_state)) * 0.1

    def plot_rollouts(self, fig):
        # Dummy implementation for plotting rollouts
        pass

# Param definition
n_samples = 400    # Number of rollout trajectories
horizon = 25       # Prediction horizon represented as number of steps
lambda_ = 10       # Temperature - Selectiveness of trajectories by their costs
nu = 500           # Exploration variance
R = np.diag([1, 5])# Control weighting matrix
cov = [1, 0.4]     # Variance of control inputs disturbance 
dt = 0.1           # Time step of controller and simulation

init_state = np.array([0, 0, 0, 0, 0])  # x, y, phi, v, steer
goal_state = np.array([6, 6, 0])

# Define environment - obstacles [x, y, radius]
n_obstacles = 40
obstacles = np.hstack((np.random.rand(n_obstacles, 2) * 4 + 1, 0.2 * np.ones((n_obstacles, 1))))

# Init
car_real = VehicleModel()
car = VehicleModel()
controller = MPPIController(lambda_, cov, nu, R, horizon, n_samples, car, dt, goal_state, obstacles)

# Prepare visualization
plt.figure()
plt.axis('equal')
plt.xlim([-0.5 + min(init_state[0], goal_state[0]), 0.5 + max(init_state[0], goal_state[0])])
plt.ylim([-0.5 + min(init_state[1], goal_state[1]), 0.5 + max(init_state[1], goal_state[1])])
plot_state(init_state, 'bo')
plot_state(goal_state, 'ro')
plot_obstacles(obstacles)

# Control
car_state = init_state

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# Initialize list to store frame filenames
filenames = []

for i in range(100):
    action = controller.get_action(car_state)
    controller.plot_rollouts(plt.gcf())
    
    car_state = car_real.step(action, dt, car_state)
    plot_state(car_state, 'go')
    
    plt.draw()
    plt.pause(0.1)
    
    # Save the current frame as a PNG file
    filename = f'frame_{i}.png'
    plt.savefig(filename)
    filenames.append(filename)

# After the loop, compile all the PNGs into a GIF
with imageio.get_writer('Python/animation.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, remove the individual frame files to clean up
for filename in filenames:
    os.remove(filename)

plt.show()

