import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from MPPIControllerGPU import MPPIController
from VehicleModel import VehicleModel
import io

# Utility functions
def plot_obstacles(obstacles):
    for obs in obstacles:
        r = obs[2]
        rect = Rectangle((obs[0] - r, obs[1] - r), 2 * r, 2 * r, linewidth=0, edgecolor='none', facecolor='k', zorder=1)
        plt.gca().add_patch(rect)

def plot_state(state, style):
    # Convert state tensor to CPU if it is on GPU
    if state.device.type == 'cuda':
        state_cpu = state.cpu().numpy()
    else:
        state_cpu = state.numpy()

    x, y, phi = state_cpu[0], state_cpu[1], state_cpu[2]

    plt.plot(x, y, style)

    # Convert phi to a tensor if it's a float
    phi_tensor = torch.tensor(phi, dtype=torch.float32) if isinstance(phi, (float, np.float32)) else phi

    # Calculate delta_x and delta_y
    delta_x = torch.cos(phi_tensor).item() * 0.5
    delta_y = torch.sin(phi_tensor).item() * 0.5

    # Use delta_x and delta_y in plt.quiver
    plt.quiver(x, y, delta_x, delta_y)

# Param definition
n_samples = 1000    # Number of rollout trajectories
horizon = 25       # Prediction horizon represented as number of steps
lambda_ = 10       # Temperature - Selectiveness of trajectories by their costs
nu = 500           # Exploration variance
R = torch.diag(torch.tensor([1, 5], dtype=torch.float32))  # Control weighting matrix
cov = [1, 0.4]     # Variance of control inputs disturbance 
dt = 0.1           # Time step of controller and simulation

# Initialize state and goal
init_state = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)  # x, y, phi, v, steer
goal_state = torch.tensor([6, 6, 0], dtype=torch.float32)

# Define environment - obstacles [x, y, radius]
n_obstacles = 10
obstacles = torch.cat((torch.rand(n_obstacles, 2) * 4 + 1, 0.2 * torch.ones(n_obstacles, 1)), dim=1)

# Init
car_real = VehicleModel()
car = VehicleModel()
controller = MPPIController(lambda_, cov, nu, R, horizon, n_samples, car, dt, goal_state, obstacles)

# Prepare visualization
fig, ax = plt.subplots()
ax.set_xlim([-0.5 + min(init_state[0], goal_state[0]), 0.5 + max(init_state[0], goal_state[0])])
ax.set_ylim([-0.5 + min(init_state[1], goal_state[1]), 0.5 + max(init_state[1], goal_state[1])])
plot_state(init_state, 'bo')
plot_state(goal_state, 'ro')
plot_obstacles(obstacles.cpu().numpy())  # Convert to NumPy for plotting

# Initialize state
car_state = init_state

# Update function for animation
def update(frame):
    global car_state
    ax.clear()
    plot_obstacles(obstacles.cpu().numpy())  # Redraw obstacles
    action = controller.get_action(car_state)
    controller.plot_rollouts(fig)
    
    car_state = car_real.step(action, dt, car_state)
    plot_state(car_state, 'go')

# Create animation
ani = FuncAnimation(fig, update, frames=100, repeat=False, interval=100)

# Save animation as GIF
ani.save('animation.gif', writer='pillow', fps=10)

plt.show()
