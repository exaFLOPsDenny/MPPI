import torch
import matplotlib.pyplot as plt

class MPPIController:
    def __init__(self, lambda_, cov, nu, R, horizon, n_samples, model, dt, goal, obstacles):
        self.lambda_ = lambda_
        self.horizon = horizon
        self.n_samples = n_samples
        self.cov = cov
        self.nu = nu
        self.R = torch.tensor(R, device='cuda')  # Move R to GPU
        self.model = model
        self.dt = dt
        self.goal = torch.tensor(goal, device='cuda')  # Move goal to GPU
        self.obstacles = torch.tensor(obstacles, device='cuda') if obstacles is not None else None

        self.control_sequence = torch.zeros((2, self.horizon), device='cuda')
        self.optimized_control_sequence = None

        self.state = None
        self.n_states = 5

        self.rollouts_states = None
        self.rollouts_costs = None
        self.rollouts_plot_handle = []

        self.max_vel = 5
        self.max_steer = 0.5

    def get_action(self, state):
        # Convert state to tensor and move to GPU
        self.state = torch.tensor(state, device='cuda', dtype=torch.float32)
        init_state = torch.tensor(state, device='cuda', dtype=torch.float32)

        # Initialize states and costs as tensors on GPU
        states = torch.zeros((self.n_states, self.horizon + 1), device='cuda')
        S = torch.zeros(self.n_samples, device='cuda')

        self.rollouts_states = torch.zeros((self.n_samples, self.horizon + 1, self.n_states), device='cuda')
        self.rollouts_costs = torch.zeros(self.n_samples, device='cuda')

        # Generate random control input disturbances as tensors on GPU
        delta_u_vel = torch.normal(0, self.cov[0], (self.n_samples, self.horizon), device='cuda')
        delta_u_steer = torch.normal(0, self.cov[1], (self.n_samples, self.horizon), device='cuda')
        delta_u_steer = torch.clamp(delta_u_steer, -0.5, 0.5)

        delta_u = torch.stack((delta_u_vel, delta_u_steer), dim=0)

        # Simulate rollouts in parallel
        for k in range(self.n_samples):
            states[:, 0] = init_state

            for i in range(self.horizon):
                # Ensure both control_sequence and delta_u are tensors on GPU
                states[:, i + 1] = self.model.step(
                    self.control_sequence[:, i] + delta_u[:, k, i],
                    self.dt,
                    states[:, i]
                )
                S[k] += self.cost_function(states[:, i + 1], self.control_sequence[:, i], delta_u[:, k, i])

            self.rollouts_states[k, :, :] = states.T
            self.rollouts_costs[k] = S[k]

        # Normalize costs and update control sequence
        S_normalized = S - torch.min(S)
        for i in range(self.horizon):
            self.control_sequence[:, i] += self.total_entropy(delta_u[:, :, i].T, S_normalized)

        # Saturate control output
        self.control_sequence[0, self.control_sequence[0, :] > self.max_vel] = self.max_vel
        self.control_sequence[0, self.control_sequence[0, :] < -self.max_vel] = -self.max_vel

        self.control_sequence[1, self.control_sequence[1, :] > self.max_steer] = self.max_steer
        self.control_sequence[1, self.control_sequence[1, :] < -self.max_steer] = -self.max_steer

        # Select control action
        self.optimized_control_sequence = self.control_sequence.clone()
        action = self.control_sequence[:, 0]

        # Shift the control sequence and append a zero column
        self.control_sequence = torch.cat((self.control_sequence[:, 1:], torch.zeros((2, 1), device='cuda')), dim=1)

        return action.cpu().numpy()  # Convert back to NumPy for output

    def cost_function(self, state, u, du):
        state_cost = self.state_cost_function(state)
        control_cost = self.control_cost_function(u, du)
        return state_cost + control_cost

    def state_cost_function(self, state):
        obstacle_cost = self.obstacle_cost_function(state)
        heading_cost = self.heading_cost_function(state)
        distance_cost = self.distance_cost_function(state)
        return distance_cost + heading_cost + obstacle_cost

    def distance_cost_function(self, state):
        weight = 100
        cost = weight * torch.dot((self.goal[0:2] - state[0:2]), (self.goal[0:2] - state[0:2]))
        return cost

    def heading_cost_function(self, state):
        weight = 50
        pow_ = 2
        cost = weight * torch.abs(self.get_angle_diff( self.goal[2], state[2] ))**pow_
        return cost

    def control_cost_function(self, u, du):
        cost = (1 - 1 / self.nu) / 2 * torch.matmul(du.T, torch.matmul(self.R, du)) + torch.matmul(u.T, torch.matmul(self.R, du)) + 0.5 * torch.matmul(u.T, torch.matmul(self.R, u))
        return cost

    def obstacle_cost_function(self, state):
        if self.obstacles is None or len(self.obstacles) == 0:
            return 0

        distance_to_obstacle = torch.norm(state[0:2] - self.obstacles[:, 0:2], dim=1)
        min_dist, min_dist_idx = torch.min(distance_to_obstacle, dim=0)

        hit = 1 if min_dist <= self.obstacles[min_dist_idx, 2] else 0

        obstacle_cost = 550 * torch.exp(-min_dist / 5) + 1e6 * hit
        return obstacle_cost

    def total_entropy(self, du, trajectory_cost):
        exponents = torch.exp(-1 / self.lambda_ * trajectory_cost)
        value = torch.sum(exponents[:, None] * du, axis=0) / torch.sum(exponents)
        return value
    
    def plot_rollouts(self, fig):
        # Remove old plot handles if they exist
        if self.rollouts_plot_handle:
            for handle in self.rollouts_plot_handle:
                handle.remove()
            self.rollouts_plot_handle = []

        plt.figure(fig.number)

        # Normalize costs for coloring trajectories
        rollouts_costs_cpu = self.rollouts_costs.cpu()  # Move data to CPU for normalization
        costs = (rollouts_costs_cpu - torch.min(rollouts_costs_cpu)) / (torch.max(rollouts_costs_cpu) - torch.min(rollouts_costs_cpu))
        min_idx = torch.argmin(rollouts_costs_cpu).item()

        rollouts_states_cpu = self.rollouts_states.cpu()  # Move rollout states to CPU for plotting

        # Plot all rollouts
        for i in range(self.n_samples):
            if i == min_idx:
                color = [0, 1, 1]  # Cyan for the best trajectory
            else:
                color = [1 - costs[i].item(), 0, 0.2]  # Color based on normalized cost

            # Convert tensors to NumPy arrays for plotting
            line, = plt.plot(rollouts_states_cpu[i, :, 0].numpy(), rollouts_states_cpu[i, :, 1].numpy(), '-', color=color)
            self.rollouts_plot_handle.append(line)

        # Plot the rollout of the selected trajectory
        states = torch.zeros((self.n_states, self.horizon + 1), device='cuda')
        states[:, 0] = self.state

        for i in range(self.horizon):
            states[:, i + 1] = self.model.step(self.optimized_control_sequence[:, i], self.dt, states[:, i])

        states_cpu = states.cpu()  # Move states to CPU for plotting
        line, = plt.plot(states_cpu[0, :].numpy(), states_cpu[1, :].numpy(), '--', color=[0, 1, 0])  # Plot the selected trajectory in green
        self.rollouts_plot_handle.append(line)


    def get_angle_diff(self, angle1, angle2):
        angle_diff = angle1 - angle2
        angle = torch.remainder(angle_diff + torch.pi, 2 * torch.pi) - torch.pi
        return angle
