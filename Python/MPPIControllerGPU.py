import cupy as np  # Replace numpy with cupy
import matplotlib.pyplot as plt

class MPPIController:
    def __init__(self, lambda_, cov, nu, R, horizon, n_samples, model, dt, goal, obstacles):
        self.lambda_ = lambda_
        self.horizon = horizon
        self.n_samples = n_samples
        self.cov = cov
        self.nu = nu
        self.R = R
        self.model = model
        self.dt = dt
        self.goal = goal
        self.obstacles = obstacles

        self.control_sequence = np.zeros((2, self.horizon))
        self.optimized_control_sequence = None

        self.state = None
        self.n_states = 5

        self.rollouts_states = None
        self.rollouts_costs = None
        self.rollouts_plot_handle = []

        self.max_vel = 5
        self.max_steer = 0.5

def get_action(self, state):
    # Initialize tensors on GPU
    self.state = torch.tensor(state, device='cuda')
    init_state = torch.tensor(state, device='cuda')
    
    states = torch.zeros((self.n_states, self.horizon + 1), device='cuda')
    S = torch.zeros(self.n_samples, device='cuda')

    self.rollouts_states = torch.zeros((self.n_samples, self.horizon + 1, self.n_states), device='cuda')
    self.rollouts_costs = torch.zeros(self.n_samples, device='cuda')

    # Generate random control input disturbances on GPU
    delta_u_vel = torch.normal(0, self.cov[0], (self.n_samples, self.horizon), device='cuda')
    delta_u_steer = torch.normal(0, self.cov[1], (self.n_samples, self.horizon), device='cuda')
    delta_u_steer = torch.clamp(delta_u_steer, -0.5, 0.5)

    delta_u = torch.stack((delta_u_vel, delta_u_steer), dim=0)

    # Simulate rollouts in parallel
    for k in range(self.n_samples):
        states[:, 0] = init_state
        for i in range(self.horizon):
            states[:, i + 1] = self.model.step(self.control_sequence[:, i] + delta_u[:, k, i], self.dt, states[:, i])
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
    self.control_sequence = torch.cat((self.control_sequence[:, 1:], torch.zeros((2, 1), device='cuda')), dim=1)

    return action.cpu().numpy()  # Returning the result as numpy array for further processing

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
        cost = weight * np.dot((self.goal[0:2] - state[0:2]), (self.goal[0:2] - state[0:2]))
        return cost

    def heading_cost_function(self, state):
        weight = 50
        pow_ = 2
        cost = weight * abs(self.get_angle_diff(self.goal[2], state[2]))**pow_
        return cost

    def control_cost_function(self, u, du):
        cost = (1 - 1 / self.nu) / 2 * np.dot(du.T, np.dot(self.R, du)) + np.dot(u.T, np.dot(self.R, du)) + 0.5 * np.dot(u.T, np.dot(self.R, u))
        return cost

    def obstacle_cost_function(self, state):
        if self.obstacles is None or len(self.obstacles) == 0:
            return 0

        distance_to_obstacle = np.linalg.norm(state[0:2] - self.obstacles[:, 0:2], axis=1)
        min_dist = np.min(distance_to_obstacle)
        min_dist_idx = np.argmin(distance_to_obstacle)

        hit = 1 if min_dist <= self.obstacles[min_dist_idx, 2] else 0

        obstacle_cost = 550 * np.exp(-min_dist / 5) + 1e6 * hit
        return obstacle_cost

    def total_entropy(self, du, trajectory_cost):
        exponents = np.exp(-1 / self.lambda_ * trajectory_cost)
        value = np.sum(exponents[:, None] * du, axis=0) / np.sum(exponents)
        return value

    def plot_rollouts(self, fig):
        if self.rollouts_plot_handle:
            for handle in self.rollouts_plot_handle:
                handle.remove()
            self.rollouts_plot_handle = []

        plt.figure(fig.number)
        costs = (self.rollouts_costs - np.min(self.rollouts_costs)) / (np.max(self.rollouts_costs) - np.min(self.rollouts_costs))
        min_idx = np.argmin(costs)

        for i in range(self.n_samples):
            if i == min_idx:
                color = [0, 1, 1]
            else:
                color = [1 - costs[i], 0, 0.2]

            line, = plt.plot(self.rollouts_states[i, :, 0], self.rollouts_states[i, :, 1], '-', color=color)
            self.rollouts_plot_handle.append(line)

        # Rollout of selected trajectory
        states = np.zeros((self.n_states, self.horizon + 1))
        states[:, 0] = self.state

        for i in range(self.horizon):
            states[:, i + 1] = self.model.step(self.optimized_control_sequence[:, i], self.dt, states[:, i])

        line, = plt.plot(states[0, :], states[1, :], '--', color=[0, 1, 0])
        self.rollouts_plot_handle.append(line)

    @staticmethod
    def get_angle_diff(angle1, angle2):
        angle_diff = angle1 - angle2
        angle = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi
        return angle
