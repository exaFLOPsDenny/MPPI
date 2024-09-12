import torch

class oMMPI:
    def __init__(self, n_rollouts, time_steps, temperature_parameter):
        self.n_rollouts = n_rollouts # Notated as M in the paper
        self.time_steps = time_steps 
        self.lambda_ = temperature_parameter
        
        self.state = None
        self.n_states = 5 # 5가 5차식까지 고려해서인가?

    # (i) selection of the cost function
    def cost_function(self, state, u):
        pass
    
    # (ii) Sampling in the output space;
    def sampling_trajectory(self, state):
        
        # Convert state to tensor and move to GPU
        self.state = torch.tensor(state, device='cuda', dtype=torch.float32)
        init_state = torch.tensor(state, device='cuda', dtype=torch.float32)

        init_Y = torch.tensor(self.inverse_map(init_state), device='cuda', dtype=torch.float32)
        
        # Initialize states, costs and m^th Y_0 as tensors on GPU
        states = torch.zeros((self.n_states, self.time_steps + 1), device='cuda')
        S = torch.zeros(self.n_rollouts, device='cuda')

        self.rollouts_states = torch.zeros((self.n_rollouts, self.time_steps + 1, self.n_states), device='cuda')
        self.rollouts_costs = torch.zeros(self.n_rollouts, device='cuda')


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

    
    # (iii) Inversion to find the associated inputs
    def inverse_map(self):
        pass

