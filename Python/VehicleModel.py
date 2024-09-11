import torch

class VehicleModel:
    def __init__(self):
        self.m = 2000.0                # Mass (kg)
        self.tau_steering = 2          # Steering time constant
        self.tau_velocity = 3          # Velocity time constant
        self.max_vel = 5               # Maximum velocity

    def step(self, action, dt, state):
        # Ensure state and action are tensors on GPU
        x = state[0]
        y = state[1]
        phi = state[2]
        prev_vel = state[3]
        prev_steer = state[4]

        vel = action[0]
        steer = action[1]

        # Update velocity and steering with time constants
        vel = prev_vel + dt * (vel - prev_vel) / self.tau_velocity
        steer = prev_steer + dt * (steer - prev_steer) / self.tau_steering

        # Saturate velocity to maximum value
        vel = torch.clamp(vel, max=self.max_vel)

        # Update position and orientation
        x += vel * dt * torch.cos(steer + phi)
        y += vel * dt * torch.sin(steer + phi)
        phi += steer

        # Return updated state as a tensor on GPU
        return torch.tensor([x, y, phi, vel, steer], device='cuda', dtype=torch.float32)
