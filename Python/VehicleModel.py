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

class VehicleModel:
    def __init__(self):
        self.m = 2000.0                # Mass (kg)
        self.tau_steering = 2          # Steering time constant
        self.tau_velocity = 3          # Velocity time constant
        self.max_vel = 5               # Maximum velocity

    def step(self, action, dt, state):
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
        if vel > self.max_vel:
            vel = min(vel, self.max_vel)

        # Update position and orientation
        x += vel * dt * np.cos(steer + phi)
        y += vel * dt * np.sin(steer + phi)
        phi += steer

        # Return updated state
        return np.array([x, y, phi, vel, steer])
