import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QuantumZenoEffectSimulator:
    def __init__(self, photon_states, measurement_rate, relaxation_rate, time_steps):
        """
        Initialize the simulator for the partial Zeno effect.

        Args:
            photon_states (int): Number of photon states.
            measurement_rate (float): Rate of measurement (frequency of observation).
            relaxation_rate (float): Rate of relaxation (decay).
            time_steps (int): Number of time steps to simulate.
        """
        self.photon_states = photon_states
        self.measurement_rate = measurement_rate
        self.relaxation_rate = relaxation_rate
        self.time_steps = time_steps
        self.state = torch.zeros(photon_states)
        self.state[0] = 1.0  # Initial state: all probability in the ground state

    def apply_relaxation(self):
        """
        Apply relaxation to the quantum state.
        """
        relaxation_matrix = torch.zeros((self.photon_states, self.photon_states))
        for i in range(self.photon_states - 1):
            relaxation_matrix[i, i] = 1 - self.relaxation_rate
            relaxation_matrix[i + 1, i] = self.relaxation_rate
        relaxation_matrix[-1, -1] = 1  # Last state remains unchanged
        self.state = torch.matmul(relaxation_matrix, self.state)

    def apply_measurement(self):
        """
        Apply measurement to the quantum state (partial Zeno effect).
        """
        measurement_matrix = torch.eye(self.photon_states) * (1 - self.measurement_rate)
        self.state = torch.matmul(measurement_matrix, self.state)
        self.state = self.state / self.state.sum()  # Normalize the state

    def simulate(self):
        """
        Simulate the quantum system over time steps.
        """
        history = []
        for _ in range(self.time_steps):
            self.apply_relaxation()
            self.apply_measurement()
            history.append(self.state.clone().numpy())
        return np.array(history)

if __name__ == '__main__':
    # Parameters for the simulation
    photon_states = 5  # Number of photon states
    measurement_rate = 0.1  # Rate of measurement
    relaxation_rate = 0.05  # Rate of relaxation
    time_steps = 50  # Number of time steps

    # Initialize the simulator
    simulator = QuantumZenoEffectSimulator(photon_states, measurement_rate, relaxation_rate, time_steps)

    # Run the simulation
    history = simulator.simulate()

    # Print the results
    print("Photon state probabilities over time:")
    print(history)