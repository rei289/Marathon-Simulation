"""
This is a script which runs the Kellner Model

The model assumes throughout the run there are 3 main phases:
    1. Acceleration phase
    2. Constant velocity phase
    3. Deceleration phase
"""

import numpy as np
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, params):
        # Physiological Constants from the paper
        self.F = params['F']          # Max thrust [cite: 9]
        self.E0 = params['E0']        # Initial energy [cite: 9]
        self.tau = params['tau']      # Resistance coefficient [cite: 9]
        self.sigma = params['sigma']  # Energy supply rate [cite: 9]
        self.gamma = params['gamma']  # Fatigue constant [cite: 9]
        self.k = self.gamma*2
        
        # State Variables
        self.velocity = [0.0]
        self.distance_covered = 0.0
        self.energy = [self.E0]

class MarathonSimulation:
    def __init__(self, runner, target_distance=42195, dt=0.01, const_v=None):
        self.runner = runner
        self.target_dist = target_distance
        self.dt = dt
        self.time_elapsed = [0]
        self.const_v = const_v if const_v is not None else self._constant_velocity()

        self.t1 = None
        self.t2 = None
        self.time_phase = False if self.t1 is None or self.t2 is None else True
        
    def step(self):
        """
        Runs one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run.
        """
        # if self.t1 is not None and self.t2 is not None:
        if self.time_phase:
            if self.time_elapsed[-1] < self.t1:
                self.phase_1()
            elif self.t1 <= self.time_elapsed[-1] < self.t2:
                self.phase_2()
            else:
                self.phase_3()
        else:  
            # determine which phase we are in based on certain factors
            if self.runner.velocity[-1] < self.const_v and self.runner.energy[-1] > 0: # if we havent reached the constant velocity yet, we are in phase 1
            # if self.runner.velocity[-1] < self.const_v and self.runner.energy[-1] > 0: # if we havent reached the constant velocity yet, we are in phase 1
                self.phase_1()
            elif self.runner.velocity[-1] >= self.const_v and self.runner.energy[-1] > 0: # if we have reached the constant velocity and still have energy, we are in phase 2
            # elif self.runner.velocity[-1] >= self.const_v and self.runner.energy[-1] > 0: # if we have reached the constant velocity and still have energy, we are in phase 2
                self.t1 = self.time_elapsed[-1] if self.t1 is None else self.t1
                self.phase_2()
            elif self.runner.energy[-1] <= 0: # if we have no energy left, we are in phase 3
            # elif self.runner.energy[-1] <= 0: # if we have no energy left, we are in phase 3
                self.t2 = self.time_elapsed[-1] if self.t2 is None else self.t2
                self.phase_3()

            else:
                raise ValueError("Invalid state: velocity and energy values do not correspond to any phase.")

        # now update the time and distance covered
        self.time_elapsed.append(self.time_elapsed[-1] + self.dt)
        self.runner.distance_covered += self.runner.velocity[-1] * self.dt

    def phase_1(self) -> None:
        """
        This function provides the Acceleration Phase Logic"""
        self.runner.velocity.append(self.runner.F*self.runner.tau*(1-np.exp(-self.time_elapsed[-1]/self.runner.tau)))
        energy_change = (self.runner.sigma - self.runner.F*self.runner.velocity[-1])*self.dt
        self.runner.energy.append(self.runner.energy[-1] + (energy_change if energy_change < 0 else 0))

    def phase_2(self) -> None:
        """
        This function provides the Constant Velocity Phase Logic
        """
        self.runner.velocity.append(self.const_v)
        self.runner.energy.append(self.runner.energy[-1] + (self.runner.sigma - (self.const_v**2)/self.runner.tau - (self.runner.k*self.const_v**2*self.time_elapsed[-1])/self.runner.tau)*self.dt)

    def phase_3(self) -> None:
        """
        This function provides the Deceleration Phase Logic
        """
        velocity = (self.const_v**2 + self.runner.k*self.t2*self.const_v**2 - self.runner.tau*self.runner.sigma - 0.5*self.runner.k*(self.const_v**2)*self.runner.tau)*np.exp(-2*(self.time_elapsed[-1]-self.t2)/self.runner.tau) \
                                     + self.runner.sigma*self.runner.tau  \
                                     + 0.5*self.runner.k*(self.const_v**2)*(self.runner.tau-2*self.time_elapsed[-1])
        self.runner.velocity.append(np.sqrt(max(0, velocity)))
        self.runner.energy.append(0.0)
    
    def _constant_velocity(self):
        """
        Calculates the constant velocity during phase 2 based on the model equations.
        """
        const_t = 1/(2*self.runner.sigma) * ((self.runner.E0**2 + (4*self.runner.sigma*self.target_dist**2)/self.runner.tau)**0.5 - self.runner.E0)
        const_v = (((self.runner.E0*self.runner.tau)/const_t) + self.runner.sigma*self.runner.tau)**0.5
        return const_v

    def loop(self):
        """
        Runs the simulation until the target distance is reached
        """
        while self.runner.distance_covered < self.target_dist:
            self.step()

        print(f"Time Elapsed: {self.time_elapsed[-1]} seconds")
        print(f"Distance Covered: {self.runner.distance_covered} meters")
        print(f"t1 (start of constant velocity phase): {self.t1} seconds")
        print(f"t2 (start of deceleration phase): {self.t2} seconds")
        print(f"Constant Velocity (v): {self.const_v} m/s")

if __name__ == "__main__":
    # specify the runner parameters based on the paper's values for now
    runner = Runner(params={
        # these are mens average
        'F': 14.36,         # Max thrust (m/s^2)
        'tau': 0.739,       # Resistance coefficient (s)
        'E0': 3114.0,       # Initial energy (m^2/s^2)
        'sigma': 58.0,      # Energy supply rate (m^2/s^3)
        'gamma': 4.08e-5    # Fatigue constant 
    })
    sim = MarathonSimulation(runner, target_distance=4300)
    # sim = MarathonSimulation(runner, target_distance=4300, const_v=5.0)
    sim.loop()

    # plotting results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sim.runner.velocity, label='Velocity (m/s)')
    plt.title('Runner Velocity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(sim.runner.energy, label='Energy (J)')
    plt.title('Runner Energy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.tight_layout()
    plt.show()