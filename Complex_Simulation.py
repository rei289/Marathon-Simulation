"""
This is a script which runs the more complex model which incorporates the effects of terrain and weather on the runner's performance.

The model assumes throughout the run there are 3 main phases:
    1. Acceleration phase
    2. Constant velocity phase
    3. Deceleration phase

In addition to the Kellner model, this simulation incorporates:
    - Slope of the terrain (theta) which affects the runner's velocity and energy expenditure
    - Air resistance which is proportional to the square of the velocity and a drag coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

class Runner:
    def __init__(self, params):
        # Physiological Constants from the paper
        self.F = params['F']          # Max thrust [cite: 9]
        self.E0 = params['E0']        # Initial energy [cite: 9]
        self.tau = params['tau']      # Resistance coefficient [cite: 9]
        self.sigma = params['sigma']  # Energy supply rate [cite: 9]
        self.gamma = params['gamma']  # Fatigue constant [cite: 9]
        self.k = self.gamma*2

        # other physiological constants used for the more complex model
        self.cd = params['drag_coefficient']  # Drag coefficient for air resistance
        self.area = params['frontal_area']          # Frontal area of the runner (m^2)
        self.mass = params['mass']              # Mass of the runner (kg)
        
        # State Variables
        self.velocity = [0.0]
        self.energy = [self.E0]

class Terrain:
    def __init__(self, params, csv_data, json_data):
        self.df = pd.read_csv(csv_data)[['distance_m', 'grade_percent', 'headwind_mps']].fillna(0)
        with open(json_data, 'r') as f:
            self.weather_info = json.load(f)["weather"]

        self.rho = params['air_density'] # air density (kg/m^3)
        self.convection = params['convection_heat_transfer_coefficient'] # convection heat transfer coefficient (W/m^2K)
        self.alpha = params['absorption_coefficient'] # absorption coefficient for solar radiation (dimensionless)
        self.psi = params['psi']

        self.temp_d = self.weather_info["temp"]
        self.humidity = self.weather_info["humidity"]
        self.solar_radiation = self.weather_info["solarradiation"]

    def get_grade(self, distance) -> float:
        """
        Returns the grade (theta) in radians at a given distance along the course.
        """
        # from the distance column, check which distance is the closest
        closest_idx = (self.df['distance_m'] - distance).abs().idxmin()
        grade_percent = self.df.loc[closest_idx, 'grade_percent']

        return np.arctan(grade_percent / 100.0)  # convert percent grade to radians (assuming small angles)
    
    def get_headwind(self, distance) -> float:
        """
        Returns the headwind speed in m/s at a given distance along the course.
        """
        closest_idx = (self.df['distance_m'] - distance).abs().idxmin()
        headwind = self.df.loc[closest_idx, 'headwind_mps']
        return headwind
    
    def get_wbgt(self) -> float:
        """
        Returns the Wet Bulb Globe Temperature (WBGT) based on the weather information.
        This is a simplified calculation and can be expanded to include more factors.
        """
        temp_w = self.temp_d * np.arctan(0.151977*(self.humidity + 8.313659)**(1/2)) \
            + np.arctan(self.temp_d + self.humidity) \
            - np.arctan(self.humidity - 1.676331) \
            + 0.00391838*(self.humidity)**(3/2) * np.arctan(0.023101 * self.humidity) \
            - 4.686
        temp_g = self.temp_d + (self.solar_radiation) / (self.convection*self.alpha)  # simplified effect of solar radiation on perceived temperature
        
        return 0.7*temp_w + 0.2*temp_g + 0.1*self.temp_d  # weighted average to get a single WBGT value

class MarathonSimulation:
    def __init__(self, runner, terrain, target_distance=42195, dt=0.01, const_v=None):
        self.runner = runner
        self.terrain = terrain
        self.distance_covered = 0.0

        self.target_dist = target_distance
        self.dt = dt
        self.time_elapsed = [0]
        self.const_v = const_v if const_v is not None else self._constant_velocity()

        self.g = 9.81  # gravitational acceleration (m/s^2)

        self.t1 = None
        self.t2 = None
        self.time_phase = False if self.t1 is None or self.t2 is None else True

        self.elevation_profile = [0.0]
        self.headwind_profile = [0.0]
        
    def step(self):
        """
        Runs one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run.
        """
        # determine the current terrain conditions based on the distance covered
        theta = self.terrain.get_grade(self.distance_covered)
        headwind = self.terrain.get_headwind(self.distance_covered)
        # theta = 0.0
        # headwind = 0.0

        # add it to a list for plotting later
        self.elevation_profile.append(theta)
        self.headwind_profile.append(headwind)

        if self.time_phase:
            if self.time_elapsed[-1] < self.t1:
                self.phase_1(theta, headwind)
            elif self.t1 <= self.time_elapsed[-1] < self.t2:
                self.phase_2(theta, headwind)
            else:
                self.phase_3(theta, headwind)
        else:  
            # determine which phase we are in based on certain factors
            if self.runner.velocity[-1] < self.const_v and self.runner.energy[-1] > 0: # if we havent reached the constant velocity yet, we are in phase 1
                self.phase_1(theta, headwind)
            elif self.runner.velocity[-1] >= self.const_v and self.runner.energy[-1] > 0: # if we have reached the constant velocity and still have energy, we are in phase 2
                self.t1 = self.time_elapsed[-1] if self.t1 is None else self.t1
                self.phase_2(theta, headwind)
            elif self.runner.energy[-1] <= 0: # if we have no energy left, we are in phase 3
                self.t2 = self.time_elapsed[-1] if self.t2 is None else self.t2
                self.phase_3(theta, headwind)

            else:
                raise ValueError("Invalid state: velocity and energy values do not correspond to any phase.")

        # now update the time and distance covered
        self.time_elapsed.append(self.time_elapsed[-1] + self.dt)
        self.distance_covered += self.runner.velocity[-1] * self.dt

    def phase_1(self, theta:float = 0.0, headwind:float = 0.0) -> None:
        """
        This function provides the Acceleration Phase Logic"""
        self.runner.velocity.append(self.runner.velocity[-1] \
                                + (self.runner.F         \
                                    - (1/self.runner.tau) * self.runner.velocity[-1] \
                                    - self.g*np.sin(theta) \
                                    - (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.runner.velocity[-1] + headwind)**2)/self.runner.mass)*self.dt)
        energy_change = (self.runner.sigma - self.runner.F*self.runner.velocity[-1])*self.dt
        self.runner.energy.append(self.runner.energy[-1] + (energy_change if energy_change < 0 else 0))

    def phase_2(self, theta:float = 0.0, headwind:float = 0.0) -> None:
        """
        This function provides the Constant Velocity Phase Logic
        """
        self.runner.velocity.append(self.const_v)
        self.runner.energy.append(self.runner.energy[-1] \
                                + (self.runner.sigma \
                                     - self.const_v*(self.const_v/self.runner.tau \
                                            + self.g*np.sin(theta) \
                                            + (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.const_v + headwind)**2)/self.runner.mass) \
                                - (self.runner.k*self.const_v**2*self.time_elapsed[-1])/self.runner.tau)*self.dt)

    def phase_3(self, theta:float = 0.0, headwind:float = 0.0) -> None:
        """
        This function provides the Deceleration Phase Logic
        """
        self.runner.velocity.append(self.runner.velocity[-1] \
                                +  (self.runner.sigma/self.runner.velocity[-1] \
                                - (1/self.runner.tau)*self.runner.velocity[-1] \
                                - self.g*np.sin(theta) \
                                - (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.runner.velocity[-1] + headwind)**2)/self.runner.mass \
                                - (self.runner.k*self.const_v**2*self.time_elapsed[-1])/(self.runner.tau*self.runner.velocity[-1]))*self.dt)
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
        # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
        print(f"Initial Energy Supply Rate (sigma): {self.runner.sigma:.2f} m^2/s^3")
        wbgt = self.terrain.get_wbgt()
        self.runner.sigma *= 1 - self.terrain.psi*max(0, wbgt - 15)
        print(f"Calculated WBGT: {wbgt:.2f} °C")
        print(f"Adjusted Energy Supply Rate (sigma): {self.runner.sigma:.2f} m^2/s^3")
        # while self.time_elapsed[-1] < 10:
        while self.distance_covered < self.target_dist:
            self.step()

        print(f"Time Elapsed: {self.time_elapsed[-1]} seconds")
        print(f"Distance Covered: {self.distance_covered} meters")
        print(f"t1 (start of constant velocity phase): {self.t1} seconds")
        print(f"t2 (start of deceleration phase): {self.t2} seconds")
        print(f"Constant Velocity (v): {self.const_v} m/s")

if __name__ == "__main__":
    # for a more reasonable run 
    runner = Runner(params={
        # these are mens average
        'F': 10.5,          # Max thrust (m/s^2) (9.0–12.0)
        'E0': 2200.0,       # Initial energy (m^2/s^2) (1800–2600)
        'tau': 1.0,         # Resistance coefficient (s) (0.8–1.2)
        'sigma': 45.0,      # Energy supply rate (m^2/s^3) (35-55)
        'gamma': 5.0e-5,    # Fatigue constant (3e-5 to 8e-5)

        # 'drag_coefficient': 0.65,  # Drag coefficient for a runner (dimensionless)
        'drag_coefficient': 1.0,  # Drag coefficient for a runner (dimensionless) (0.9-1.1)
        'frontal_area': 0.47,      # Frontal area of the runner (m^2) (0.4-0.55)
        'mass': 70.0              # Mass of the runner (kg) (60-80)

    })
    # # specify the runner parameters based on the paper's values for now
    # runner = Runner(params={
    #     # these are mens average
    #     'F': 14.36,         # Max thrust (m/s^2)
    #     'tau': 0.739,       # Resistance coefficient (s)
    #     'E0': 3114.0,       # Initial energy (m^2/s^2)
    #     'sigma': 58.0,      # Energy supply rate (m^2/s^3)
    #     'gamma': 4.08e-5,    # Fatigue constant 

    #     'drag_coefficient': 0.65,  # Drag coefficient for a runner (dimensionless)
    #     # 'drag_coefficient': 1.0,  # Drag coefficient for a runner (dimensionless)
    #     'frontal_area': 0.5,      # Frontal area of the runner (m^2)

    # })

    terrain = Terrain(params={
        'air_density': 1.225,  # air density at sea level (kg/m^3)
        'convection_heat_transfer_coefficient': 10.0,  # convection heat transfer coefficient (W/m^2K)
        'absorption_coefficient': 0.7,  # absorption coefficient for solar radiation (dimensionless)
        'psi': 0.005,  # weighting factor for the drop in aerobic power per temperature (dimensionless)
    }, 
    csv_data="runs/2025-10-10_10-42/2025-10-10_10-42_streams.csv",  # flat terrain for now, can be modified to include elevation changes
    json_data="runs/2025-10-10_10-42/2025-10-10_10-42_overall.json")
    # print(terrain.df.head())  # check the terrain data

    sim = MarathonSimulation(runner, terrain, target_distance=4300, dt=0.01)
    # sim = MarathonSimulation(runner, terrain, target_distance=4300, dt=0.01, const_v=4.0)
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

    # plotting elevation profile
    plt.figure(figsize=(12, 4))
    plt.plot(sim.time_elapsed, sim.elevation_profile, label='Elevation Profile (radians)')
    plt.title('Elevation Profile Over Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Grade (radians)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plotting headwind profile
    plt.figure(figsize=(12, 4))
    plt.plot(sim.time_elapsed, sim.headwind_profile, label='Headwind Profile (m/s)')
    plt.title('Headwind Profile Over Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Headwind Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.show()
