"""
This is a script which runs the more complex model which incorporates the effects of terrain and weather on the runner's performance.

The model assumes throughout the run there are 3 main phases:
    1. Acceleration phase
    2. Constant velocity phase
    3. Deceleration phase

In addition to the Kellner model, this simulation incorporates:
    - Slope of the terrain (theta) which affects the runner's velocity and energy expenditure
    - Air resistance which is proportional to the square of the velocity and a drag coefficient
    - Heat stress which reduces the effective aerobic power supply (sigma) based on the Wet Bulb Globe Temperature (WBGT)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from DataClasses import SimConfig, Params
from dataclasses import asdict

class MonteCarloSimulation:
    def __init__(self, cfg: SimConfig, params: Params, csv_data: str, json_data: str):
        self.target_dist = cfg.target_dist
        self.num_sim = cfg.num_sim
        self.dt = cfg.dt
        self.max_steps = cfg.max_steps

        self.df = pd.read_csv(csv_data)[['distance_m', 'grade_percent', 'headwind_mps']].fillna(0)
        with open(json_data, 'r') as f:
            self.weather_info = json.load(f)["weather"]

        self.temp_d = self.weather_info["temp"]
        self.humidity = self.weather_info["humidity"]
        self.solar_radiation = self.weather_info["solarradiation"]

        self.cfg = cfg 
        self.params = params
        # self.cfg.sigma *= 1 - self.cfg.psi*max(0, self._get_wbgt() - 15)

        self.g = 9.81  # gravitational acceleration (m/s^2)

        # create a numpy array that contains the random distribution of the parameters for multiple simulations
        for param, bounds in asdict(self.params).items():
            # we make a very small adjustment
            if len(bounds) == 1:
                setattr(self, f"{param}_values", np.full(self.num_sim, bounds[0]))
            
            elif len(bounds) == 2:
                setattr(self, f"{param}_values", np.random.uniform(bounds[0], bounds[1], self.num_sim))

            else:
                raise ValueError(f"Invalid bounds for parameter {param}: {bounds}")

        # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
        self.sigma_values *= np.ones(self.num_sim) - self.psi_values*np.maximum(0, self._get_wbgt() - 15)  # adjust sigma for heat stress for each simulation
        # add the k_values which is derived from the gamma values
        self.k_values = self.gamma_values*2

        # self.F_values = np.random.uniform(self.params.F[0], self.params.F[1], self.num_sim)
        # self.E0_values = np.random.uniform(1800, 2600, self.num_sim)
        # self.tau_values = np.random.uniform(0.8, 1.2, self.num_sim)
        # self.sigma_values = np.random.uniform(35, 55, self.num_sim)
        # self.gamma_values = np.random.uniform(3e-5, 8e-5, self.num_sim)
        # self.k_values = self.gamma_values*2
        # self.drag_coefficient_values = np.random.uniform(0.9, 1.1, self.num_sim)
        # self.frontal_area_values = np.random.uniform(0.4, 0.55, self.num_sim)
        # self.mass_values = np.random.uniform(60, 80, self.num_sim)
        # self.alpha_values = np.random.uniform(0.6, 0.8, self.num_sim)
        # self.psi_values = np.random.uniform(0.003, 0.007, self.num_sim)

        # create an array to store the results of multiple simulations
        self.velocity = np.full((self.max_steps, self.num_sim), np.nan)
        self.energy = np.full((self.max_steps, self.num_sim), np.nan)
        self.time_elapsed = np.full(self.max_steps, np.nan)
        self.distance_covered = np.full((self.max_steps, self.num_sim), np.nan)
        self.elevation_profile = np.full(self.max_steps, np.nan)
        self.headwind_profile = np.full(self.max_steps, np.nan)
        self.finish_time = np.full(self.num_sim, np.nan)

        # update the first row with the initial conditions
        self.velocity[0] = np.zeros(self.num_sim)
        self.energy[0] = self.E0_values
        self.time_elapsed[0] = 0.0
        self.distance_covered[0] = np.zeros(self.num_sim)
        self.elevation_profile[0] = 0.0
        self.headwind_profile[0] = 0.0

        self.active = np.ones(self.num_sim, dtype=bool)   # True = still running

        self.const_v = cfg.const_v if cfg.const_v is not None else self._constant_velocity()
        self.iteration = 0 

    def _constant_velocity(self):
        """
        Calculates the constant velocity during phase 2 based on the model equations.
        """
        const_t = 1/(2*self.sigma_values) * ((self.E0_values**2 + (4*self.sigma_values*self.target_dist**2)/self.tau_values)**0.5 - self.E0_values)
        const_v = (((self.E0_values*self.tau_values)/const_t) + self.sigma_values*self.tau_values)**0.5
        return const_v

    
    def _get_grade(self, distance: np.ndarray) -> float:
        """
        Returns the grade (theta) in radians at a given distance along the course.
        """
        dist = self.df['distance_m'].values
        gp = self.df['grade_percent'].values

        # find the two points in the distance array that are closest to the current distance
        right = np.searchsorted(dist, distance, side='right')
        right = np.clip(right, 0, len(dist) - 1)
        left = np.clip(right - 1, 0, len(dist) - 1)

        closest_idx = np.where(np.abs(distance - dist[left]) <= np.abs(dist[right] - distance), left, right)
        grade_percent = gp[closest_idx]

        return np.arctan(grade_percent / 100.0)  # convert percent grade to radians (assuming small angles)
    
    def _get_headwind(self, distance: np.ndarray) -> float:
        """
        Returns the headwind speed in m/s at a given distance along the course.
        """
        dist = self.df['distance_m'].values
        hw = self.df['headwind_mps'].values

        # find the two points in the distance array that are closest to the current distance
        right = np.searchsorted(dist, distance, side='right')
        right = np.clip(right, 0, len(dist) - 1)
        left = np.clip(right - 1, 0, len(dist) - 1)

        closest_idx = np.where(np.abs(distance - dist[left]) <= np.abs(dist[right] - distance), left, right)
        headwind = hw[closest_idx]

        return headwind
    
    def _get_wbgt(self) -> np.ndarray:
        """
        Returns the Wet Bulb Globe Temperature (WBGT) based on the weather information.
        This is a simplified calculation and can be expanded to include more factors.
        """
        temp_w = self.temp_d * np.arctan(0.151977*(self.humidity + 8.313659)**(1/2)) \
            + np.arctan(self.temp_d + self.humidity) \
            - np.arctan(self.humidity - 1.676331) \
            + 0.00391838*(self.humidity)**(3/2) * np.arctan(0.023101 * self.humidity) \
            - 4.686
        temp_g = self.temp_d + (self.solar_radiation) / (self.convection_values*self.alpha_values)  # simplified effect of solar radiation on perceived temperature
        # temp_g = self.temp_d + (self.solar_radiation) / (self.cfg.convection*self.cfg.alpha)  # simplified effect of solar radiation on perceived temperature
        
        return 0.7*temp_w + 0.2*temp_g + 0.1*self.temp_d  # weighted average to get a single WBGT value
        
    def phase_1(self, theta:np.ndarray, headwind:np.ndarray, mask:np.ndarray) -> None:
        """
        This function provides the Acceleration Phase Logic
        """
        dv = self.F_values[mask] \
            - (1/self.tau_values[mask]) * self.velocity[self.iteration][mask] \
            - self.g*np.sin(theta[mask]) \
            - (0.5*self.rho_values[mask]*self.drag_coefficient_values[mask] * self.frontal_area_values[mask] * (self.velocity[self.iteration][mask] + headwind[mask])**2) / self.mass_values[mask]

        self.velocity[self.iteration + 1][mask] = self.velocity[self.iteration][mask] + dv*self.dt

        dE = self.sigma_values[mask] - self.F_values[mask]*self.velocity[self.iteration][mask]

        self.energy[self.iteration + 1][mask] = self.energy[self.iteration][mask] + dE*self.dt
        # self.energy[self.iteration + 1] = self.energy[self.iteration] + (dE if dE < 0 else 0)*self.dt
        # self.velocity.append(self.velocity[-1] \
        #                         + (self.F         \
        #                             - (1/self.tau) * self.velocity[-1] \
        #                             - self.g*np.sin(theta) \
        #                             - (0.5*self.rho*self.cd*self.area*(self.velocity[-1] + headwind)**2)/self.mass)*self.dt)
        # energy_change = (self.sigma - self.F*self.velocity[-1])*self.dt
        # self.energy.append(self.energy[-1] + (energy_change if energy_change < 0 else 0))

    def phase_2(self, theta:np.ndarray, headwind:np.ndarray, mask:np.ndarray) -> None:
        """
        This function provides the Constant Velocity Phase Logic
        """
        self.velocity[self.iteration + 1][mask] = self.const_v[mask]

        dE = self.sigma_values[mask] \
                - self.const_v[mask]*(self.const_v[mask]/self.tau_values[mask] \
                    + self.g*np.sin(theta[mask]) \
                    + (0.5*self.rho_values[mask]*self.drag_coefficient_values[mask]*self.frontal_area_values[mask]*(self.const_v[mask] + headwind[mask])**2)/self.mass_values[mask]) \
                - (self.k_values[mask]*self.const_v[mask]**2*self.time_elapsed[self.iteration])/self.tau_values[mask]

        self.energy[self.iteration + 1][mask] = self.energy[self.iteration][mask] + dE*self.dt
        # self.velocity.append(self.const_v)
        # self.energy.append(self.energy[-1] \
        #                         + (self.sigma \
        #                              - self.const_v*(self.const_v/self.tau \
        #                                     + self.g*np.sin(theta) \
        #                                     + (0.5*self.rho*self.cd*self.area*(self.const_v + headwind)**2)/self.mass) \
        #                         - (self.k*self.const_v**2*self.time_elapsed[-1])/self.tau)*self.dt)

    def phase_3(self, theta:np.ndarray, headwind:np.ndarray, mask:np.ndarray) -> None:
        """
        This function provides the Deceleration Phase Logic
        """
        dv = self.sigma_values[mask]/self.velocity[self.iteration][mask] \
                - (1/self.tau_values[mask]) * self.velocity[self.iteration][mask] \
                - self.g*np.sin(theta[mask]) \
                - (0.5*self.rho_values[mask]*self.drag_coefficient_values[mask]*self.frontal_area_values[mask]*(self.velocity[self.iteration][mask] + headwind[mask])**2)/self.mass_values[mask] \
                - (self.k_values[mask]*self.const_v[mask]**2*self.time_elapsed[self.iteration])/(self.tau_values[mask]*self.velocity[self.iteration][mask])
        
        self.velocity[self.iteration + 1][mask] = self.velocity[self.iteration][mask] + dv*self.dt

        self.energy[self.iteration + 1][mask] = np.zeros(self.num_sim)[mask]
        # self.velocity.append(self.velocity[-1] \
        #                         +  (self.sigma/self.velocity[-1] \
        #                         - (1/self.tau)*self.velocity[-1] \
        #                         - self.g*np.sin(theta) \
        #                         - (0.5*self.rho*self.cd*self.area*(self.velocity[-1] + headwind)**2)/self.mass \
        #                         - (self.k*self.const_v**2*self.time_elapsed[-1])/(self.tau*self.velocity[-1]))*self.dt)
        # self.energy.append(0.0)
    
    def step(self):
        """
        Runs one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run.
        """
        # determine the current terrain conditions based on the distance covered
        # print(self.distance_covered[self.iteration])


        theta = self._get_grade(self.distance_covered[self.iteration])
        headwind = self._get_headwind(self.distance_covered[self.iteration])

        # add it to a list for plotting later
        self.elevation_profile[self.iteration] = theta[0]
        self.headwind_profile[self.iteration] = headwind[0]

        # create a vectorized conditional to determine which phase we are in for each simulation
        v = self.velocity[self.iteration]   # current velocity for all sims
        e = self.energy[self.iteration]     # current energy for all sims

        # per-column conditions (vectorized if/elif/else)
        m1 = self.active & (v < self.const_v) & (e > 0)   # phase 1
        m2 = self.active & (v >= self.const_v) & (e > 0)  # phase 2
        m3 = self.active & (e <= 0)                        # phase 3

        # phase 1
        if m1.any():
            self.phase_1(theta, headwind, m1)
        
        # phase 2
        if m2.any():
            self.phase_2(theta, headwind, m2)

        # phase 3
        if m3.any():
            self.phase_3(theta, headwind, m3)

        # if self.time_phase:
        #     if self.time_elapsed[-1] < self.t1:
        #         self.phase_1(theta, headwind)
        #     elif self.t1 <= self.time_elapsed[-1] < self.t2:
        #         self.phase_2(theta, headwind)
        #     else:
        #         self.phase_3(theta, headwind)
        # else:  
        #     # determine which phase we are in based on certain factors
        #     if self.velocity[-1] < self.const_v and self.energy[-1] > 0: # if we havent reached the constant velocity yet, we are in phase 1
        #         self.phase_1(theta, headwind)
        #     elif self.velocity[-1] >= self.const_v and self.energy[-1] > 0: # if we have reached the constant velocity and still have energy, we are in phase 2
        #         self.t1 = self.time_elapsed[-1] if self.t1 is None else self.t1
        #         self.phase_2(theta, headwind)
        #     elif self.energy[-1] <= 0: # if we have no energy left, we are in phase 3
        #         self.t2 = self.time_elapsed[-1] if self.t2 is None else self.t2
        #         self.phase_3(theta, headwind)

        #     else:
        #         raise ValueError("Invalid state: velocity and energy values do not correspond to any phase.")

        # # now update the time and distance covered
        # self.time_elapsed.append(self.time_elapsed[-1] + self.dt)
        # self.distance_covered += self.velocity[-1] * self.dt

    
    def loop(self) -> None:
        """
        Runs the simulation until the target distance is reached
        """
        # for now we will use a uniform random distribution
        np.random.seed(42)  # for reproducibility

        for step in range(self.max_steps-1):
            # print(self.velocity.shape)
            self.step()

            # determine if any simulations have finished
            just_finished = self.active & (self.distance_covered[step] >= self.target_dist)
            self.active[just_finished] = False

            # determine finish time
            self.finish_time[just_finished] = self.time_elapsed[step]

            if not self.active.any():
                break                            # all sims done → early exit

            # lastly update time and distance for active simulations
            self.time_elapsed[step + 1] = self.time_elapsed[step] + self.dt
            self.distance_covered[step + 1] = self.distance_covered[step] + np.where(self.active, self.velocity[step] * self.dt, 0.0)

            self.iteration = step + 1


def spaghetti_plot(sim: MonteCarloSimulation):
    """
    This function plots all the results of the simulation (Note costs a lot of memory)
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sim.time_elapsed, sim.velocity, color="blue", alpha=0.05, label='Velocity (m/s)')
    plt.title(f'Monte Carlo: {sim.num_sim} Simulations Runner Velocity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    # plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(sim.time_elapsed, sim.energy, color="red", alpha=0.05, label='Energy (J)')
    plt.title(f'Monte Carlo: {sim.num_sim} Simulations Runner Energy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    # plt.legend()
    plt.tight_layout()
    plt.show()

def histogram_plot(sim: MonteCarloSimulation):
    """
    This function plots a histogram of the finish times of the simulations.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(sim.finish_time, bins=30, color='green', alpha=0.7)
    plt.title('Distribution of Finish Times')
    plt.xlabel('Finish Time (s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    csv_data="runs/2025-10-10_10-42/2025-10-10_10-42_streams.csv"  # flat terrain for now, can be modified to include elevation changes
    json_data="runs/2025-10-10_10-42/2025-10-10_10-42_overall.json"
    # print(terrain.df.head())  # check the terrain data

    sim = MonteCarloSimulation(
        SimConfig(
            target_dist=4300,
            num_sim=1000,
            dt=0.1,
            max_steps=10000,
            const_v=None,
            t1=None,
            t2=None
        ),
        Params(
            F=[9.0, 12.0],
            E0=[1800.0, 2600.0],
            tau=[0.8, 1.2],
            sigma=[35.0, 55.0],
            gamma=[3e-5, 8e-5],
            drag_coefficient=[0.9, 1.1],
            frontal_area=[0.4, 0.55],
            mass=[60.0, 80.0],
            rho=[1.225],
            convection=[10.0],
            alpha=[0.6, 0.8],
            psi=[0.003, 0.007]
            # F=10.5,
            # E0=2200.0,
            # tau=1.0,
            # sigma=45.0,
            # gamma=5.0e-5,
            # drag_coefficient=1.0,
            # frontal_area=0.47,
            # mass=70.0,
            # rho=1.225,
            # convection=10.0,
            # alpha=0.7,
            # psi=0.005
        ), csv_data=csv_data, json_data=json_data)

    # perform the simulation
    sim.loop()

    # plotting results
    # spaghetti_plot(sim)
    histogram_plot(sim)

    # # plotting distance covered
    # plt.figure(figsize=(12, 4))
    # plt.plot(sim.time_elapsed, sim.distance_covered[:, 0], label='Distance Covered (m)')
    # plt.title('Distance Covered Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Distance (m)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # plotting elevation profile
    # plt.figure(figsize=(12, 4))
    # plt.plot(sim.time_elapsed, sim.elevation_profile, label='Elevation Profile (radians)')
    # plt.title('Elevation Profile Over Distance')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Grade (radians)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # plotting headwind profile
    # plt.figure(figsize=(12, 4))
    # plt.plot(sim.time_elapsed, sim.headwind_profile, label='Headwind Profile (m/s)')
    # plt.title('Headwind Profile Over Distance')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Headwind Speed (m/s)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()







# """
# This is a script which runs the more complex model which incorporates the effects of terrain and weather on the runner's performance.

# The model assumes throughout the run there are 3 main phases:
#     1. Acceleration phase
#     2. Constant velocity phase
#     3. Deceleration phase

# In addition to the Kellner model, this simulation incorporates:
#     - Slope of the terrain (theta) which affects the runner's velocity and energy expenditure
#     - Air resistance which is proportional to the square of the velocity and a drag coefficient
#     - Heat stress which reduces the effective aerobic power supply (sigma) based on the Wet Bulb Globe Temperature (WBGT)
# """

# from dataclasses import dataclass
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import json

# class Runner:
#     def __init__(self, params):
#         # Physiological Constants from the paper
#         self.F = params['F']          # Max thrust [cite: 9]
#         self.E0 = params['E0']        # Initial energy [cite: 9]
#         self.tau = params['tau']      # Resistance coefficient [cite: 9]
#         self.sigma = params['sigma']  # Energy supply rate [cite: 9]
#         self.gamma = params['gamma']  # Fatigue constant [cite: 9]
#         self.k = self.gamma*2

#         # other physiological constants used for the more complex model
#         self.cd = params['drag_coefficient']  # Drag coefficient for air resistance
#         self.area = params['frontal_area']          # Frontal area of the runner (m^2)
#         self.mass = params['mass']              # Mass of the runner (kg)
        
#         # State Variables
#         self.velocity = [0.0]
#         self.energy = [self.E0]

# class Terrain:
#     def __init__(self, params, csv_data, json_data):
#         self.df = pd.read_csv(csv_data)[['distance_m', 'grade_percent', 'headwind_mps']].fillna(0)
#         with open(json_data, 'r') as f:
#             self.weather_info = json.load(f)["weather"]

#         self.rho = params['air_density'] # air density (kg/m^3)
#         self.convection = params['convection_heat_transfer_coefficient'] # convection heat transfer coefficient (W/m^2K)
#         self.alpha = params['absorption_coefficient'] # absorption coefficient for solar radiation (dimensionless)
#         self.psi = params['psi']

#         self.temp_d = self.weather_info["temp"]
#         self.humidity = self.weather_info["humidity"]
#         self.solar_radiation = self.weather_info["solarradiation"]

#     def get_grade(self, distance) -> float:
#         """
#         Returns the grade (theta) in radians at a given distance along the course.
#         """
#         # from the distance column, check which distance is the closest
#         closest_idx = (self.df['distance_m'] - distance).abs().idxmin()
#         grade_percent = self.df.loc[closest_idx, 'grade_percent']

#         return np.arctan(grade_percent / 100.0)  # convert percent grade to radians (assuming small angles)
    
#     def get_headwind(self, distance) -> float:
#         """
#         Returns the headwind speed in m/s at a given distance along the course.
#         """
#         closest_idx = (self.df['distance_m'] - distance).abs().idxmin()
#         headwind = self.df.loc[closest_idx, 'headwind_mps']
#         return headwind
    
#     def get_wbgt(self) -> float:
#         """
#         Returns the Wet Bulb Globe Temperature (WBGT) based on the weather information.
#         This is a simplified calculation and can be expanded to include more factors.
#         """
#         temp_w = self.temp_d * np.arctan(0.151977*(self.humidity + 8.313659)**(1/2)) \
#             + np.arctan(self.temp_d + self.humidity) \
#             - np.arctan(self.humidity - 1.676331) \
#             + 0.00391838*(self.humidity)**(3/2) * np.arctan(0.023101 * self.humidity) \
#             - 4.686
#         temp_g = self.temp_d + (self.solar_radiation) / (self.convection*self.alpha)  # simplified effect of solar radiation on perceived temperature
        
#         return 0.7*temp_w + 0.2*temp_g + 0.1*self.temp_d  # weighted average to get a single WBGT value

# class MarathonSimulation:
#     def __init__(self, runner, terrain, target_distance=42195, dt=0.01, const_v=None, num_sim:int=1000):
#         self.runner = runner
#         self.terrain = terrain
#         self.distance_covered = 0.0

#         self.target_dist = target_distance
#         self.dt = dt
#         self.time_elapsed = [0]
#         self.const_v = const_v if const_v is not None else self._constant_velocity()

#         self.g = 9.81  # gravitational acceleration (m/s^2)

#         self.t1 = None
#         self.t2 = None
#         self.time_phase = False if self.t1 is None or self.t2 is None else True

#         self.elevation_profile = [0.0]
#         self.headwind_profile = [0.0]

#         self.num_sim = num_sim
        
#     def step(self):
#         """
#         Runs one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run.
#         """
#         # determine the current terrain conditions based on the distance covered
#         theta = self.terrain.get_grade(self.distance_covered)
#         headwind = self.terrain.get_headwind(self.distance_covered)
#         # theta = 0.0
#         # headwind = 0.0

#         # add it to a list for plotting later
#         self.elevation_profile.append(theta)
#         self.headwind_profile.append(headwind)

#         if self.time_phase:
#             if self.time_elapsed[-1] < self.t1:
#                 self.phase_1(theta, headwind)
#             elif self.t1 <= self.time_elapsed[-1] < self.t2:
#                 self.phase_2(theta, headwind)
#             else:
#                 self.phase_3(theta, headwind)
#         else:  
#             # determine which phase we are in based on certain factors
#             if self.runner.velocity[-1] < self.const_v and self.runner.energy[-1] > 0: # if we havent reached the constant velocity yet, we are in phase 1
#                 self.phase_1(theta, headwind)
#             elif self.runner.velocity[-1] >= self.const_v and self.runner.energy[-1] > 0: # if we have reached the constant velocity and still have energy, we are in phase 2
#                 self.t1 = self.time_elapsed[-1] if self.t1 is None else self.t1
#                 self.phase_2(theta, headwind)
#             elif self.runner.energy[-1] <= 0: # if we have no energy left, we are in phase 3
#                 self.t2 = self.time_elapsed[-1] if self.t2 is None else self.t2
#                 self.phase_3(theta, headwind)

#             else:
#                 raise ValueError("Invalid state: velocity and energy values do not correspond to any phase.")

#         # now update the time and distance covered
#         self.time_elapsed.append(self.time_elapsed[-1] + self.dt)
#         self.distance_covered += self.runner.velocity[-1] * self.dt

#     def phase_1(self, theta:float = 0.0, headwind:float = 0.0) -> None:
#         """
#         This function provides the Acceleration Phase Logic"""
#         self.runner.velocity.append(self.runner.velocity[-1] \
#                                 + (self.runner.F         \
#                                     - (1/self.runner.tau) * self.runner.velocity[-1] \
#                                     - self.g*np.sin(theta) \
#                                     - (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.runner.velocity[-1] + headwind)**2)/self.runner.mass)*self.dt)
#         energy_change = (self.runner.sigma - self.runner.F*self.runner.velocity[-1])*self.dt
#         self.runner.energy.append(self.runner.energy[-1] + (energy_change if energy_change < 0 else 0))

#     def phase_2(self, theta:float = 0.0, headwind:float = 0.0) -> None:
#         """
#         This function provides the Constant Velocity Phase Logic
#         """
#         self.runner.velocity.append(self.const_v)
#         self.runner.energy.append(self.runner.energy[-1] \
#                                 + (self.runner.sigma \
#                                      - self.const_v*(self.const_v/self.runner.tau \
#                                             + self.g*np.sin(theta) \
#                                             + (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.const_v + headwind)**2)/self.runner.mass) \
#                                 - (self.runner.k*self.const_v**2*self.time_elapsed[-1])/self.runner.tau)*self.dt)

#     def phase_3(self, theta:float = 0.0, headwind:float = 0.0) -> None:
#         """
#         This function provides the Deceleration Phase Logic
#         """
#         self.runner.velocity.append(self.runner.velocity[-1] \
#                                 +  (self.runner.sigma/self.runner.velocity[-1] \
#                                 - (1/self.runner.tau)*self.runner.velocity[-1] \
#                                 - self.g*np.sin(theta) \
#                                 - (0.5*self.terrain.rho*self.runner.cd*self.runner.area*(self.runner.velocity[-1] + headwind)**2)/self.runner.mass \
#                                 - (self.runner.k*self.const_v**2*self.time_elapsed[-1])/(self.runner.tau*self.runner.velocity[-1]))*self.dt)
#         self.runner.energy.append(0.0)
    
#     def _constant_velocity(self):
#         """
#         Calculates the constant velocity during phase 2 based on the model equations.
#         """
#         const_t = 1/(2*self.runner.sigma) * ((self.runner.E0**2 + (4*self.runner.sigma*self.target_dist**2)/self.runner.tau)**0.5 - self.runner.E0)
#         const_v = (((self.runner.E0*self.runner.tau)/const_t) + self.runner.sigma*self.runner.tau)**0.5
#         return const_v

#     def loop(self):
#         """
#         Runs the simulation until the target distance is reached
#         """
#         # TODO - make this into a monte carlo simulation, preferbaly with vectorized operations
#         # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
#         wbgt = self.terrain.get_wbgt()
#         self.runner.sigma *= 1 - self.terrain.psi*max(0, wbgt - 15)
#         # print(f"Initial Energy Supply Rate (sigma): {self.runner.sigma:.2f} m^2/s^3")
#         # print(f"Calculated WBGT: {wbgt:.2f} °C")
#         # print(f"Adjusted Energy Supply Rate (sigma): {self.runner.sigma:.2f} m^2/s^3")
#         # while self.time_elapsed[-1] < 10:

#         # create an array to store the results of multiple simulations
#         results = np.zeros((self.num_sim, ))

#         while self.distance_covered < self.target_dist:
#             self.step()

#         print(f"Time Elapsed: {self.time_elapsed[-1]} seconds")
#         print(f"Distance Covered: {self.distance_covered} meters")
#         print(f"t1 (start of constant velocity phase): {self.t1} seconds")
#         print(f"t2 (start of deceleration phase): {self.t2} seconds")
#         print(f"Constant Velocity (v): {self.const_v} m/s")

# @dataclass
# class SimConfig:
#     F: float          # Max thrust (m/s^2)
#     E0: float         # Initial energy (m^2/s^2)
#     tau: float        # Resistance coefficient (s)
#     sigma: float      # Energy supply rate (m^2/s^3)
#     gamma: float      # Fatigue constant (dimensionless)
#     drag_coefficient: float  # Drag coefficient for a runner (dimensionless)
#     frontal_area: float      # Frontal area of the runner (m^2)
#     mass: float              # Mass of the runner (kg)

# if __name__ == "__main__":
#     # for a more reasonable run 
#     runner = Runner(params={
#         # these are mens average
#         'F': 10.5,          # Max thrust (m/s^2) (9.0–12.0)
#         'E0': 2200.0,       # Initial energy (m^2/s^2) (1800–2600)
#         'tau': 1.0,         # Resistance coefficient (s) (0.8–1.2)
#         'sigma': 45.0,      # Energy supply rate (m^2/s^3) (35-55)
#         'gamma': 5.0e-5,    # Fatigue constant (3e-5 to 8e-5)

#         # 'drag_coefficient': 0.65,  # Drag coefficient for a runner (dimensionless)
#         'drag_coefficient': 1.0,  # Drag coefficient for a runner (dimensionless) (0.9-1.1)
#         'frontal_area': 0.47,      # Frontal area of the runner (m^2) (0.4-0.55)
#         'mass': 70.0              # Mass of the runner (kg) (60-80)

#     })
#     # # specify the runner parameters based on the paper's values for now
#     # runner = Runner(params={
#     #     # these are mens average
#     #     'F': 14.36,         # Max thrust (m/s^2)
#     #     'tau': 0.739,       # Resistance coefficient (s)
#     #     'E0': 3114.0,       # Initial energy (m^2/s^2)
#     #     'sigma': 58.0,      # Energy supply rate (m^2/s^3)
#     #     'gamma': 4.08e-5,    # Fatigue constant 

#     #     'drag_coefficient': 0.65,  # Drag coefficient for a runner (dimensionless)
#     #     # 'drag_coefficient': 1.0,  # Drag coefficient for a runner (dimensionless)
#     #     'frontal_area': 0.5,      # Frontal area of the runner (m^2)

#     # })

#     terrain = Terrain(params={
#         'air_density': 1.225,  # air density at sea level (kg/m^3)
#         'convection_heat_transfer_coefficient': 10.0,  # convection heat transfer coefficient (W/m^2K)
#         'absorption_coefficient': 0.7,  # absorption coefficient for solar radiation (dimensionless)
#         'psi': 0.005,  # weighting factor for the drop in aerobic power per temperature (dimensionless)
#     }, 
#     csv_data="runs/2025-10-10_10-42/2025-10-10_10-42_streams.csv",  # flat terrain for now, can be modified to include elevation changes
#     json_data="runs/2025-10-10_10-42/2025-10-10_10-42_overall.json")
#     # print(terrain.df.head())  # check the terrain data

#     sim = MarathonSimulation(runner, terrain, target_distance=4300, dt=0.01, num_sim=2)
#     # sim = MarathonSimulation(runner, terrain, target_distance=4300, dt=0.01, const_v=4.0)
#     sim.loop()

#     # plotting results
#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(sim.runner.velocity, label='Velocity (m/s)')
#     plt.title('Runner Velocity Over Time')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Velocity (m/s)')
#     plt.legend()
#     plt.subplot(2, 1, 2)
#     plt.plot(sim.runner.energy, label='Energy (J)')
#     plt.title('Runner Energy Over Time')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Energy (J)')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # plotting elevation profile
#     plt.figure(figsize=(12, 4))
#     plt.plot(sim.time_elapsed, sim.elevation_profile, label='Elevation Profile (radians)')
#     plt.title('Elevation Profile Over Distance')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Grade (radians)')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # plotting headwind profile
#     plt.figure(figsize=(12, 4))
#     plt.plot(sim.time_elapsed, sim.headwind_profile, label='Headwind Profile (m/s)')
#     plt.title('Headwind Profile Over Distance')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Headwind Speed (m/s)')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
