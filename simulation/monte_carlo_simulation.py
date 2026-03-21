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
from simulation.data_classes import SimConfig, Params
from dataclasses import asdict

# TODO: make monte carlo simulation into 1 phase rather than 3 separate functions, and use controller logic to determine pacing strategy
class MonteCarloSimulation:
    def __init__(self, cfg: SimConfig, df_input: pd.DataFrame, csv_data: str|None, json_data: str|None):
        self.target_dist = cfg.target_dist
        self.num_sim = cfg.num_sim
        self.dt = cfg.dt
        self.max_steps = cfg.max_steps

        self.df = pd.read_csv(csv_data)[['distance_m', 'grade_percent', 'headwind_mps']].fillna(0) if csv_data is not None else pd.DataFrame({"distance_m": [0], "grade_percent": [0], "headwind_mps": [0]})
        
        if json_data is not None:
            with open(json_data, 'r') as f:
                self.weather_info = json.load(f)["weather"]
        else:
                self.weather_info = {"temp": 20.0, "humidity": 50.0, "solarradiation": 50.0}  # default weather conditions

        self.temp_d = self.weather_info["temp"]
        self.humidity = self.weather_info["humidity"]
        self.solar_radiation = self.weather_info["solarradiation"]

        self.cfg = cfg 

        self.g = 9.81  # gravitational acceleration (m/s^2)

        # get the parameter values from the input dataframe
        for input_var in df_input.columns:
            setattr(self, f"{input_var}_values", df_input[input_var].values)

        # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
        self.sigma_values *= np.ones(self.num_sim) - self.psi_values*np.maximum(0, self._get_wbgt() - 15)  # adjust sigma for heat stress for each simulation
        # add the k_values which is derived from the gamma values
        self.k_values = self.gamma_values*2

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
    
    def step(self):
        """
        Runs one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run.
        """
        # determine the current terrain conditions based on the distance covered
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

def create_dataframes(params: Params, num_sample: int) -> pd.DataFrame:
    """
    Use to create input dataframe which can then be used to run the simulation
    """
    # create a numpy array that contains the random distribution of the parameters for multiple simulations
    df = pd.DataFrame()
    for param, bounds in asdict(params).items():
        # we make a very small adjustment
        if len(bounds) == 1:
            df[param] = np.full(num_sample, bounds[0])
        
        elif len(bounds) == 2:
            df[param] = np.random.uniform(bounds[0], bounds[1], num_sample)

        else:
            raise ValueError(f"Invalid bounds for parameter {param}: {bounds}")

    return df


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
