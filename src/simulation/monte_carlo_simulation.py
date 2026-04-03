"""Use to run a complex running model which incorporates the effects of terrain and weather on the runner's performance.

The model assumes throughout the run there are 3 main phases:
    1. Acceleration phase
    2. Constant velocity phase
    3. Deceleration phase

In addition to the Kellner model, this simulation incorporates:
    - Slope of the terrain (theta) which affects the runner's velocity and energy expenditure
    - Air resistance which is proportional to the square of the velocity and a drag coefficient
    - Heat stress which reduces the effective aerobic power supply (sigma) based on the Wet Bulb Globe Temperature (WBGT)
"""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.data_classes import Params, SimConfig
from src.simulation.pacing_strategy import ConstantPaceStrategy, EvenEffortStrategy, PacingContext


class MonteCarloSimulation:
    """Class to run a Monte Carlo simulation of the marathon model with varying parameters and conditions."""

    def __init__(self, cfg: SimConfig, df_input: pd.DataFrame, csv_data: str|None, json_data: str|None) -> None:
        """Use to initialize the simulation with the given configuration, input parameters, and optional course and weather data."""
        self.target_dist = cfg.target_dist
        self.num_sim = cfg.num_sim
        self.dt = cfg.dt
        self.max_steps = cfg.max_steps

        self.df = pd.read_csv(csv_data)[["distance_m", "grade_percent", "headwind_mps"]].fillna(0) if csv_data is not None \
                                        else pd.DataFrame({"distance_m": [0], "grade_percent": [0], "headwind_mps": [0]})

        if json_data is not None:
            with Path.open(json_data, "r") as f:
                self.weather_info = json.load(f)["weather"]
        else:
                self.weather_info = {"temp": 20.0, "humidity": 50.0, "solarradiation": 50.0}  # default weather conditions

        self.temp_d = self.weather_info["temp"]
        self.humidity = self.weather_info["humidity"]
        self.solar_radiation = self.weather_info["solarradiation"]

        self.cfg = cfg
        self.strat = ConstantPaceStrategy(cfg) if cfg.pacing == "constant velocity" else \
                        EvenEffortStrategy(cfg) if cfg.pacing == "even effort" else None
        print(f"Running Monte Carlo Simulation with strategy: {self.strat.pace_type} and {self.num_sim} simulations.")

        self.g = 9.81  # gravitational acceleration (m/s^2)

        # get the parameter values from the input dataframe
        for input_var in df_input.columns:
            setattr(self, f"{input_var}_values", df_input[input_var].to_numpy().copy())

        # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
        self.sigma_values *= np.ones(self.num_sim) - self.psi_values*np.maximum(0, self._get_wbgt() - 15)  # adjust sigma for heat stress
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
        self.velocity[0] = np.full(self.num_sim, 1e-8)  # start with a very small velocity to avoid division by zero
        self.energy[0] = self.e_init_values
        self.time_elapsed[0] = 0.0
        self.distance_covered[0] = np.zeros(self.num_sim)
        self.elevation_profile[0] = 0.0
        self.headwind_profile[0] = 0.0

        self.active = np.ones(self.num_sim, dtype=bool)   # True = still running

        self.const_v = cfg.const_v if cfg.const_v is not None else self._constant_velocity()
        self.iteration = 0

    def _constant_velocity(self) -> np.ndarray:
        """Use to calculate the constant velocity during phase 2 based on the model equations."""
        const_t = 1/(2*self.sigma_values)*((self.e_init_values**2+(4*self.sigma_values*self.target_dist**2)/self.tau_values)**0.5-self.e_init_values)
        return (((self.e_init_values*self.tau_values)/const_t) + self.sigma_values*self.tau_values)**0.5

    def _get_grade(self, distance: np.ndarray) -> float:
        """Use to returns the grade (theta) in radians at a given distance along the course."""
        dist = self.df["distance_m"].to_numpy()
        gp = self.df["grade_percent"].to_numpy()

        # find the two points in the distance array that are closest to the current distance
        right = np.searchsorted(dist, distance, side="right")
        right = np.clip(right, 0, len(dist) - 1)
        left = np.clip(right - 1, 0, len(dist) - 1)

        closest_idx = np.where(np.abs(distance - dist[left]) <= np.abs(dist[right] - distance), left, right)
        grade_percent = gp[closest_idx]

        return np.arctan(grade_percent / 100.0)  # convert percent grade to radians (assuming small angles)

    def _get_headwind(self, distance: np.ndarray) -> float:
        """Use to returns the headwind speed in m/s at a given distance along the course."""
        dist = self.df["distance_m"].to_numpy()
        hw = self.df["headwind_mps"].to_numpy()

        # find the two points in the distance array that are closest to the current distance
        right = np.searchsorted(dist, distance, side="right")
        right = np.clip(right, 0, len(dist) - 1)
        left = np.clip(right - 1, 0, len(dist) - 1)

        closest_idx = np.where(np.abs(distance - dist[left]) <= np.abs(dist[right] - distance), left, right)

        return hw[closest_idx]

    def _get_wbgt(self) -> np.ndarray:
        """Use to calculate the Wet Bulb Globe Temperature (WBGT) based on the weather information.

        This is a simplified calculation and can be expanded to include more factors.
        """
        temp_w = self.temp_d * np.arctan(0.151977*(self.humidity + 8.313659)**(1/2)) \
            + np.arctan(self.temp_d + self.humidity) \
            - np.arctan(self.humidity - 1.676331) \
            + 0.00391838*(self.humidity)**(3/2) * np.arctan(0.023101 * self.humidity) \
            - 4.686
        temp_g = self.temp_d + (self.solar_radiation) / (self.convection_values*self.alpha_values)  # effect of radiation on perceived temperature

        return 0.7*temp_w + 0.2*temp_g + 0.1*self.temp_d  # weighted average to get a single WBGT value

    def math_model(self, v_target: np.ndarray, theta:np.ndarray, headwind:np.ndarray) -> None:
        """Use to provide the equation logic for the simulation, incorporating the effects of terrain and weather on the runner's performance."""
        # calculate all the resistive forces
        f_resistance = self.g*np.sin(theta) \
        + (0.5*self.rho_values*self.drag_coefficient_values*self.frontal_area_values*(self.velocity[self.iteration] + headwind)**2)/self.mass_values

        # calculate amount of force we would like to apply to reach the target velocity, not accounting for resistive forces
        f_desired = (v_target - self.velocity[self.iteration])/self.dt
        # calculate the actual force applied by the runner, which is limited by the maximum thrust
        f_desired = np.minimum(self.f_max_values, f_desired)

        # check if the runner has enough energy to apply the actual force
        f_desired = np.where(self.energy[self.iteration] > 0, f_desired, \
                (self.sigma_values \
                - (self.k_values*self.velocity[self.iteration]**2*self.time_elapsed[self.iteration])/self.tau_values)/self.velocity[self.iteration])

        # calculate the change in velocity and energy based on the actual force applied
        dv = f_desired - f_resistance - (1/self.tau_values) * self.velocity[self.iteration]
        de = np.where(self.energy[self.iteration] > 0, \
                      self.sigma_values \
                      - (f_desired + f_resistance)*self.velocity[self.iteration] \
                      - (self.k_values*self.velocity[self.iteration]**2*self.time_elapsed[self.iteration])/self.tau_values, 0)

        # update velocity and energy for the next iteration
        self.velocity[self.iteration + 1] = np.maximum(0.0, self.velocity[self.iteration] + dv*self.dt) # velocity cannot be negative
        self.energy[self.iteration + 1] = np.clip(self.energy[self.iteration] + de*self.dt, 0.0, self.e_init_values) # energy cannot exceed initial

    def step(self) -> None:
        """Use to run one step of the simulation, updating the runner's velocity, energy, and distance based on the current phase of the run."""
        # determine the current terrain conditions based on the distance covered
        theta = self._get_grade(self.distance_covered[self.iteration])
        headwind = self._get_headwind(self.distance_covered[self.iteration])

        # add it to a list for plotting later
        self.elevation_profile[self.iteration] = theta[0]
        self.headwind_profile[self.iteration] = headwind[0]

        # now we calculate the new velocity and energy based on controller logic which determines the target velocity
        v_target = self.strat.get_target_velocity(ctx=PacingContext(
            dt=self.dt,
            velocity=self.velocity[self.iteration],
            energy=self.energy[self.iteration],
            theta=theta,
            headwind=headwind,
            tau=self.tau_values,
            mass=self.mass_values,
            rho=self.rho_values,
            drag_coefficient=self.drag_coefficient_values,
            frontal_area=self.frontal_area_values,
            f_max=self.f_max_values,
            g=self.g,
        ))
        self.math_model(v_target, theta, headwind)

    def run(self) -> None:
        """Use to run the simulation until the target distance is reached."""
        # loop through each time step until we reach the target distance or exceed max steps
        for step in range(self.max_steps-1):
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

    def save_results(self) -> None:
        """Use to save the results metadata, and configuration of the simulation."""
        df_results = pd.DataFrame({
            "finish_time": self.finish_time,
            "velocity_profile": self.velocity,
            "energy_profile": self.energy,
        })
        df_results.to_parquet("simulation_results.parquet", index=False)

def create_dataframes(params: Params, num_sample: int, seed: int=42) -> pd.DataFrame:
    """Use to create input dataframe which can then be used to run the simulation."""
    # create a numpy array that contains the random distribution of the parameters for multiple simulations
    bounds_length_single = 1
    bounds_length_range = 2
    df = pd.DataFrame()
    rng = np.random.default_rng(seed)
    for param, bounds in asdict(params).items():
        # if the bounds is a single value, fill the column with that value, if it's a range, sample from a uniform distribution within that range
        if len(bounds) == bounds_length_single:
            df[param] = np.full(num_sample, bounds[0])

        elif len(bounds) == bounds_length_range:
            df[param] = rng.uniform(bounds[0], bounds[1], num_sample)

        else:
            error_msg = f"Invalid bounds for parameter {param}: {bounds}"
            raise ValueError(error_msg)

    return df

def spaghetti_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot all the results of the simulation (Note costs a lot of memory)."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sim.time_elapsed, sim.velocity, color="blue", alpha=0.05, label="Velocity (m/s)")
    plt.title(f"Monte Carlo: {sim.num_sim} Simulations Runner Velocity Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.subplot(2, 1, 2)
    plt.plot(sim.time_elapsed, sim.energy, color="red", alpha=0.05, label="Energy (J)")
    plt.title(f"Monte Carlo: {sim.num_sim} Simulations Runner Energy Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.tight_layout()
    plt.show()

def histogram_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot a histogram of the finish times of the simulations."""
    plt.figure(figsize=(10, 6))
    plt.hist(sim.finish_time, bins=30, color="green", alpha=0.7)
    plt.title("Distribution of Finish Times")
    plt.xlabel("Finish Time (s)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

def elevation_headwind_plots(sim: MonteCarloSimulation) -> None:
    """Use this function to plot the elevation and headwind profiles of the course."""
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(sim.time_elapsed, sim.elevation_profile, label="Elevation Profile (radians)")
    ax[0].set_title("Elevation Profile Over Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Grade (radians)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(sim.time_elapsed, sim.headwind_profile, label="Headwind Profile (m/s)", color="orange")
    ax[1].set_title("Headwind Profile Over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Headwind Speed (m/s)")
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.show()

def distance_covered_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot the distance covered over time for all simulations."""
    # plotting distance covered
    plt.figure(figsize=(12, 4))
    plt.plot(sim.time_elapsed, sim.distance_covered[:, 0], label="Distance Covered (m)")
    plt.title("Distance Covered Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
