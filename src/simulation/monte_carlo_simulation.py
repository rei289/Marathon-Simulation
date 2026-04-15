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
from io import BytesIO
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage

from src.simulation.data_classes import PacingContext, Params, SimConfig
from src.simulation.pacing_strategy import PacingStrategy
from src.utilis.helper import get_constant_params


class MonteCarloSimulation:
    """Class to run a Monte Carlo simulation of the marathon model with varying parameters and conditions."""

    def __init__(self, logger: Logger, cfg: SimConfig, df_input: pd.DataFrame, parquet_data: str|None, json_data: str|None) -> None:
        """Use to initialize the simulation with the given configuration, input parameters, and optional course and weather data."""
        self.logger = logger
        self.target_dist = cfg.target_dist
        self.num_sim = cfg.num_sim
        self.dt = cfg.dt
        self.max_steps = cfg.max_steps

        # create unique simulation ids for every trajectory which we can use to track the results across different data structures
        self.sim_number = np.array([f"sim_{i}" for i in range(self.num_sim)])

        self.df = pd.read_parquet(parquet_data, engine="pyarrow")[["distance_m", "grade_percent", "headwind_mps"]].fillna(0) \
                                        if parquet_data is not None \
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
        self.strat = PacingStrategy(cfg)
        self.logger.info(f"Running Monte Carlo Simulation with {self.num_sim} simulations.")

        self.g = get_constant_params("gravity")  # gravitational acceleration (m/s^2)

        # get the parameter values from the input dataframe
        for input_var in df_input.columns:
            setattr(self, f"{input_var}_values", df_input[input_var].to_numpy().copy())

        # before we start, calculate the effective aerobic supply using the WBGT to adjust the sigma value based on the heat stress
        self.sigma_values *= np.ones(self.num_sim) - self.psi_values*np.maximum(0, self._get_wbgt() - get_constant_params("reference_temp"))
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

    def math_model(self, f_desired: np.ndarray, theta:np.ndarray, headwind:np.ndarray) -> None:
        """Use to provide the equation logic for the simulation, incorporating the effects of terrain and weather on the runner's performance."""
        # calculate all the resistive forces
        f_resistance = self.g*np.sin(theta) \
        + (0.5*self.rho_values*self.drag_coefficient_values*self.frontal_area_values*(self.velocity[self.iteration] + headwind)**2)/self.mass_values

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
        # determine the target velocity based on the pacing strategy and current conditions
        m1 = self.pacing_strat_values == "constant velocity"
        m2 = self.pacing_strat_values == "even effort"

        v_target = np.zeros(self.num_sim)
        if m1.any():
            v_target[m1] = self.strat.constant_pace(ctx=PacingContext(
                const_v=self.const_v_values[m1],
                velocity=self.velocity[self.iteration][m1],
                energy=self.energy[self.iteration][m1],
                theta=theta[m1],
                headwind=headwind[m1],
                tau=self.tau_values[m1],
                mass=self.mass_values[m1],
                rho=self.rho_values[m1],
                drag_coefficient=self.drag_coefficient_values[m1],
                frontal_area=self.frontal_area_values[m1],
                f_max=self.f_max_values[m1],
            ))

        # calculate the desired force
        f_desired = (v_target - self.velocity[self.iteration])/self.dt
        if m2.any():
            f_desired[m2] = self.strat.even_effort_pace(ctx=PacingContext(
                const_v=self.const_v_values[m2],
                velocity=self.velocity[self.iteration][m2],
                energy=self.energy[self.iteration][m2],
                theta=theta[m2],
                headwind=headwind[m2],
                tau=self.tau_values[m2],
                mass=self.mass_values[m2],
                rho=self.rho_values[m2],
                drag_coefficient=self.drag_coefficient_values[m2],
                frontal_area=self.frontal_area_values[m2],
                f_max=self.f_max_values[m2],
            ))
        self.math_model(f_desired, theta, headwind)

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

            # we also want to stop any simulation where velocity has dropped to zero and energy is depleted
            just_stopped = self.active & (self.velocity[step] <= 1e-8) & (self.energy[step] <= 1e-8)
            self.active[just_stopped] = False

            if not self.active.any():
                break                            # all sims done → early exit

            # lastly update time and distance for active simulations
            self.time_elapsed[step + 1] = self.time_elapsed[step] + self.dt
            self.distance_covered[step + 1] = self.distance_covered[step] + np.where(self.active, self.velocity[step] * self.dt, 0.0)

            self.iteration = step + 1

    def save_to_cloud_results(self, bucket_name: str, simulation_folder: str, job_id: str, ts: str) -> None:
        """Use to save the results metadata, and configuration of the simulation."""
        # create a unique job id and base path for storing results in the bucket
        base_path = f"{simulation_folder}/{job_id}"
        self.logger.info(f"Saving results to cloud storage at: {bucket_name}/{base_path}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # save core results
        # keep only the simulated steps
        valid_step_mask = ~np.isnan(self.time_elapsed)
        step_idx = np.where(valid_step_mask)[0]
        time_vals = self.time_elapsed[valid_step_mask]

        # shape of the following are: (n_valid_steps, num_sim)
        dist = self.distance_covered[valid_step_mask, :]
        vel = self.velocity[valid_step_mask, :]
        eng = self.energy[valid_step_mask, :]

        n_steps = len(step_idx)
        n_sim = self.num_sim

        df_results = pd.DataFrame({
            "step": np.repeat(step_idx, n_sim),
            "time_s": np.repeat(time_vals, n_sim),
            "sim_number": np.tile(self.sim_number, n_steps),
            "distance_m": dist.reshape(-1),
            "velocity_mps": vel.reshape(-1),
            "energy": eng.reshape(-1),
        })

        buffer_results = BytesIO()
        df_results.to_parquet(buffer_results, index=False)
        buffer_results.seek(0)
        blob = bucket.blob(f"{base_path}/simulation_results.parquet")
        blob.upload_from_file(buffer_results, content_type="application/octet-stream")

        # save simulation configuration
        config_data = asdict(self.cfg)
        config_blob = bucket.blob(f"{base_path}/config.json")
        config_blob.upload_from_string(json.dumps(config_data), content_type="application/json")

        # save metadata
        metadata = {
            "job_id": job_id,
            "created_at": ts,
            "bucket": bucket_name,
        }

        metadata_blob = bucket.blob(f"{base_path}/metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json",
        )

    def save_to_local_results(self, bucket_name: str, simulation_folder: str, job_id: str, ts: str) -> None:
        """Use to save the results metadata, and configuration of the simulation."""
        # create a unique job id and base path for storing results in the bucket
        base_path = f"{bucket_name}/{simulation_folder}/{job_id}"
        self.logger.info(f"Saving results to local storage at: {base_path}")

        # create output folder if it doesn't exist
        output_folder_path = Path(base_path)
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # save core results
        # keep only the simulated steps
        valid_step_mask = ~np.isnan(self.time_elapsed)
        step_idx = np.where(valid_step_mask)[0]
        time_vals = self.time_elapsed[valid_step_mask]

        # shape of the following are: (n_valid_steps, num_sim)
        dist = self.distance_covered[valid_step_mask, :]
        vel = self.velocity[valid_step_mask, :]
        eng = self.energy[valid_step_mask, :]

        n_steps = len(step_idx)
        n_sim = self.num_sim

        df_results = pd.DataFrame({
            "step": np.repeat(step_idx, n_sim),
            "time_s": np.repeat(time_vals, n_sim),
            "sim_number": np.tile(self.sim_number, n_steps),
            "distance_m": dist.reshape(-1),
            "velocity_mps": vel.reshape(-1),
            "energy": eng.reshape(-1),
        })
        df_results.to_parquet(output_folder_path / "simulation_results.parquet", index=False)

        # save simulation configuration
        config_data = asdict(self.cfg)
        config_file_path = output_folder_path / "config.json"
        config_file_path.write_text(json.dumps(config_data, indent=4))

        # save metadata
        metadata = {
            "job_id": job_id,
            "created_at": ts,
            "bucket": bucket_name,
        }
        metadata_file_path = output_folder_path / "metadata.json"
        metadata_file_path.write_text(json.dumps(metadata, indent=4))

def create_dataframes(params: Params, num_sample: int, seed: int=42) -> pd.DataFrame:
    """Use to create input dataframe which can then be used to run the simulation."""
    # create a numpy array that contains the random distribution of the parameters for multiple simulations
    bounds_length_single = 1
    bounds_length_range = 2
    df = pd.DataFrame()
    rng = np.random.default_rng(seed)
    for param, bounds in asdict(params).items():
        if param == "pacing_strat":
            df[param] = rng.choice(bounds, size=num_sample)
            continue
        # if the bounds is a single value, fill the column with that value, if it's a range, sample from a uniform distribution within that range
        if len(bounds) == bounds_length_single:
            df[param] = np.full(num_sample, bounds[0])

        elif len(bounds) == bounds_length_range:
            df[param] = rng.uniform(bounds[0], bounds[1], num_sample)

        else:
            error_msg = f"Invalid bounds for parameter {param}: {bounds}"
            raise ValueError(error_msg)

    return df
