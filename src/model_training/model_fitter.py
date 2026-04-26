"""Use this script to fit the model parameters to the data."""

import json
from logging import Logger
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import stride_sim_rust
from scipy import signal
from scipy.signal import butter, filtfilt

from src.simulation.monte_carlo_simulation import MonteCarloSimulation


class ModelFitter:
    """Use class to fit the model parameters to the data."""

    def __init__(self, logger: Logger, parquet_path: str, json_path: str) -> None:
        """Initialize the model fitter with the paths to the csv and json data."""
        self.parquet_path = parquet_path
        self.json_path = json_path
        self.logger = logger

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        # expected: time (s), distance (m), velocity (m/s)
        df["time_datetime"] = pd.to_datetime(df["time_datetime"])
        t_obs = (df["time_datetime"] - df["time_datetime"].iloc[0]).dt.total_seconds().to_numpy()
        d_obs = df["distance_m"].to_numpy()
        v_obs = df["smooth_velocity_mps"].to_numpy()
        grade = df["grade_percent"].to_numpy()
        headwind = df["headwind_mps"].to_numpy()

        json_path = Path(json_path)
        content = json_path.read_text()
        overall_data = json.loads(content)

        self.run_data = {
            "time": t_obs,
            "distance": d_obs,
            "velocity": v_obs,
            "grade": grade,
            "headwind": headwind,
            "total_distance": overall_data["distance"],
            "temperature": overall_data["weather"]["temp"],
            "humidity": overall_data["weather"]["humidity"],
            "solarradiation": overall_data["weather"]["solarradiation"],
        }

        # get the actual velocity and time arrays from the run data and make it into a pandas dataframe
        v_obs = self.run_data["velocity"]
        t_obs = self.run_data["time"]
        self.df_obs = pd.DataFrame({
            "time": t_obs,
            "velocity": v_obs,
        })

    def run_simulation(self, params: dict) -> pd.DataFrame:
        """Run the simulation with the given parameters and return the simulated velocity and time as a dataframe."""
        config = stride_sim_rust.SimulationConfig(
            target_dist=43_000,
            num_sim=1,
            dt=0.1,
            max_steps=200_000,
            sample_rate=1.0,  # sample every 2 seconds
            result_path=None,
        )

        weather = stride_sim_rust.Weather(
            temperature=20.0,
            humidity=0.50,
            solar_radiation=800.0,
        )

        course = stride_sim_rust.CourseProfile(
            distance=[0.0, 10_000.0, 20_000.0, 30_000.0, 42_195.0],
            grade=[0.0, 0.0, 0.0, 0.0, 0.0],
            headwind=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

        runners = [stride_sim_rust.RunnerParams(
            runner_id=0,
            f_max=params["f_max"],
            e_init=params["e_init"],
            tau=params["tau"],
            sigma=params["sigma"],
            gamma=params["gamma"],
            drag_coefficient=params["drag_coefficient"],
            frontal_area=params["frontal_area"],
            mass=params["mass"],
            rho=params["rho"],
            convection=params["convection"],
            alpha=params["alpha"],
            psi=params["psi"],
            const_v=params["const_v"],
            pacing=params["pacing"],
        )]

        # run simulation
        sim = MonteCarloSimulation(self.logger, runners, config, weather, course)
        sim_results = sim.run_collect()

        # get the velocity and time arrays from the simulation and make it into a pandas dataframe
        v_sim = sim_results[2]  # velocity
        t_sim = sim_results[1]  # time

        return pd.DataFrame({
            "time": t_sim,
            "velocity": v_sim,
        })

    def objective_function(self, trial: optuna.Trial) -> float:
        """Objective function to minimize the difference between the observed and simulated finish times."""
        f_max = trial.suggest_float("f_max", 9.0, 12.0)
        e_init = trial.suggest_float("e_init", 1800.0, 2600.0)
        tau = trial.suggest_float("tau", 0.8, 1.2)
        sigma = trial.suggest_float("sigma", 20.0, 35.0)
        gamma = trial.suggest_float("gamma", 3e-5, 8e-5)
        drag_coefficient = trial.suggest_float("drag_coefficient", 0.9, 1.1)
        frontal_area = trial.suggest_float("frontal_area", 0.4, 0.55)
        mass = trial.suggest_float("mass", 55.0, 58.0)
        rho = trial.suggest_float("rho", 1.225, 1.225)
        convection = trial.suggest_float("convection", 10.0, 10.0)
        alpha = trial.suggest_float("alpha", 0.6, 0.8)
        psi = trial.suggest_float("psi", 0.003, 0.007)
        pacing = trial.suggest_categorical("pacing", ["constant velocity", "even effort"])
        const_v = trial.suggest_float("const_v", 4.0, 5.0)


        df_sim = self.run_simulation(params = {
            "f_max": f_max,
            "e_init": e_init,
            "tau": tau,
            "sigma": sigma,
            "gamma": gamma,
            "drag_coefficient": drag_coefficient,
            "frontal_area": frontal_area,
            "mass": mass,
            "rho": rho,
            "convection": convection,
            "alpha": alpha,
            "psi": psi,
            "pacing": pacing,
            "const_v": const_v,
        })

        # create a mask to filter the simulation dataframe to only include the time points that are present in the observed dataframe
        df_sim_masked = pd.DataFrame({
            "time": self.df_obs["time"].to_numpy(),
            "velocity": np.interp(self.df_obs["time"].to_numpy(), df_sim["time"].to_numpy(), df_sim["velocity"].to_numpy()),
        })

        # calculate the mean squared error between the observed and simulated velocities at the masked time points
        return np.mean((df_sim_masked["velocity"] - self.df_obs["velocity"]) ** 2)


def generate_realistic_noise(length: int, target_rmse: float, cutoff_hz: float = 0.1, fs: float = 1.0, seed: int = 42) -> np.ndarray:
    """Use to generate realistic noise that can be added to the simulation output to better match the observed data."""
    # generate raw Gaussian noise
    rng = np.random.default_rng(seed)
    raw_noise = rng.normal(0, 1, length)

    # design a Low-Pass Butterworth Filter
    # nyquist frequency is half the sampling rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(N=2, Wn=normal_cutoff, btype="low", analog=False)

    # apply the filter (filtfilt prevents phase shift/delay)
    smooth_noise = filtfilt(b, a, raw_noise)

    # scale it to match the Optuna Error
    # scale by the ratio of target_rmse to the current std of the smooth signal
    current_std = np.std(smooth_noise)
    return smooth_noise * (target_rmse / current_std)

def automatic_cutoff(velocity_residuals: pd.Series, fs: float = 1.0, threshold: float = 0.90) -> float:
    """Use to automatically determine the cutoff frequency."""
    # compute PSD
    freqs, psd = signal.welch(velocity_residuals.to_numpy(), fs, nperseg=256)

    # compute cumulative power
    cumulative_psd = np.cumsum(psd)
    total_power = cumulative_psd[-1]

    # find where the power crosses the threshold (e.g., 90%)
    cutoff_idx = np.where(cumulative_psd >= total_power * threshold)[0][0]
    return freqs[cutoff_idx]

def model_fitting(logger: Logger, date: str, bucket_name: str, train_folder: str = "02_trainings") -> None:
    """Use to fit the model parameters to the data."""
    # determine which run to use for fitting the model parameters
    parquet_data = f"{bucket_name}/01_runs/{date}/streams.parquet"
    json_data = f"{bucket_name}/01_runs/{date}/overall.json"

    # create the model fitter class
    fitter = ModelFitter(logger, parquet_path = parquet_data, json_path = json_data)

    # create a study object
    study = optuna.create_study(direction="minimize")
    study.optimize(fitter.objective_function, n_trials=50)

    # put this in a json file
    data = study.best_params
    data["mse"] = study.best_value

    # now determine cutoff frequency for the noise generation
    df_sim = fitter.run_simulation(params = data)
    # create a mask to filter the simulation dataframe to only include the time points that are present in the observed dataframe
    df_sim_masked = pd.DataFrame({
        "time": fitter.df_obs["time"].to_numpy(),
        "velocity": np.interp(fitter.df_obs["time"].to_numpy(), df_sim["time"].to_numpy(), df_sim["velocity"].to_numpy()),
    })

    velocity_residuals = fitter.df_obs["velocity"] - df_sim_masked["velocity"]
    cutoff_freq = automatic_cutoff(velocity_residuals)

    data["noise_cutoff_freq"] = cutoff_freq

    # lastly add the run date to know which run these parameters correspond to
    data["run_date"] = date

    # delete the model_coefficients.json if it exists
    output_folder_path = Path(f"{bucket_name}/{train_folder}/{date}")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    file = output_folder_path / "model_coefficients.json"

    file.write_text(json.dumps(data, indent=4))

