"""Use this script to fit the model parameters to the data."""

import json
from io import BytesIO
from logging import Logger
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import stride_sim_rust
from google.cloud import storage
from scipy import signal

from src.simulation.monte_carlo_simulation import MonteCarloSimulation
from src.utilis.logger import StrideSimLogger


class ModelFitter:
    """Use class to fit the model parameters to the data."""

    def __init__(self, logger: Logger, run_data: dict) -> None:
        """Initialize the model fitter with the run data."""
        self.run_data = run_data
        self.logger = logger

        # get the actual velocity and time arrays from the run data and make it into a pandas dataframe
        v_obs = self.run_data["velocity"]
        t_obs = self.run_data["time"]
        self.df_obs = pd.DataFrame({
            "time": t_obs,
            "velocity": v_obs,
        })

        self.config = stride_sim_rust.SimulationConfig(
            target_dist=self.run_data["total_distance"],
            num_sim=1,
            dt=0.1,
            max_steps=200_000,
            sample_rate=1.0,  # sample every 1 seconds
            result_path=None,
        )

        self.weather = stride_sim_rust.Weather(
            temperature=self.run_data["temperature"],
            humidity=self.run_data["humidity"],
            solar_radiation=self.run_data["solar_radiation"],
        )

        self.course = stride_sim_rust.CourseProfile(
            distance=self.run_data["distance"],
            grade=self.run_data["grade"],
            headwind=self.run_data["headwind"],
        )

    def run_simulation(self, params: dict) -> pd.DataFrame:
        """Run the simulation with the given parameters and return the simulated velocity and time as a dataframe."""
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
        sim = MonteCarloSimulation(self.logger, runners, self.config, self.weather, self.course)
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

def read_run_data(logger: Logger, logger_mgr: StrideSimLogger, date: str, parquet_path: str, json_path: str) -> dict:
    """Use to read the run data from the given paths."""
    # get the bucket name from the logger manager
    bucket_name = logger_mgr.bucket_name
    runs_folder = logger_mgr.folder_name.split("/")[0]

    # first determine where the data is stored based on the execution environment
    if logger_mgr.execution_env == "local":
        logger.info(f"Reading run data from local file system at: {parquet_path} and {json_path}")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        json_content = Path(json_path).read_text()
        overall_data = json.loads(json_content)
    elif logger_mgr.execution_env == "gcp":
        logger.info(f"Reading run data from GCP bucket at: {parquet_path} and {json_path}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        parquet_blob_path = f"{runs_folder}/{date}/streams.parquet"
        json_blob_path = f"{runs_folder}/{date}/overall.json"

        parquet_blob = bucket.blob(parquet_blob_path)
        json_blob = bucket.blob(json_blob_path)

        if not parquet_blob.exists():
            error = f"GCS object not found: gs://{bucket_name}/{parquet_blob_path}"
            logger.error(error)
            raise FileNotFoundError(error)
        if not json_blob.exists():
            error = f"GCS object not found: gs://{bucket_name}/{json_blob_path}"
            logger.error(error)
            raise FileNotFoundError(error)

        parquet_bytes = parquet_blob.download_as_bytes()
        df = pd.read_parquet(BytesIO(parquet_bytes), engine="pyarrow")

        overall_data = json.loads(json_blob.download_as_text())
    else:
        error = f"Unknown execution environment: {logger_mgr.execution_env}. Cannot read run data."
        logger.error(error)
        raise ValueError(error)

    # create a dictionary to hold the run data and return it
    # expected: time (s), distance (m), velocity (m/s)
    # remove the first row of the dataframe since it is always 0 and can cause issues with the fitting
    df = df.iloc[1:].copy()
    df["time_datetime"] = pd.to_datetime(df["time_datetime"])
    t_obs = (df["time_datetime"] - df["time_datetime"].iloc[0]).dt.total_seconds().to_numpy()
    d_obs = df["distance_m"].to_numpy()
    v_obs = df["smooth_velocity_mps"].to_numpy()
    grade = df["grade_percent"].to_numpy()
    headwind = df["headwind_mps"].to_numpy()

    json_path = Path(json_path)
    content = json_path.read_text()
    overall_data = json.loads(content)

    return {
        "time": t_obs,
        "distance": d_obs,
        "velocity": v_obs,
        "grade": grade,
        "headwind": headwind,
        "total_distance": overall_data["distance"],
        "temperature": overall_data["weather"]["temp"],
        "humidity": overall_data["weather"]["humidity"],
        "solar_radiation": overall_data["weather"]["solarradiation"],
    }


def model_fitting(logger: Logger, logger_mgr: StrideSimLogger, date: str) -> None:
    """Use to fit the model parameters to the data."""
    # get the bucket name from the logger manager
    bucket_name = logger_mgr.bucket_name
    train_folder = logger_mgr.folder_name.split("/")[0]

    parquet_data = f"{bucket_name}/01_runs/{date}/streams.parquet"
    json_data = f"{bucket_name}/01_runs/{date}/overall.json"

    # read the observed data
    run_data = read_run_data(logger, logger_mgr, date, parquet_data, json_data)

    # create the model fitter class
    fitter = ModelFitter(logger, run_data)

    # create a study object
    study = optuna.create_study(direction="minimize")
    study.optimize(fitter.objective_function, n_trials=100, show_progress_bar=False)

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

    # lastly add the run date and cutoff frequency
    data["noise_cutoff_freq"] = cutoff_freq
    data["run_date"] = date

    # save the model coefficients to a json file in the output folder
    output_blob_path = f"{train_folder}/{date}/model_coefficients.json"


    if logger_mgr.execution_env == "local":
        output_folder_path = Path(f"{bucket_name}/{train_folder}/{date}")
        output_folder_path.mkdir(parents=True, exist_ok=True)
        file = output_folder_path / "model_coefficients.json"
        file.write_text(json.dumps(data, indent=4))
        logger.info(f"Model coefficients saved locally at: {file}")

    elif logger_mgr.execution_env == "gcp":
        logger.info(f"Saving model coefficients to GCP: gs://{bucket_name}/{output_blob_path}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(output_blob_path)
        blob.upload_from_string(
            json.dumps(data, indent=4),
            content_type="application/json",
        )
        logger.info("Model coefficients saved to GCP.")
    else:
        error = f"Unknown execution environment: {logger_mgr.execution_env}. Cannot save model coefficients."
        logger.error(error)
        raise ValueError(error)


