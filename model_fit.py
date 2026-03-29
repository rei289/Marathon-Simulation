"""Use this script to fit the model parameters to the data."""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from simulation.data_classes import PacingContext, Params, SimConfig
from simulation.monte_carlo_simulation import MonteCarloSimulation, create_dataframes
from simulation.pacing_strategy import ConstantPaceStrategy, EvenEffortStrategy


class ModelFitter:
    """Use class to fit the model parameters to the data."""

    def __init__(self, csv_path: str, json_path: str) -> None:
        """Initialize the model fitter with the paths to the csv and json data."""
        self.csv_path = csv_path
        self.json_path = json_path
        self.run_data = load_run_data(csv_path, json_path)

    def objective_function(self, trial) -> float:
        """Objective function to minimize the difference between the observed and simulated finish times."""
        F = trial.suggest_float("F", 9.0, 12.0)
        E0 = trial.suggest_float("E0", 1800.0, 2600.0)
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
        pacing = trial.suggest_categorical("pacing", ["constant", "even_effort"])
        const_v = trial.suggest_float("const_v", 4.0, 5.0)


        sim_cfg = SimConfig(
            target_dist=self.run_data["total_distance"],
            num_sim=1,
            dt=0.1,
            max_steps=20000,
            const_v=const_v,
            t1=None,
            t2=None,
        )

        pacing_strategy = ConstantPaceStrategy(sim_cfg) if pacing == "constant" else EvenEffortStrategy(sim_cfg)

        # create input dataframe
        df_input = pd.DataFrame({
            "F": [F],
            "E0": [E0],
            "tau": [tau],
            "sigma": [sigma],
            "gamma": [gamma],
            "drag_coefficient": [drag_coefficient],
            "frontal_area": [frontal_area],
            "mass": [mass],
            "rho": [rho],
            "convection": [convection],
            "alpha": [alpha],
            "psi": [psi],
        })

        # run simulation
        sim = MonteCarloSimulation(sim_cfg, pacing_strategy, df_input=df_input, csv_data=self.csv_path, json_data=self.json_path)
        sim.run()

        # get the velocity and time arrays from the simulation and make it into a pandas dataframe
        v_sim = sim.velocity[:, 0]
        t_sim = sim.time_elapsed
        df_sim = pd.DataFrame({
            "time": t_sim,
            "velocity": v_sim,
        })

        # get the actual velocity and time arrays from the run data and make it into a pandas dataframe
        v_obs = self.run_data["velocity"]
        t_obs = self.run_data["time"]
        df_obs = pd.DataFrame({
            "time": t_obs,
            "velocity": v_obs,
        })

        # create a mask to filter the simulation dataframe to only include the time points that are present in the observed dataframe
        df_sim_masked = pd.DataFrame({
            "time": df_obs["time"].to_numpy(),
            "velocity": np.interp(df_obs["time"].to_numpy(), df_sim["time"].to_numpy(), df_sim["velocity"].to_numpy()),
        })

        # calculate the mean squared error between the observed and simulated velocities at the masked time points
        return np.mean((df_sim_masked["velocity"] - df_obs["velocity"]) ** 2)


def load_run_data(csv_path: str, json_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use to load the actual data from a run for fitting the model parameters."""
    df = pd.read_csv(csv_path)
    # expected: time (s), distance (m), velocity (m/s)
    df["time_datetime"] = pd.to_datetime(df["time_datetime"])
    t_obs = (df["time_datetime"] - df["time_datetime"].iloc[0]).dt.total_seconds().to_numpy()
    d_obs = df["distance_m"].to_numpy()
    v_obs = df["smooth_velocity_mps"].to_numpy()
    grade = df["grade_percent"].to_numpy()
    headwind = df["headwind_mps"].to_numpy()

    with open(json_path, "r") as f:
        overall_data = json.load(f)


    return {
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

def constant_data(run_data: dict) -> dict:
    """Use to add constant terms (like runner mass) to the data for fitting the model parameters."""
    return {
        "mass": run_data["mass"],
        "rho": run_data["rho"],
        "convection": run_data["convection"],
    }


if __name__ == "__main__":
    # determine which run to use for fitting the model parameters
    date = "2025-08-06_20-14"
    csv_data = f"runs/{date}/{date}_streams.csv"
    json_data = f"runs/{date}/{date}_overall.json"

    # get input dataframe for the simulation
    run_data = load_run_data(csv_data, json_data)

    # create the model fitter class
    fitter = ModelFitter(csv_path = csv_data, json_path = json_data)

    # create a study object
    study = optuna.create_study(direction="minimize")
    study.optimize(fitter.objective_function, n_trials=50)

    # put this in a json file
    data = study.best_params
    data["error"] = study.best_value

    # delete the model_coefficients.json if it exists

    if Path.exists("model_coefficients.json"):
        Path.unlink("model_coefficients.json")

    with open("model_coefficients.json", "w") as f:
        json.dump(data, f, indent=4)

    # print(f"\nBest parameters: {study.best_params}")
    # print(f"Lowest error found: {study.best_value}")
