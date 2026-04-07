"""Test script to deploy directly in GCP."""
import os

import numpy as np
from dotenv import load_dotenv

from src.simulation.data_classes import Params, SimConfig
from src.simulation.monte_carlo_simulation import (
    MonteCarloSimulation,
    create_dataframes,
)

params = Params(
    f_max=[9.0, 12.0],
    e_init=[1800.0, 2600.0],
    tau=[0.8, 1.2],
    sigma=[20.0, 35.0],
    gamma=[3e-5, 8e-5],
    drag_coefficient=[0.9, 1.1],
    frontal_area=[0.4, 0.55],
    mass=[60.0, 80.0],
    rho=[1.225],
    convection=[10.0],
    alpha=[0.6, 0.8],
    psi=[0.003, 0.007],
)

sim_cfg = SimConfig(
    target_dist=4300,
    num_sim=100,
    dt=0.1,
    max_steps=20000,
    const_v=5.0,
    pacing="constant velocity",
)

if __name__ == "__main__":
    # @fix make this more flexible by allowing the user to specify the date of the run to use for fitting the model parameters
    df_input = create_dataframes(params, sim_cfg.num_sim)
    csv_data=None
    json_data=None

    sim = MonteCarloSimulation(sim_cfg, df_input=df_input, csv_data=csv_data, json_data=json_data)

    # perform the simulation
    sim.run()

    # determine execution environment
    execution_env = os.getenv("EXECUTION_ENV", "unknown")

    if execution_env == "local":
        print("Running in local environment")
        # save results to local file system
        load_dotenv()
        bucket_name = os.getenv("BUCKET_NAME", "local_results")
        sim.save_to_local_results(bucket_name, simulation_folder="03_simulations")
    elif execution_env == "gcp":
        print("Running in GCP environment")

        # get bucket name from environment variable
        bucket_name = os.getenv("BUCKET_NAME")
        if not bucket_name:
            error = "The BUCKET_NAME environment variable is not set!"
            raise ValueError(error)

        # save results to cloud storage
        sim.save_to_cloud_results(bucket_name, simulation_folder="03_simulations")
    else:
        print(f"Running in unknown environment: {execution_env}")

    # print results
    print(f"Average finish time (s): {np.mean(sim.finish_time)}")
