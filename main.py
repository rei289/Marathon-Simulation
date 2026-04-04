"""Test script to deploy directly in GCP."""
import os

import numpy as np

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
    # get bucket name from environment variable
    bucket_name = os.getenv("BUCKET_NAME")
    # get input dataframe for the simulation
    df_input = create_dataframes(params, sim_cfg.num_sim)

    csv_data=None
    json_data=None

    sim = MonteCarloSimulation(sim_cfg, df_input=df_input, csv_data=csv_data, json_data=json_data)

    # perform the simulation
    sim.run()
    sim.save_to_cloud_results(bucket_name)

    # print results
    print(f"Average finish time (s): {np.mean(sim.finish_time)}")
    print(f"Finish times (s): {sim.finish_time} seconds")
