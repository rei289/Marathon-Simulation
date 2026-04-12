"""Test script to deploy directly in GCP."""
import time

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
    const_v=[4.0, 5.0],
    pacing_strat=["constant velocity", "even effort"],
)

sim_cfg = SimConfig(
    target_dist=4300,
    num_sim=100,
    dt=0.1,
    max_steps=20000,
)

if __name__ == "__main__":
    # @fix make this more flexible by allowing the user to specify the date of the run to use for fitting the model parameters
    df_input = create_dataframes(params, sim_cfg.num_sim)
    csv_data=None
    json_data=None

    start_time = time.time()
    sim = MonteCarloSimulation(sim_cfg, df_input=df_input, parquet_data=csv_data, json_data=json_data)

    # perform the simulation
    sim.run()
    end_time = time.time()
    print(f"Simulation time (s): {end_time - start_time}")

    # print results

    print(f"Average finish time (s): {np.mean(sim.finish_time)}")
    print(f"Finish times (s): {sim.finish_time}")
