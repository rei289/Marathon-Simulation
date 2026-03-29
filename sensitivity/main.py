"""Use script to run the sensitivity analysis."""

from dataclasses import asdict

import pandas as pd
from sensitivity_shap import generate_lhs_samples, run_shap_analysis

from simulation.data_classes import Params, SimConfig
from simulation.monte_carlo_simulation import MonteCarloSimulation

if __name__ == "__main__":
    params = Params(
        f_max=[9.0, 12.0],
        e_init=[1800.0, 2600.0],
        tau=[0.8, 1.2],
        sigma=[35.0, 55.0],
        gamma=[3e-5, 8e-5],
        drag_coefficient=[0.9, 1.1],
        frontal_area=[0.4, 0.55],
        mass=[60.0, 80.0],
        rho=[1.2, 1.4],
        convection=[10.0, 12.0],
        alpha=[0.6, 0.8],
        psi=[0.003, 0.007],
    )

    sim_cfg = SimConfig(
        target_dist=4300,
        num_sim=10000,
        dt=0.1,
        max_steps=10000,
        const_v=5.0,
        pacing="constant velocity",
    )

    params_dict = asdict(params)

    n_samples = sim_cfg.num_sim
    X = generate_lhs_samples(n_samples, len(params_dict), params_dict)

    # convert the scaled samples to a DataFrame with appropriate column names
    df_input = pd.DataFrame(X, columns=params_dict.keys())

    # run the simulation for each sample and collect the outputs
    sim = MonteCarloSimulation(sim_cfg, df_input, csv_data=None, json_data=None)

    sim.run()

    y = sim.finish_time  # run the simulation for each sample and collect the outputs

    # run SHAP analysis on the collected data
    run_shap_analysis(df_input, y)
