"""
This is the main file for retrieving data and running the marathon simulation.
"""
from simulation.data_classes import Params, SimConfig
from simulation.monte_carlo_simulation import (
    MonteCarloSimulation,
    create_dataframes,
    spaghetti_plot,
)

params = Params(
    F=[9.0, 12.0],
    E0=[1800.0, 2600.0],
    tau=[0.8, 1.2],
    sigma=[35.0, 55.0],
    gamma=[3e-5, 8e-5],
    drag_coefficient=[0.9, 1.1],
    frontal_area=[0.4, 0.55],
    mass=[60.0, 80.0],
    rho=[1.225],
    convection=[10.0],
    alpha=[0.6, 0.8],
    psi=[0.003, 0.007]
)

sim_cfg = SimConfig(
    target_dist=4300,
    num_sim=10,
    dt=0.1,
    max_steps=10000,
    const_v=7.0,
    t1=None,
    t2=None
)


if __name__ == "__main__":


    # UNCOMMENT THIS TO RETRIEVE DATA FROM STRAVA AND VISUAL CROSSING
    
    # # Load configuration from globals.json
    # num_runs = extract_global_json("num_runs")
    # output_folder = extract_global_json("output_folder")
    # # Run the main function
    # retrieve_run(num_runs, output_folder)


    # get input dataframe for the simulation

    df_input = create_dataframes(params, sim_cfg.num_sim)


    
    csv_data="runs/2025-10-10_10-42/2025-10-10_10-42_streams.csv"
    json_data="runs/2025-10-10_10-42/2025-10-10_10-42_overall.json"
    # print(terrain.df.head())  # check the terrain data

    sim = MonteCarloSimulation(sim_cfg, df_input=df_input, csv_data=csv_data, json_data=json_data)

    # perform the simulation
    sim.run()

    # plotting results
    spaghetti_plot(sim)
    # histogram_plot(sim)