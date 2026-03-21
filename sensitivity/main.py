"""
This script serves as the main script to run the sensitivity analysis
"""
from simulation.data_classes import SimConfig, Params
from simulation.monte_carlo_simulation import MonteCarloSimulation, create_dataframes

# csv_data="runs/2025-10-10_10-42/2025-10-10_10-42_streams.csv"
# json_data="runs/2025-10-10_10-42/2025-10-10_10-42_overall.json"
# print(terrain.df.head())  # check the terrain data

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
    num_sim=1000,
    dt=0.1,
    max_steps=10000,
    const_v=None,
    t1=None,
    t2=None
)

sim = MonteCarloSimulation(sim_cfg, df_input=create_dataframes(params, sim_cfg.num_sim), csv_data=None, json_data=None)
# sim = MonteCarloSimulation(
#     SimConfig(
#         target_dist=4300,
#         num_sim=1000,
#         dt=0.1,
#         max_steps=10000,
#         const_v=None,
#         t1=None,
#         t2=None
#     ),
#     Params(
#         F=[9.0, 12.0],
#         E0=[1800.0, 2600.0],
#         tau=[0.8, 1.2],
#         sigma=[35.0, 55.0],
#         gamma=[3e-5, 8e-5],
#         drag_coefficient=[0.9, 1.1],
#         frontal_area=[0.4, 0.55],
#         mass=[60.0, 80.0],
#         rho=[1.225],
#         convection=[10.0],
#         alpha=[0.6, 0.8],
#         psi=[0.003, 0.007]
#     ), csv_data=csv_data, json_data=json_data)

# perform the simulation
sim.loop()