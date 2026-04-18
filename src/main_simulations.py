"""Test script to deploy directly in GCP."""
import os

import numpy as np
from dotenv import load_dotenv

from src.simulation.data_classes import Params, SimConfig
from src.simulation.monte_carlo_simulation import (
    MonteCarloSimulation,
    create_dataframes,
)
from src.utilis.helper import job_id, time_now
from src.utilis.logger import StrideSimLogger

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
    const_v=[3.0, 5.0],
    pacing_strat=["constant velocity", "even effort"],
)

sim_cfg = SimConfig(
    target_dist=4300,
    num_sim=100,
    dt=0.1,
    max_steps=20000,
)

if __name__ == "__main__":
    # get the time now
    ts = time_now()
    jid = job_id(ts)
    # define folder name for results
    folder_name = "03_simulations"

    # determine execution environment
    execution_env = os.getenv("EXECUTION_ENV", "unknown")

    if execution_env == "local":
        # save results to local file system
        load_dotenv()
        bucket_name = os.getenv("BUCKET_NAME", "local_results")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/{jid}")
        logger = logger_mgr.setup_logger()
        logger.info("Running in local environment")
    elif execution_env == "gcp":
        # get bucket name from environment variable
        bucket_name = os.getenv("BUCKET_NAME")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/{jid}")
        logger = logger_mgr.setup_logger()
        if not bucket_name:
            error = "The BUCKET_NAME environment variable is not set!"
            logger.error(error)
            raise ValueError(error)

        logger.info("Running in GCP environment")
    else:
        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=None, folder_name=f"{folder_name}/{jid}")
        logger = logger_mgr.setup_logger()
        warn_message = f"Running in unknown environment: {execution_env}. Results will not be saved."
        logger.warning(warn_message)

    try:
        logger.info("Starting Monte Carlo Simulation...")
        # @fix make this more flexible by allowing the user to specify the date of the run to use for fitting the model parameters
        df_input = create_dataframes(params, sim_cfg.num_sim)
        parquet_data=None
        json_data=None

        logger.info("Retrieved input data and created dataframes for simulation.")

        sim = MonteCarloSimulation(logger=logger, cfg=sim_cfg, df_input=df_input, parquet_data=parquet_data, json_data=json_data)

        # perform the simulation
        sim.run()

        logger.info("Monte Carlo Simulation completed successfully.")

        if execution_env == "local":
            # save results to local file system
            logger.info(f"Saving results to local bucket folder: {bucket_name}")
            sim.save_to_local_results(bucket_name, folder_name, jid, ts)
            logger.info("Results saved")
            # print results
            logger.info(f"Average finish time (s): {np.mean(sim.finish_time)}")
        elif execution_env == "gcp":
            # save results to cloud storage
            logger.info(f"Saving results to GCP bucket: {bucket_name}")
            sim.save_to_cloud_results(bucket_name, folder_name, jid, ts)
            logger.info("Results saved")
            # print results
            logger.info(f"Average finish time (s): {np.mean(sim.finish_time)}")
            # save logs to cloud storage
            logger.info("Uploading logs to GCP...")
            log_blob_path = logger_mgr.upload_log_to_gcs(bucket_name)
            logger.info(f"Logs uploaded to GCP at: {log_blob_path}")
    except Exception as e:
        error = f"An error occurred during the simulation: {e}"
        logger.exception(error)
        logger.exception("Simulation run failed")
    finally:
        logger.info("Closing logger")
        logger_mgr.close_logger(logger)


