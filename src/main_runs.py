"""Main file for retrieving runs."""
import os

from dotenv import load_dotenv

from src.process_runs.run_retriever import retrieve_run
from src.utilis.helper import job_id, time_now
from src.utilis.logger import StrideSimLogger

if __name__ == "__main__":
    # get the time now
    ts = time_now()
    jid = job_id(ts)
    # define folder name for results
    folder_name = "01_runs"

    # determine execution environment
    execution_env = os.getenv("EXECUTION_ENV", "unknown")

    if execution_env == "local":
        # load configuration from .env file
        load_dotenv()
        num_runs = int(os.getenv("NUM_RUNS", "1"))
        bucket_name = os.getenv("BUCKET_NAME", "local_results")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/logs/{jid}")
        logger = logger_mgr.setup_logger()
        logger.info("Running in local environment")
        # save results to local file system
        # Run the main function
        retrieve_run(logger, num_runs, bucket_name, runs_folder=folder_name)
        logger.info("Data retrieval completed successfully.")
        logger_mgr.close_logger(logger)
    elif execution_env == "gcp":
        # get bucket name from environment variable
        num_runs = int(os.getenv("NUM_RUNS", "1"))
        bucket_name = os.getenv("BUCKET_NAME", "local_results")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/logs/{jid}")
        logger = logger_mgr.setup_logger()
        logger.info("Running in GCP environment")
    else:
        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=None, folder_name=f"{folder_name}/logs/{jid}")
        logger = logger_mgr.setup_logger()
        warn_message = f"Running in unknown environment: {execution_env}. Results will not be saved."
        logger.warning(warn_message)
