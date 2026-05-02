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
    execution_env = os.getenv("EXECUTION_ENV", "local")

    if execution_env == "local":
        # load configuration from .env file
        load_dotenv()
        num_runs = int(os.getenv("NUM_RUNS", "1"))
        bucket_name = os.getenv("BUCKET_NAME", "local_results")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/logs/{jid}")
        logger = logger_mgr.setup_logger()
        logger.info("Running in local environment")
        # save results to local file system
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
        logger_mgr.close_logger(logger)

    try:
        logger.info("Starting run data retrieval...")
        retrieve_run(logger, logger_mgr, num_runs)
        logger.info("Run data retrieval completed successfully.")
    except Exception as e:
        error = f"An error occurred during run data retrieval: {e}"
        logger.exception(error)
    finally:
        # if execution is done in gcp, we want to move the logs from the /tmp folder to the bucket for easier access
        if execution_env == "gcp":
            logger.info("Moving logs from local /tmp folder to GCP bucket for easier access...")
            log_blob_path = logger_mgr.upload_log_to_gcs(bucket_name)
            logger.info(f"Logs uploaded to GCP at: {log_blob_path}")
        logger_mgr.close_logger(logger)
