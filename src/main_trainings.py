"""Use this script to fit the model parameters to the data."""
import os
import time

from dotenv import load_dotenv

from src.model_training.model_fitter import model_fitting
from src.utilis.helper import job_id, time_now
from src.utilis.logger import StrideSimLogger

if __name__ == "__main__":
    # get the time now
    ts = time_now()
    jid = job_id(ts)
    # define folder name for results
    folder_name = "02_trainings"


    # determine which run to use for fitting the model parameters
    date = "2026-04-15_19-56"

    # determine execution environment
    execution_env = os.getenv("EXECUTION_ENV", "local")

    if execution_env == "local":
        # save results to local file system
        load_dotenv()
        bucket_name = os.getenv("BUCKET_NAME", "local_results")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/logs/{jid}")
        logger = logger_mgr.setup_logger()
        logger.info("Running in local environment")

    elif execution_env == "gcp":
        # get bucket name from environment variable
        bucket_name = os.getenv("BUCKET_NAME")

        logger_mgr = StrideSimLogger(execution_env=execution_env, bucket_name=bucket_name, folder_name=f"{folder_name}/logs/{jid}")
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
        start_time = time.perf_counter()
        logger.info("Starting Model Fitting...")

        model_fitting(logger, logger_mgr, date)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Model Fitting completed in {elapsed_time:.2f} seconds.")
        logger.info("Model Fitting completed successfully.")
    except Exception as e:
        error = f"Model Fitting failed with error: {e}"
        logger.exception(error)
    finally:
        if execution_env == "gcp":
            logger.info("Moving logs from local /tmp folder to GCP bucket for easier access...")
            log_blob_path = logger_mgr.upload_log_to_gcs(bucket_name)
            logger.info(f"Logs uploaded to GCP at: {log_blob_path}")
        logger.info("Closing logger...")
        logger_mgr.close_logger(logger)





