"""Use this script to fit the model parameters to the data."""
import io
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

from src.model_training.model_fitter import model_fitting
from src.utilis.helper import job_id, time_now
from src.utilis.logger import StrideSimLogger


def _latest_local_run_dates(bucket_name: str, limit: int) -> list[str]:
    """Return the most recent retrieved run folder names from local storage."""
    runs_root = Path(bucket_name) / "01_runs"
    if not runs_root.exists():
        return []

    run_dirs = [path.name for path in runs_root.iterdir() if path.is_dir() and path.name != "logs"]
    return sorted(run_dirs, reverse=True)[:limit]


def _latest_gcp_run_dates(bucket_name: str, limit: int) -> list[str]:
    """Return the most recent retrieved run folder names from GCS."""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix="01_runs/")

    run_dates = {
        blob.name.split("/")[1]
        for blob in blobs
        if len(blob.name.split("/")) >= 3 and blob.name.split("/")[1] != "logs"
    }
    return sorted(run_dates, reverse=True)[:limit]


def get_latest_run_dates(execution_env: str, bucket_name: str, limit: int) -> list[str]:
    """Resolve the latest retrieved run dates for the current environment."""
    if execution_env == "local":
        return _latest_local_run_dates(bucket_name, limit)
    if execution_env == "gcp":
        return _latest_gcp_run_dates(bucket_name, limit)
    return []


def initialize_runtime(folder_name: str, jid: str, execution_env: str) -> tuple[StrideSimLogger, logging.Logger, str | None, int]:
    """Create the logger manager and logger for the active environment."""
    if execution_env == "local":
        load_dotenv()
        bucket_name = os.getenv("BUCKET_NAME", "local_results")
        training_limit = int(os.getenv("NUM_RUNS", "1"))
    elif execution_env == "gcp":
        bucket_name = os.getenv("BUCKET_NAME")
        training_limit = int(os.getenv("NUM_RUNS", "1"))
    else:
        bucket_name = None
        training_limit = 1

    logger_mgr = StrideSimLogger(
        execution_env=execution_env,
        bucket_name=bucket_name,
        folder_name=f"{folder_name}/logs/{jid}" if execution_env != "unknown" else f"{folder_name}/{jid}",
    )
    logger = logger_mgr.setup_logger()

    if execution_env == "local":
        logger.info("Running in local environment")
    elif execution_env == "gcp":
        if not bucket_name:
            error = "The BUCKET_NAME environment variable is not set!"
            logger.error(error)
            raise ValueError(error)
        logger.info("Running in GCP environment")
    else:
        warn_message = f"Running in unknown environment: {execution_env}. Results will not be saved."
        logger.warning(warn_message)

    return logger_mgr, logger, bucket_name, training_limit


def fit_latest_runs(logger: logging.Logger, logger_mgr: StrideSimLogger, execution_env: str, bucket_name: str, training_limit: int) -> None:
    """Train only on the newest retrieved run folders."""
    run_dates = get_latest_run_dates(execution_env, bucket_name or "", training_limit)
    if not run_dates:
        error = "No retrieved runs were found to train on."
        logger.error(error)
        raise FileNotFoundError(error)

    logger.info("Training on the latest retrieved run(s): %s", ", ".join(run_dates))
    start_time = time.perf_counter()
    for date in run_dates:
        logger.info("Starting Model Fitting for run %s...", date)
        model_fitting(logger, logger_mgr, date)

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Model Fitting completed in {elapsed_time:.2f} seconds.")
    logger.info("Model Fitting completed successfully.")

def combine_training_results(logger: logging.Logger, logger_mgr: StrideSimLogger, execution_env: str) -> None:
    """Combine individual training results into one summary parquet file."""
    bucket_name = logger_mgr.bucket_name
    train_folder = logger_mgr.folder_name.split("/")[0]
    if execution_env == "local" and bucket_name:
        # combine results from local file system
        local_results_path = Path(bucket_name) / train_folder
        if not local_results_path.exists():
            error = f"No training results found at {local_results_path} to combine."
            logger.error(error)
            raise FileNotFoundError(error)

        # get all individual training result files
        parameter_files = list(local_results_path.glob("*.json"))
        if not parameter_files:
            error = f"No individual training result files found in {local_results_path} to combine."
            logger.error(error)
            raise FileNotFoundError(error)
        logger.info(f"Found the following individual training result files to combine: {[f.name for f in parameter_files]}")

        # combine the results into a single parquet file
        combined_results_path = local_results_path / "runner_parameters.parquet"
        data_list = [json.loads(f.read_text()) for f in parameter_files]
        pd.DataFrame(data_list).to_parquet(combined_results_path, index=False, engine="pyarrow")
        logger.info("All model coefficients saved locally.")

    elif execution_env == "gcp" and bucket_name:
        # combine results from GCP bucket
        client = storage.Client()
        prefix = f"{train_folder}/"
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        parameter_blobs = [blob for blob in blobs if blob.name.endswith(".json")]
        if not parameter_blobs:
            error = f"No individual training result files found in GCP bucket {bucket_name} with prefix {prefix} to combine."
            logger.error(error)
            raise FileNotFoundError(error)

        buffer = io.BytesIO()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{train_folder}/runner_parameters.parquet")
        data_list = []
        for blob in parameter_blobs:
            content = blob.download_as_text()
            data_list.append(json.loads(content))
        pd.DataFrame(data_list).to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")
        logger.info("All model coefficients saved to GCP.")

    else:
        logger.warning("Cannot combine results in unknown environment. No results will be saved.")

def finalize_runtime(logger_mgr: StrideSimLogger, logger: logging.Logger, execution_env: str, bucket_name: str | None) -> None:
    """Upload logs when needed and close the logger."""
    if execution_env == "gcp" and bucket_name:
        logger.info("Moving logs from local /tmp folder to GCP bucket for easier access...")
        log_blob_path = logger_mgr.upload_log_to_gcs(bucket_name)
        logger.info(f"Logs uploaded to GCP at: {log_blob_path}")

    logger.info("Closing logger...")
    logger_mgr.close_logger(logger)


if __name__ == "__main__":
    ts = time_now()
    jid = job_id(ts)
    folder_name = "02_trainings"

    execution_env = os.getenv("EXECUTION_ENV", "local")

    logger_mgr, logger, bucket_name, training_limit = initialize_runtime(folder_name, jid, execution_env)

    logger.info(f"Training a total of {training_limit} run(s) based on the latest retrieved data.")
    try:
        fit_latest_runs(logger, logger_mgr, execution_env, bucket_name or "", training_limit)
        # combine all individual training results into one summary file for easier analysis
        logger.info("Combining individual training results into one summary file...")
        # combine the results in the respective environment
        combine_training_results(logger, logger_mgr, execution_env)
        logger.info("Results combined")
    finally:
        finalize_runtime(logger_mgr, logger, execution_env, bucket_name)





