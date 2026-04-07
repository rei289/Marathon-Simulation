"""Main file for retrieving runs."""
import os

from dotenv import load_dotenv

from src.process_runs.run_retriever import retrieve_run

if __name__ == "__main__":

    # determine execution environment
    execution_env = os.getenv("EXECUTION_ENV", "unknown")

    if execution_env == "local":
        print("Running in local environment")
        # load configuration from .env file
        load_dotenv()
        num_runs = int(os.getenv("NUM_RUNS", "1"))
        # save results to local file system
        bucket_name = os.getenv("BUCKET_NAME", "local_results")
        # Run the main function
        retrieve_run(num_runs, bucket_name, runs_folder="01_runs")
    elif execution_env == "gcp":
        print("Running in GCP environment")
    else:
        print("Unknown execution environment. Please set EXECUTION_ENV to 'local' or 'gcp'.")
