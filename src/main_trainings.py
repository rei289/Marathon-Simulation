"""Use this script to fit the model parameters to the data."""
import os

from dotenv import load_dotenv

from src.model_training.model_fitter import model_fitting

if __name__ == "__main__":
    # determine which run to use for fitting the model parameters
    date = "2026-04-06_13-50"

    # save results to local file system
    load_dotenv()
    bucket_name = os.getenv("BUCKET_NAME", "local_results")
    model_fitting(date, bucket_name, train_folder="02_trainings")
