"""Main file for retrieving Strava and Visual Crossing data for marathon simulations."""
import logging
from pathlib import Path

from dateutil import parser

from src.process_runs.api.strava import StravaDataRetriever
from src.process_runs.api.visual_crossing import VisualCrossingDataRetriever
from src.process_runs.process_data import DataProcessor


def retrieve_run(logger: logging.Logger, num_runs: int, bucket_name: str, runs_folder: str = "01_runs") -> None:
    """Use to retrieve data from Strava and Visual Crossing."""
    # ensure the output folder exists
    output_folder_path = Path(f"{bucket_name}/{runs_folder}")
    if not output_folder_path.exists():
        logger.info(f"Creating output folder at: {output_folder_path}")
        output_folder_path.mkdir(exist_ok=True, parents=True)

    # initialize data retrievers
    strava_retriever = StravaDataRetriever(logger)
    visual_crossing_retriever = VisualCrossingDataRetriever(logger)

    # now we want to retrieve the data from the past # runs specified by the user
    # fetch activities
    activities = strava_retriever.fetch_activities()

    # filter runs
    runs = strava_retriever.filter_runs(activities, limit=num_runs)

    # setup output folder
    logger.info(f"Saving data to folder: {bucket_name}/{runs_folder}")

    # process each run
    for i, run in enumerate(runs, 1):
        logger.info(f"\nProcessing run {i}/{len(runs)}")

        # get the start date and format it
        start_date = run.get("start_date_local", "")
        date_str = parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")
        # create a folder under output_folder for each run
        run_folder = f"{date_str}"
        run_folder_path = output_folder_path / run_folder
        run_folder_path.mkdir(exist_ok=True)

        logger.info(f"Created folder for run: {run_folder_path}")

        # check this run data to make sure all its componets are present
        # parquet file data
        parquet_data = strava_retriever.parse_to_parquet(run)

        # json file data
        json_data = strava_retriever.parse_to_json(run)

        # now we want to retrieve the weather data for this run
        json_data["weather"] = visual_crossing_retriever.get_weather_openweather(json_data)

        # CLEAN UP DATA AND PERFORM FEATURE ENGINEERING HERE
        data_processor = DataProcessor(logger, parquet_data, json_data)
        # clean and perform feature engineering
        data_processor.process()

        # Save to JSON
        json_filename = "overall.json"
        data_processor.save_to_json(run_folder_path, json_filename)

        # Save to parquet
        parquet_filename = "streams.parquet"
        data_processor.save_to_parquet(run_folder_path, parquet_filename)

    logger.info("\nDone saving all running data!")
