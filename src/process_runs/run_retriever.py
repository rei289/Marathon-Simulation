"""Main file for retrieving Strava and Visual Crossing data for marathon simulations."""
import logging

from dateutil import parser

from src.process_runs.api.strava import StravaDataRetriever
from src.process_runs.api.visual_crossing import VisualCrossingDataRetriever
from src.process_runs.process_data import DataProcessor
from src.utilis.logger import StrideSimLogger


def retrieve_run(logger: logging.Logger, logger_mgr: StrideSimLogger, num_runs: int) -> None:
    """Use to retrieve data from Strava and Visual Crossing."""
    # define runs folder name from logger manager
    runs_folder = logger_mgr.folder_name.split("/")[0]

    # initialize data retrievers
    strava_retriever = StravaDataRetriever(logger, logger_mgr.execution_env)
    visual_crossing_retriever = VisualCrossingDataRetriever(logger, logger_mgr.execution_env)

    # now we want to retrieve the data from the past # runs specified by the user
    # fetch activities
    activities = strava_retriever.fetch_activities()

    # filter runs
    runs = strava_retriever.filter_runs(activities, limit=num_runs)

    # process each run
    for i, run in enumerate(runs, 1):
        logger.info(f"------------- Processing run {i}/{len(runs)} -------------")

        # get the start date and format it
        start_date = run.get("start_date_local", "")
        date_str = parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")

        # check this run data to make sure all its componets are present
        # parquet file data
        parquet_data = strava_retriever.parse_to_parquet(run)

        # json file data
        json_data = strava_retriever.parse_to_json(run)

        # now we want to retrieve the weather data for this run
        json_data["weather"] = visual_crossing_retriever.get_weather_openweather(json_data)

        # lastly add the job id to the json data for easier reference later on
        json_data["jid"] = logger_mgr.folder_name.split("/")[2]

        # CLEAN UP DATA AND PERFORM FEATURE ENGINEERING HERE
        data_processor = DataProcessor(logger, parquet_data, json_data)
        # clean and perform feature engineering
        data_processor.process()

        # save the processed data to the output folder
        if logger_mgr.execution_env == "local":
            data_processor.save_to_local_results(logger_mgr.bucket_name, runs_folder, date_str, "streams.parquet", "overall.json")
        elif logger_mgr.execution_env == "gcp":
            data_processor.save_to_cloud_results(logger_mgr.bucket_name, runs_folder, date_str, "streams.parquet", "overall.json")

    logger.info("\nDone saving all running data!")
