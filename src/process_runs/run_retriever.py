"""Main file for retrieving Strava and Visual Crossing data for marathon simulations."""
from pathlib import Path

from dateutil import parser

from src.process_runs.api.strava import StravaDataRetriever
from src.process_runs.api.visual_crossing import VisualCrossingDataRetriever
from src.process_runs.process_data import DataProcessor


def retrieve_run(num_runs: int = 10, output_folder: str = "data") -> None:
    """Use to retrieve data from Strava and Visual Crossing."""
    # ensure the output folder exists
    output_folder_path = Path(output_folder)
    if not output_folder_path.exists():
        output_folder_path.mkdir(exist_ok=True)

    # initialize data retrievers
    strava_retriever = StravaDataRetriever()
    visual_crossing_retriever = VisualCrossingDataRetriever()

    # now we want to retrieve the data from the past # runs specified by the user
    # fetch activities
    activities = strava_retriever.fetch_activities()

    # filter runs
    runs = strava_retriever.filter_runs(activities, limit=num_runs)

    # setup output folder
    print(f"\n📁 Saving data to folder: {output_folder}\n")

    # process each run
    for i, run in enumerate(runs, 1):
        print(f"\n--- Processing run {i}/{len(runs)} ---")

        # get the start date and format it
        start_date = run.get("start_date_local", "")
        date_str = parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")
        # create a folder under output_folder for each run
        run_folder = f"{date_str}"
        run_folder_path = output_folder_path / run_folder
        run_folder_path.mkdir(exist_ok=True)

        # check this run data to make sure all its componets are present
        # csv file data
        csv_data = strava_retriever.parse_to_csv(run)

        # json file data
        json_data = strava_retriever.parse_to_json(run)

        # now we want to retrieve the weather data for this run
        json_data["weather"] = visual_crossing_retriever.get_weather_openweather(json_data)

        # CLEAN UP DATA AND PERFORM FEATURE ENGINEERING HERE
        data_processor = DataProcessor(csv_data, json_data)
        # clean and perform feature engineering
        data_processor.process()

        # Save to JSON
        json_filename = f"{date_str}_overall.json"
        data_processor.save_to_json(run_folder_path, json_filename)

        # Save to CSV
        csv_filename = f"{date_str}_streams.csv"
        data_processor.save_to_csv(run_folder_path, csv_filename)

    print("\n✅ Done saving all running data!")
