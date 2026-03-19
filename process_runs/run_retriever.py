"""
This is the main file for retrieving Strava and Visual Crossing data for marathon simulations.
"""
from process_runs.api.strava import StravaDataRetriever
from process_runs.api.visual_crossing import VisualCrossingDataRetriever
from process_runs.process_data import DataProcessor
from utilis.helper import extract_global_json
import json
import csv
import os
from dateutil import parser

def retrieve_run(num_runs: int = 10, output_folder: str = "data"):
    """
    Main function to retrieve data from Strava and Visual Crossing.
    """
    # Ensure the output folder exists
    if not output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Initialize data retrievers
    strava_retriever = StravaDataRetriever()
    visual_crossing_retriever = VisualCrossingDataRetriever()

    # now we want to retrieve the data from the past 10 runs
    strava_retriever.authenticate()  # Ensure authentication is done before retrieving data

    # Fetch activities
    activities = strava_retriever.fetch_activities()

    # Filter runs
    runs = strava_retriever.filter_runs(activities, limit=num_runs)

    # Setup output folder
    print(f"\n📁 Saving data to folder: {output_folder}\n")

    # process each run
    for i, run in enumerate(runs, 1):
        print(f"\n--- Processing run {i}/{len(runs)} ---")

        # get the start date and format it
        start_date = run.get("start_date_local", "")
        date_str = parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")
        # create a folder under output_folder for each run
        run_folder = f"{date_str}"
        os.makedirs(output_folder+"/"+run_folder, exist_ok=True)

        # check this run data to make sure all its componets are present


        # csv file data
        csv_data = strava_retriever.parse_to_csv(run)

        # json file data
        json_data = strava_retriever.parse_to_json(run)

        # now we want to retrieve the weather data for this run
        json_data['weather'] = visual_crossing_retriever.get_weather_openweather(json_data)

        # CLEAN UP DATA AND PERFORM FEATURE ENGINEERING HERE
        data_processor = DataProcessor(csv_data, json_data)
        # clean and perform feature engineering
        data_processor.process()
        # # clean the data
        # data_processor.interpolate_missing_data()
        # data_processor.unit_conversion()  # Convert per minute to per second
        # data_processor.smooth_data(window_size=10)  # Smooth the data

        # # perform feature engineering
        # data_processor.feature_engineering(resting_heart_rate=60)

        # Save to JSON
        json_filename = f"{date_str}_overall.json"
        data_processor.save_to_json(os.path.join(output_folder, run_folder), json_filename)
        # save_to_json(json_data, os.path.join(output_folder, run_folder), json_filename)

        # Save to CSV
        csv_filename = f"{date_str}_streams.csv"
        data_processor.save_to_csv(os.path.join(output_folder, run_folder), csv_filename)
        # save_to_csv(csv_data, os.path.join(output_folder, run_folder), csv_filename)

    print("\n✅ Done saving all running data!")