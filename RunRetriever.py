"""
This is the main file for retrieving Strava and Visual Crossing data for marathon simulations.
"""
from API.Strava import StravaDataRetriever
from API.VisualCrossing import VisualCrossingDataRetriever
from utilis.helper import extract_global_json
import json
import csv
import os
from dateutil import parser
import pprint


def save_to_json(data: dict, folder_path: str, filename: str) -> None:
    """
    Save data to a JSON file.
    """
    with open(os.path.join(folder_path, filename), 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ Saved streams to {filename}")

def save_to_csv(data: dict, folder_path: str, filename: str) -> None:
    """
    Save data to a CSV file.
    """
    with open(os.path.join(folder_path, filename), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "time_s", 
            "heartrate_bpm",
            "cadence_rpm", 
            "distance_m", 
            "altitude_m",
            "velocity_mps",
            "grade_percent",
            "moving",
            "latitude",
            "longitude"
        ])
        
        for row in zip(
            data["time"],
            data['heartrate'],
            data['cadence'],
            data['distance'],
            data['altitude'],
            data['velocity'],
            data['grade'],
            data['moving'],
            data["latitude"],
            data["longitude"]
        ):
            writer.writerow(row)

    print(f"✅ Saved streams to {filename}")

def main(num_runs: int = 10, output_folder: str = "data"):
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

        # perform feature engineering on the csv dataset
        

        # Save to JSON
        json_filename = f"{date_str}_overall.json"
        save_to_json(json_data, os.path.join(output_folder, run_folder), json_filename)

        # Save to CSV
        csv_filename = f"{date_str}_streams.csv"
        save_to_csv(csv_data, os.path.join(output_folder, run_folder), csv_filename)

    print("\n✅ Done saving all running data!")


if __name__ == "__main__":
    # Load configuration from globals.json
    num_runs = extract_global_json("num_runs")
    output_folder = extract_global_json("output_folder")
    # Run the main function
    main(num_runs, output_folder)