"""
This file contains class for preprocessing data for marathon simulations.
"""
import pandas as pd
import math
import numpy as np
import os
import json

class DataProcessor:
    """
    Class for processing data from Strava and Visual Crossing to prepare it for 
    """
    
    def __init__(self, csv_data: dict, json_data: dict):
        # we first convert dictionary to pandas DataFrame
        self.csv_data = pd.DataFrame(csv_data)
        self.json_data = json_data
        self.rename_columns()

    def rename_columns(self) -> None:
        """
        Rename columns in the CSV data to match the expected format.
        """
        self.csv_data.rename(columns={
            "time": "time_s",
            "heartrate": "heartrate_bpm",
            "cadence": "cadence_rpm",
            "distance": "distance_m",
            "altitude": "altitude_m",
            "velocity": "velocity_mps",
            "grade": "grade_percent",
            "moving": "moving",
            "latitude": "latitude",
            "longitude": "longitude"
        }, inplace=True)

    def interpolate_missing_data(self) -> None:
        """
        This function fills in missing values using linear interpolation.
        """
        # resample the data to interpolate missing rows
        # Create a column to mark original data points
        self.csv_data["is_original"] = True

        # resample the data to interpolate missing rows
        # first time_s row should be start_date_local from overall.json
        start_date_local = self.json_data["start_date_local"]
        # Convert to pandas.Timestamp and remove timezone info if present
        start_date_local_naive = pd.to_datetime(start_date_local).tz_localize(None)
        self.csv_data["time_s"] = pd.to_datetime(self.csv_data["time_s"], unit='s', origin=start_date_local_naive)
        self.csv_data.set_index("time_s", inplace=True)

        # Create a complete time range for every second
        start_time = self.csv_data.index.min()
        end_time = self.csv_data.index.max()
        complete_time_range = pd.date_range(start=start_time, end=end_time, freq='1s')

        # Reindex to include all seconds, marking new rows as interpolated
        self.csv_data = self.csv_data.reindex(complete_time_range)
        self.csv_data["is_original"] = self.csv_data["is_original"]

        # Interpolate the missing values
        numeric_columns = self.csv_data.select_dtypes(include=[np.number]).columns
        self.csv_data[numeric_columns] = self.csv_data[numeric_columns].interpolate(method="linear")

        # convert the index to seconds
        self.csv_data.reset_index(inplace=True)
        self.csv_data.rename(columns={"index": "time_s"}, inplace=True)


    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculates custom bearing where:
        0° = South, 90° = West, 180° = North, 270° = East
        """

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        delta_lon = lon2 - lon1

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

        standard_bearing = math.atan2(x, y)
        standard_bearing_deg = (math.degrees(standard_bearing) + 360) % 360  # 0° = North

        # # Rotate so 0° = South, 90° = West
        # custom_bearing = (standard_bearing_deg + 180) % 360

        return standard_bearing_deg  # Return the standard bearing in degrees


    def feature_engineering(self, resting_heart_rate: float = 60, shift: int = 3) -> None:
        """
        This function performs feature engineering on the data.
        """
        # --------------------------- WIND DIRECTION AND SPEED ---------------------------
        # determine direction in which person is currently moving
        # create a new column for the bearing
        # first row will have no bearing, so we will calculate the bearing for the first two rows
        self.csv_data["delta_latitude"] = self.csv_data["latitude"].diff()  # Change in latitude
        self.csv_data["delta_longitude"] = self.csv_data["longitude"].diff()  # Change in longitude
        for i in range(1, len(self.csv_data)):
            lat1, lon1 = self.csv_data.loc[i - 1, ["latitude", "longitude"]]
            lat2, lon2 = self.csv_data.loc[i, ["latitude", "longitude"]]
            self.csv_data.at[i, "athletedir_degree"] = self._calculate_bearing(lat1, lon1, lat2, lon2)
        self.csv_data["athletedir_degree"] = self.csv_data["athletedir_degree"]

        # add wind direction and speed to the data
        self.csv_data["winddir_degree"] = self.json_data["weather"]["winddir"]  # Wind direction in degrees
        self.csv_data["windspeed_mps"] = self.json_data["weather"]["windspeed"]*1000/3600  # Convert from km/h to m/s

        # calculate the relative wind direction
        self.csv_data["relative_winddir_degree"] = (self.csv_data["winddir_degree"] - self.csv_data["athletedir_degree"]) % 360
        self.csv_data["headwind_mps"] = self.csv_data["windspeed_mps"] * np.cos(np.radians(self.csv_data["relative_winddir_degree"]))
        self.csv_data["crosswind_mps"] = self.csv_data["windspeed_mps"] * np.sin(np.radians(self.csv_data["relative_winddir_degree"]))
        self.csv_data["headwind_mps"] = self.csv_data["headwind_mps"]
        self.csv_data["crosswind_mps"] = self.csv_data["crosswind_mps"]

        # --------------------------- STRIDE LENGTH ---------------------------
        self.csv_data["stride_length_m"] = self.csv_data["velocity_mps"] / (self.csv_data["cadence_rpm"] / 60)  # meters per stride
        self.csv_data["stride_length_m"] = self.csv_data["stride_length_m"]
        # set stride length to 0 if it is infinity
        self.csv_data.loc[self.csv_data["stride_length_m"] == np.inf, "stride_length_m"] = 0  # Set infinity to 0

        # --------------------------- DIFF ALTITUDE ---------------------------
        self.csv_data["diff_altitude_m"] = self.csv_data["altitude_m"].diff()

        # --------------------------- PACE EFFICIENCY ---------------------------
        # data["pace_efficiency"] = data["velocity_mps"] / (data["heartrate_bpm"] / 60)  # m/s per bpm
        # resting heart rate
        self.csv_data["pace_efficiency"] = (self.csv_data["velocity_mps"] / self.json_data["max_speed"]) * (1 - (self.csv_data["heartrate_bpm"] - resting_heart_rate) / self.json_data["max_heartrate"])
        self.csv_data["diff_pace_efficiency"] = self.csv_data["pace_efficiency"].diff()  # Change in pace efficiency

        # --------------------------- DIFF HEART RATE ---------------------------
        self.csv_data["diff_heartrate_bpm"] = self.csv_data["heartrate_bpm"].diff()  # Change in heart rate
        self.csv_data["diff_heartrate_shift_bpm"] = self.csv_data["diff_heartrate_bpm"].shift(shift)  # Shift the diff_heartrate_bpm column by 1 row

        # --------------------------- ACCELERATION ---------------------------
        self.csv_data["acceleration_mps2"] = self.csv_data["velocity_mps"].diff()  # m/s^2
        self.csv_data["acceleration_shift_mps2"] = self.csv_data["acceleration_mps2"].shift(shift)  # Shift the acceleration column by 1 row

        # # remove rows with NaN values
        # self.csv_data.dropna(inplace=True)

    def save_to_csv(self, folder_path: str, filename: str) -> None:
        """
        Save processed data to a CSV file.
        """
        self.csv_data.to_csv(os.path.join(folder_path, filename), index=False)
        print(f"✅ Saved streams to {filename}")

    def save_to_json(self, folder_path: str, filename: str) -> None:
        """
        Save processed data to a JSON file.
        """
        with open(os.path.join(folder_path, filename), 'w') as f:
            json.dump(self.json_data, f, indent=4)
        print(f"✅ Saved overall data to {filename}")


