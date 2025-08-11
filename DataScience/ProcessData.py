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
        # self.csv_data = pd.DataFrame(csv_data).fillna(0)  # Fill NaN values with 0
        self.json_data = json_data
        self.rename_columns()

    def rename_columns(self) -> None:
        """
        Rename columns in the CSV data to match the expected format.
        """
        self.csv_data.rename(columns={
            "time": "time_datetime",
            "heartrate": "heartrate_bpm",
            "cadence": "cadence_rpm",
            "distance": "distance_m",
            "altitude": "altitude_m",
            "velocity": "velocity_mps",
            "grade": "grade_percent",
            "moving": "moving",
            "latitude": "latitude_degree",
            "longitude": "longitude_degree"
        }, inplace=True)

    def interpolate_missing_data(self) -> None:
        """
        This function fills in missing values using linear interpolation.
        """
        # resample the data to interpolate missing rows
        # Create a column to mark original data points
        self.csv_data["is_original"] = True

        # resample the data to interpolate missing rows
        # first time_datetime row should be start_date_local from overall.json
        start_date_local = self.json_data["start_date_local"]
        # Convert to pandas.Timestamp and remove timezone info if present
        start_date_local_naive = pd.to_datetime(start_date_local).tz_localize(None)
        self.csv_data["time_datetime"] = pd.to_datetime(self.csv_data["time_datetime"], unit='s', origin=start_date_local_naive)
        self.csv_data.set_index("time_datetime", inplace=True)

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
        self.csv_data.rename(columns={"index": "time_datetime"}, inplace=True)

    def smooth_data(self, features: list, window_size: int = 10) -> None:
        """
        Smooth the data using a rolling average.
        """
        for feature in features:
            self.csv_data[f"smooth_{feature}"] = self.csv_data[feature].rolling(window=window_size, min_periods=1).mean()


        # self.csv_data["heartrate_smooth_bps"] = self.csv_data["heartrate_bpm"].rolling(window=window_size, min_periods=1).mean() / 60  # Convert bpm to bps
        # self.csv_data["velocity_smooth_mps"] = self.csv_data["velocity_mps"].rolling(window=window_size, min_periods=1).mean()
        # self.csv_data["cadence_smooth_rps"] = self.csv_data["cadence_rps"].rolling(window=window_size, min_periods=1).mean()
        # self.csv_data["altitude_smooth_m"] = self.csv_data["altitude_m"].rolling(window=window_size, min_periods=1).mean()
        # self.csv_data["grade_smooth_percent"] = self.csv_data["grade_percent"].rolling(window=window_size, min_periods=1).mean()

    def _minute_to_second(self) -> None:
        """
        Convert units in the data.
        """
        # Convert minutes to seconds
        self.csv_data["heartrate_bps"] = self.csv_data["heartrate_bpm"] / 60  # Convert bpm to bps
        self.csv_data["cadence_rps"] = self.csv_data["cadence_rpm"] / 60  # Convert rpm to rps

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


    def feature_engineering(self, resting_heart_rate: float = 60) -> None:
        """
        This function performs feature engineering on the data.
        """
        # make constants to make it easier to change later
        lat = "latitude_degree"
        lon = "longitude_degree"
        alt = "smooth_altitude_m"
        hr = "smooth_heartrate_bps"
        vel = "smooth_velocity_mps"
        cad = "smooth_cadence_rps"


        # --------------------------- WIND DIRECTION AND SPEED ---------------------------
        # determine direction in which person is currently moving
        # create a new column for the bearing
        # first row will have no bearing, so we will calculate the bearing for the first two rows
        self.csv_data["delta_latitude"] = self.csv_data[lat].diff()  # Change in latitude
        self.csv_data["delta_longitude"] = self.csv_data[lon].diff()  # Change in longitude
        for i in range(1, len(self.csv_data)):
            lat1, lon1 = self.csv_data.loc[i - 1, [lat, lon]]
            lat2, lon2 = self.csv_data.loc[i, [lat, lon]]
            self.csv_data.at[i, "athletedir_degree"] = self._calculate_bearing(lat1, lon1, lat2, lon2)
        self.csv_data["athletedir_degree"] = self.csv_data["athletedir_degree"]

        # add wind direction and speed to the data
        self.csv_data["winddir_degree"] = self.json_data["weather"]["winddir"]  # Wind direction in degrees
        self.csv_data["windspeed_mps"] = self.json_data["weather"]["windspeed"]*1000/3600  # Convert from km/h to m/s

        # calculate the relative wind direction
        self.csv_data["relative_winddir_degree"] = (self.csv_data["winddir_degree"] - self.csv_data["athletedir_degree"]) % 360
        self.csv_data["headwind_mps"] = self.csv_data["windspeed_mps"] * np.cos(np.radians(self.csv_data["relative_winddir_degree"]))
        self.csv_data["crosswind_mps"] = self.csv_data["windspeed_mps"] * np.sin(np.radians(self.csv_data["relative_winddir_degree"]))

        # delete the temporary columns
        self.csv_data.drop(columns=["delta_latitude", "delta_longitude", "windspeed_mps", "athletedir_degree", "winddir_degree"], inplace=True)

        # --------------------------- STRIDE LENGTH ---------------------------
        self.csv_data["stride_length_m"] = self.csv_data[vel] / self.csv_data[cad]  # meters per stride for 1 stride
        # set stride length to 0 if it is infinity
        self.csv_data.loc[self.csv_data["stride_length_m"] == np.inf, "stride_length_m"] = 0  # Set infinity to 0

        # --------------------------- DIFF ALTITUDE ---------------------------
        self.csv_data["diff_altitude_mps"] = self.csv_data[alt].diff()

        # --------------------------- PACE EFFICIENCY ---------------------------
        # data["pace_efficiency"] = data["velocity_mps"] / (data["heartrate_bpm"] / 60)  # m/s per bpm
        # resting heart rate
        self.csv_data["smooth_pace_efficiency"] = (self.csv_data[vel] / self.json_data["max_speed"]) * (1 - (self.csv_data[hr] - (resting_heart_rate/60)) / (self.json_data["max_heartrate"]/60))
        self.csv_data["pace_efficiency"] = (self.csv_data["velocity_mps"] / self.json_data["max_speed"]) * (1 - (self.csv_data["heartrate_bps"] - (resting_heart_rate/60)) / (self.json_data["max_heartrate"]/60))
        self.csv_data["diff_pace_efficiency"] = self.csv_data["smooth_pace_efficiency"].diff()  # Change in pace efficiency

        # --------------------------- DIFF HEART RATE ---------------------------
        self.csv_data["diff_heartrate_bps2"] = self.csv_data[hr].diff()  # Change in heart rate

        # --------------------------- ACCELERATION ---------------------------
        self.csv_data["diff_velocity_mps2"] = self.csv_data[vel].diff()  # m/s^2

        # # remove rows with NaN values
        # self.csv_data.dropna(inplace=True)

    def process(self) -> None:
        """
        Process the data by cleaning, interpolating, unit conversion, smoothing, and feature engineering.
        """
        # interpolate missing data
        self.interpolate_missing_data()
        # convert units
        self._minute_to_second()
        # smooth the data
        self.smooth_data(window_size=5, features=[
            "heartrate_bps", 
            "velocity_mps", 
            "cadence_rps", 
            "altitude_m", 
            "grade_percent",
        ])

        # perform feature engineering
        self.feature_engineering(resting_heart_rate=60)

        # smooth the data again after feature engineering
        self.smooth_data(window_size=5, features=[
            "headwind_mps",
            "crosswind_mps",
        ])

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


