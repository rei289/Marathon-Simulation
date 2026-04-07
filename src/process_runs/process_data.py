"""File contains class for preprocessing data for marathon simulations."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class DataProcessor:
    """Class for processing data from Strava and Visual Crossing to prepare it for use in the marathon simulation."""

    def __init__(self, parquet_data: dict, json_data: dict) -> None:
        """Initialize the DataProcessor with raw CSV and JSON data."""
        # we first convert dictionary to pandas DataFrame
        self.parquet_data = pd.DataFrame(parquet_data)
        self.json_data = json_data
        self.rename_columns()

    def rename_columns(self) -> None:
        """Rename columns in the CSV data to match the expected format."""
        self.parquet_data.rename(columns={
            "time": "time_datetime",
            "heartrate": "heartrate_bpm",
            "cadence": "cadence_rpm",
            "distance": "distance_m",
            "altitude": "altitude_m",
            "velocity": "velocity_mps",
            "grade": "grade_percent",
            "moving": "moving",
            "latitude": "latitude_degree",
            "longitude": "longitude_degree",
        }, inplace=True)

    def interpolate_missing_data(self) -> None:
        """Use function to fill in missing values using linear interpolation."""
        # resample the data to interpolate missing rows
        # create a column to mark original data points
        self.parquet_data["is_original"] = True

        # resample the data to interpolate missing rows
        # first time_datetime row should be start_date_local from overall.json
        start_date_local = self.json_data["start_date_local"]
        # Convert to pandas.Timestamp and remove timezone info if present
        start_date_local_naive = pd.to_datetime(start_date_local).tz_localize(None)
        self.parquet_data["time_datetime"] = pd.to_datetime(self.parquet_data["time_datetime"], unit="s", origin=start_date_local_naive)
        self.parquet_data.set_index("time_datetime", inplace=True)

        # create a complete time range for every second
        start_time = self.parquet_data.index.min()
        end_time = self.parquet_data.index.max()
        complete_time_range = pd.date_range(start=start_time, end=end_time, freq="1s")

        # reindex to include all seconds, marking new rows as interpolated
        self.parquet_data = self.parquet_data.reindex(complete_time_range)
        self.parquet_data["is_original"] = self.parquet_data["is_original"]

        # interpolate the missing values
        numeric_columns = self.parquet_data.select_dtypes(include=[np.number]).columns
        self.parquet_data[numeric_columns] = self.parquet_data[numeric_columns].interpolate(method="linear")

        # convert the index to seconds
        self.parquet_data.reset_index(inplace=True)
        self.parquet_data.rename(columns={"index": "time_datetime"}, inplace=True)

    def smooth_data(self, features: list, window_size: int = 10) -> None:
        """Smooth the data using a rolling average."""
        for feature in features:
            self.parquet_data[f"smooth_{feature}"] = self.parquet_data[feature].rolling(window=window_size, min_periods=1).mean()

    def _minute_to_second(self) -> None:
        """Convert units in the data."""
        # convert minutes to seconds
        self.parquet_data["heartrate_bps"] = self.parquet_data["heartrate_bpm"] / 60  # convert bpm to bps
        self.parquet_data["cadence_rps"] = self.parquet_data["cadence_rpm"] / 60  # convert rpm to rps

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Use to calculate custom bearing where: 0° = South, 90° = West, 180° = North, 270° = East."""
        # convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        delta_lon = lon2 - lon1

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

        standard_bearing = math.atan2(x, y)
        return (math.degrees(standard_bearing) + 360) % 360  # 0° = North

    def feature_engineering(self) -> None:
        """Use to perform feature engineering on the data."""
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
        self.parquet_data["delta_latitude"] = self.parquet_data[lat].diff()  # Change in latitude
        self.parquet_data["delta_longitude"] = self.parquet_data[lon].diff()  # Change in longitude
        for i in range(1, len(self.parquet_data)):
            lat1, lon1 = self.parquet_data.loc[i - 1, [lat, lon]]
            lat2, lon2 = self.parquet_data.loc[i, [lat, lon]]
            self.parquet_data.loc[i, "athletedir_degree"] = self._calculate_bearing(lat1, lon1, lat2, lon2)
        self.parquet_data["athletedir_degree"] = self.parquet_data["athletedir_degree"]

        # add wind direction and speed to the data
        self.parquet_data["winddir_degree"] = self.json_data["weather"]["winddir"]  # Wind direction in degrees
        self.parquet_data["windspeed_mps"] = self.json_data["weather"]["windspeed"]*1000/3600  # Convert from km/h to m/s

        # calculate the relative wind direction
        self.parquet_data["relative_winddir_degree"] = (self.parquet_data["winddir_degree"] - self.parquet_data["athletedir_degree"]) % 360
        # positive for headwind, negative for tailwind
        self.parquet_data["headwind_mps"] = self.parquet_data["windspeed_mps"] * np.cos(np.radians(self.parquet_data["relative_winddir_degree"]))
        self.parquet_data["crosswind_mps"] = self.parquet_data["windspeed_mps"] * np.sin(np.radians(self.parquet_data["relative_winddir_degree"]))

        # delete the temporary columns
        self.parquet_data.drop(columns=["delta_latitude", "delta_longitude", "windspeed_mps", "athletedir_degree", "winddir_degree"], inplace=True)

        # --------------------------- STRIDE LENGTH ---------------------------
        self.parquet_data["stride_length_m"] = self.parquet_data[vel] / self.parquet_data[cad]  # meters per stride for 1 stride
        # set stride length to 0 if it is infinity
        self.parquet_data.loc[self.parquet_data["stride_length_m"] == np.inf, "stride_length_m"] = 0  # Set infinity to 0

        # --------------------------- DIFF ALTITUDE ---------------------------
        self.parquet_data["diff_altitude_mps"] = self.parquet_data[alt].diff()

        # --------------------------- DIFF HEART RATE ---------------------------
        self.parquet_data["diff_heartrate_bps2"] = self.parquet_data[hr].diff()  # Change in heart rate

        # --------------------------- ACCELERATION ---------------------------
        self.parquet_data["diff_velocity_mps2"] = self.parquet_data[vel].diff()  # m/s^2


    def process(self) -> None:
        """Use to process the data by cleaning, interpolating, unit conversion, smoothing, and feature engineering."""
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
        self.feature_engineering()

        # smooth the data again after feature engineering
        self.smooth_data(window_size=5, features=[
            "headwind_mps",
            "crosswind_mps",
        ])

    def save_to_parquet(self, folder: str, filename: str) -> None:
        """Use to save processed data to a Parquet file."""
        folder_path = Path(folder)
        file_path = folder_path / filename
        self.parquet_data.to_parquet(file_path, index=False)
        print(f"✅ Saved streams to {filename}")

    def save_to_json(self, folder: str, filename: str) -> None:
        """Use to save processed data to a JSON file."""
        folder_path = Path(folder)
        file_path = folder_path / filename
        file_path.write_text(json.dumps(self.json_data, indent=4))
        print(f"✅ Saved overall data to {filename}")
