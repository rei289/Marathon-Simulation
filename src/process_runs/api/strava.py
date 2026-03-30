"""Structured Strava Data Retriever.

This module provides a class-based approach to retrieve and save running data from Strava API.
It organizes the functionality into logical components for better maintainability and reusability.
"""

import os
from typing import Any

from dateutil import parser
from dotenv import load_dotenv
from requests_oauthlib import OAuth2Session


class StravaDataRetriever:
    """A class to handle Strava API authentication and data retrieval."""

    def __init__(self, output_folder: str = "data") -> None:
        """Initialize the Strava data retriever.

        Args:
            output_folder (str): The folder where data will be saved

        """
        self.output_folder = output_folder
        self.session = None
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load Strava API credentials from environment variables."""
        load_dotenv()
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_url = "https://localhost"

        if not self.client_id or not self.client_secret:
            message = "CLIENT_ID and CLIENT_SECRET must be set in environment variables. Please create a .env file with these values."
            raise ValueError(message)

    def authenticate(self) -> None:
        """Use to handle OAuth2 authentication with Strava."""
        # create OAuth2 session
        self.session = OAuth2Session(
            client_id=self.client_id,
            redirect_uri=self.redirect_url,
            scope=["activity:read_all"],
        )

        # get authorization URL
        auth_url, _ = self.session.authorization_url("https://www.strava.com/oauth/authorize")
        print(f"👉 Click this URL to authorize access:\n{auth_url}")

        # get redirect response from user
        redirect_response = input("\n🔁 Paste the full redirect URL after approving: ")

        # fetch access token
        token_url = "https://www.strava.com/api/v3/oauth/token" # noqa: S105
        _token = self.session.fetch_token(
            token_url=token_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            authorization_response=redirect_response,
            include_client_id=True,
        )

        print("✅ Authentication successful!")

    def fetch_activities(self, per_page: int = 50, page: int = 1) -> list[dict[str, Any]]:
        """Use to fetch activities from Strava API.

        Args:
            per_page (int): Number of activities per page
            page (int): Page number to fetch

        Returns:
            list[dict[str, Any]]: list of activity data

        """
        if not self.session:
            error_message = "Must authenticate before fetching activities. Call the authenticate() method first."
            raise ValueError(error_message)

        print("\n📥 Fetching your latest activities...")
        activities_url = "https://www.strava.com/api/v3/athlete/activities"
        params = {"per_page": per_page, "page": page}
        response = self.session.get(activities_url, params=params)
        response.raise_for_status()
        return response.json()

    def filter_runs(self, activities: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        """Use to filter activities to get only runs.

        Args:
            activities (list[dict[str, Any]]): list of all activities
            limit (int): Maximum number of runs to return

        Returns:
            list[dict[str, Any]]: list of run activities

        """
        runs = [act for act in activities if act.get("type") == "Run"][:limit]
        print(f"📊 Found {len(runs)} recent runs")
        return runs

    def _create_date_string(self, start_date: str) -> str:
        """Convert start date to formatted string."""
        return parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")


    def fetch_activity_streams(self, run_id: int) -> dict[str, Any]:
        """Use to fetch detailed stream data for a specific activity.

        Args:
            run_id (int): Strava activity ID

        Returns:
            dict[str, Any]: Stream data from Strava

        """
        if not self.session:
            error_message = "Must authenticate before fetching streams. Call the authenticate() method first."
            raise ValueError(error_message)

        streams_url = f"https://www.strava.com/api/v3/activities/{run_id}/streams"
        params = {
            "keys": "time,heartrate,cadence,distance,altitude,velocity_smooth,grade_smooth,moving,latlng",
            "key_by_type": "true",
        }
        streams_response = self.session.get(streams_url, params=params)
        streams_response.raise_for_status()
        return streams_response.json()

    def _pad_list(self, lst: list[Any], target_length: int) -> list[Any]:
        """Pad list with None values to reach target length."""
        return lst + [None] * (target_length - len(lst))

    def _extract_coordinates(self, latlng_data: list[Any]) -> tuple[list[float|None], list[float|None]]:
        """Use to extract latitude and longitude from coordinate data.

        Args:
            latlng_data (list[Any]): list of [lat, lng] pairs

        Returns:
            tuple[list[Optional[float]], list[Optional[float]]]: Latitude and longitude lists

        """
        lat, lng = [], []
        for point in latlng_data:
            if point and isinstance(point, list) and len(point) == 2:
                lat.append(point[0])
                lng.append(point[1])
            else:
                lat.append(None)
                lng.append(None)
        return lat, lng

    def parse_to_csv(self, run: dict[str, Any]) -> dict[str, Any]:
        """Use to parse run data to CSV format.

        Args:
            run (dict[str, Any]): Run data

        Returns:
            str: CSV formatted string of run data

        """
        run_id = run["id"]
        name = run.get("name", "Unnamed_run")

        # fetch stream data
        streams = self.fetch_activity_streams(run_id)

        # extract stream data with defaults
        time_data = streams.get("time", {}).get("data", [])
        if not time_data:
            print(f"⚠️ No time stream data for run '{name}' ({run_id}), skipping streams.")
            return False

        # extract all stream types
        stream_data = {
            "time": time_data,
            "heartrate": streams.get("heartrate", {}).get("data", []),
            "distance": streams.get("distance", {}).get("data", []),
            "cadence": streams.get("cadence", {}).get("data", []),
            "altitude": streams.get("altitude", {}).get("data", []),
            "velocity": streams.get("velocity_smooth", {}).get("data", []),
            "grade": streams.get("grade_smooth", {}).get("data", []),
            "moving": streams.get("moving", {}).get("data", []),
            "latlng": streams.get("latlng", {}).get("data", []),
        }

        # pad all lists to match time_data length
        max_len = len(time_data)
        for key, value in stream_data.items():
            stream_data[key] = self._pad_list(value, max_len)

        # extract coordinates
        lat, lng = self._extract_coordinates(stream_data["latlng"])

        # put this into the stream_data
        stream_data["latitude"] = lat
        stream_data["longitude"] = lng

        # remove latlng from stream_data
        del stream_data["latlng"]

        return stream_data

    def parse_to_json(self, run: dict[str, Any]) -> dict[str, Any]:
        """Use to parse run data to JSON format.

        Args:
            run (dict[str, Any]): Run data

        Returns:
            dict[str, Any]: JSON formatted run data

        """
        # extract basic run information
        activity_id = run.get("id")
        # fetch detailed activity data (resource state 3)
        detailed_url = f"https://www.strava.com/api/v3/activities/{activity_id}"
        detailed_response = self.session.get(detailed_url)
        detailed_run = detailed_response.json()
        # extract relevant fields for JSON
        return {
            # basic run information
            "id": detailed_run.get("id"),
            "name": detailed_run.get("name"),
            "type": detailed_run.get("type"),

            # time information
            "start_date": detailed_run.get("start_date"),
            "start_date_local": detailed_run.get("start_date_local"),
            "timezone": detailed_run.get("timezone"),
            "utc_offset": detailed_run.get("utc_offset"),

            # run statistics
            "distance": detailed_run.get("distance"),
            "moving_time": detailed_run.get("moving_time"),
            "elapsed_time": detailed_run.get("elapsed_time"),

            # elevations
            "total_elevation_gain": detailed_run.get("total_elevation_gain"),
            "elev_high": detailed_run.get("elev_high"),
            "elev_low": detailed_run.get("elev_low"),
            "start_latlng": detailed_run.get("start_latlng"),
            "end_latlng": detailed_run.get("end_latlng"),

            # pace and speed
            "average_speed": detailed_run.get("average_speed"),
            "max_speed": detailed_run.get("max_speed"),
            "average_cadence": detailed_run.get("average_cadence"),

            # heart rate
            "average_heartrate": detailed_run.get("average_heartrate"),
            "max_heartrate": detailed_run.get("max_heartrate"),
            # other
            "calories": detailed_run.get("calories"),
        }
