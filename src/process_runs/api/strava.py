"""Structured Strava Data Retriever.

This module provides a class-based approach to retrieve and save running data from Strava API.
It organizes the functionality into logical components for better maintainability and reusability.
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv, set_key


class StravaDataRetriever:
    """A class to handle Strava API authentication and data retrieval."""

    def __init__(self) -> None:
        """Initialize the Strava data retriever. Load credentials and set up API endpoints."""
        self._load_credentials()
        self.token_url = "https://www.strava.com/oauth/token" # noqa: S105
        self.api_base = "https://www.strava.com/api/v3"

    def _load_credentials(self) -> None:
        """Load Strava API credentials from environment variables."""
        load_dotenv()
        self.client_id = os.getenv("STRAVA_CLIENT_ID")
        self.client_secret = os.getenv("STRAVA_CLIENT_SECRET")
        self.refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
        self.access_token = None

        if not self.client_id or not self.client_secret:
            message = "STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in environment variables. Please create a .env file with these values."
            raise ValueError(message)

    def refresh_access_token(self) -> None:
        """Use to refresh the Strava access token using the refresh token."""
        if not self.refresh_token:
            message = "Missing STRAVA_REFRESH_TOKEN in .env. Please create a .env file with this value."
            raise ValueError(message)

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        response = requests.post(
            self.token_url,
            data=payload,
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data["access_token"]

        # strava may rotate refresh_token; persist latest value
        new_refresh = token_data.get("refresh_token")
        if new_refresh and new_refresh != self.refresh_token:
            self.refresh_token = new_refresh
            set_key(".env", "STRAVA_REFRESH_TOKEN", new_refresh)

    def _auth_headers(self) -> dict[str, str]:
        """Use to generate authorization headers for Strava API requests."""
        if not self.access_token:
            self.refresh_access_token()
        return {"Authorization": f"Bearer {self.access_token}"}

    def fetch_activities(self, per_page: int = 50, page: int = 1) -> list[dict[str, Any]]:
        """Use to fetch activities from Strava API."""
        r = requests.get(
            f"{self.api_base}/athlete/activities",
            headers=self._auth_headers(),
            params={"per_page": per_page, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def fetch_activity_streams(self, run_id: int) -> dict[str, Any]:
        """Use to fetch detailed stream data for a specific activity."""
        r = requests.get(
            f"{self.api_base}/activities/{run_id}/streams",
            headers=self._auth_headers(),
            params={
                "keys": "time,heartrate,cadence,distance,altitude,velocity_smooth,grade_smooth,moving,latlng",
                "key_by_type": "true",
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

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

    def parse_to_parquet(self, run: dict[str, Any]) -> dict[str, Any]:
        """Use to parse run data to Parquet format.

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
        detailed_response = requests.get(detailed_url, headers=self._auth_headers(), timeout=30)
        detailed_response.raise_for_status()
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
