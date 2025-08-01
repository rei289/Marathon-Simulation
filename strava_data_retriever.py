"""
Structured Strava Data Retriever

This module provides a class-based approach to retrieve and save running data from Strava API.
It organizes the functionality into logical components for better maintainability and reusability.
"""

from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import os
import json
import csv
from dateutil import parser
from typing import List, Dict, Any, Optional, Tuple


class StravaDataRetriever:
    """A class to handle Strava API authentication and data retrieval."""
    
    def __init__(self, output_folder: str = "data"):
        """
        Initialize the Strava data retriever.
        
        Args:
            output_folder (str): The folder where data will be saved
        """
        self.output_folder = output_folder
        self.session = None
        self._load_credentials()
        
    def _load_credentials(self) -> None:
        """
        Load Strava API credentials from environment variables.
        """
        load_dotenv()
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_url = "https://localhost"
        
        if not self.client_id or not self.client_secret:
            raise ValueError("CLIENT_ID and CLIENT_SECRET must be set in environment variables")
    
    def authenticate(self) -> None:
        """
        Handles OAuth2 authentication with Strava.
        """
        # Create OAuth2 session
        self.session = OAuth2Session(
            client_id=self.client_id, 
            redirect_uri=self.redirect_url, 
            scope=["activity:read_all"]
        )
        
        # Get authorization URL
        auth_url, _ = self.session.authorization_url("https://www.strava.com/oauth/authorize")
        print(f"👉 Click this URL to authorize access:\n{auth_url}")
        
        # Get redirect response from user
        redirect_response = input("\n🔁 Paste the full redirect URL after approving: ")
        
        # Fetch access token
        token_url = "https://www.strava.com/api/v3/oauth/token"
        token = self.session.fetch_token(
            token_url=token_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            authorization_response=redirect_response,
            include_client_id=True,
        )
        
        print("✅ Authentication successful!")
    
    def fetch_activities(self, per_page: int = 50, page: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch activities from Strava API.
        
        Args:
            per_page (int): Number of activities per page
            page (int): Page number to fetch
            
        Returns:
            List[Dict[str, Any]]: List of activity data
        """
        if not self.session:
            raise ValueError("Must authenticate before fetching activities")
        
        print("\n📥 Fetching your latest activities...")
        activities_url = "https://www.strava.com/api/v3/athlete/activities"
        params = {"per_page": per_page, "page": page}
        response = self.session.get(activities_url, params=params)
        response.raise_for_status()
        return response.json()
    
    def filter_runs(self, activities: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Filter activities to get only runs.
        
        Args:
            activities (List[Dict[str, Any]]): List of all activities
            limit (int): Maximum number of runs to return
            
        Returns:
            List[Dict[str, Any]]: List of run activities
        """
        runs = [act for act in activities if act.get("type") == "Run"][:limit]
        print(f"📊 Found {len(runs)} recent runs")
        return runs
    
    # def _clean_filename(self, s: str) -> str:
    #     """Clean filename by replacing special characters."""
    #     return re.sub(r"[^\w\-_\. ]", "_", s)
    
    def _create_date_string(self, start_date: str) -> str:
        """Convert start date to formatted string."""
        return parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")
    
    # def _setup_output_directory(self, date_str: str) -> str:
    #     """
    #     Create output directory structure.
        
    #     Args:
    #         date_str (str): Date string for folder name
            
    #     Returns:
    #         str: Path to the run-specific folder
    #     """
    #     os.makedirs(self.output_folder, exist_ok=True)
    #     run_folder_path = os.path.join(self.output_folder, date_str)
    #     os.makedirs(run_folder_path, exist_ok=True)
    #     return run_folder_path
    
    # def save_run_overview(self, run: Dict[str, Any], run_folder_path: str, date_str: str) -> None:
    #     """
    #     Save run overview data as JSON.
        
    #     Args:
    #         run (Dict[str, Any]): Run data
    #         run_folder_path (str): Path to run folder
    #         date_str (str): Date string for filename
    #     """
    #     json_filename = f"{date_str}_overall.json"
    #     json_filepath = os.path.join(run_folder_path, json_filename)
        
    #     with open(json_filepath, "w", encoding="utf-8") as f:
    #         json.dump(run, f, indent=2)
        
    #     print(f"✅ Saved overview: {json_filename}")
    
    def fetch_activity_streams(self, run_id: int) -> Dict[str, Any]:
        """
        Fetch detailed stream data for a specific activity.
        
        Args:
            run_id (int): Strava activity ID
            
        Returns:
            Dict[str, Any]: Stream data from Strava
        """
        if not self.session:
            raise ValueError("Must authenticate before fetching streams")
        
        streams_url = f"https://www.strava.com/api/v3/activities/{run_id}/streams"
        params = {
            "keys": "time,heartrate,cadence,distance,altitude,velocity_smooth,grade_smooth,moving,latlng",
            "key_by_type": "true"
        }
        streams_response = self.session.get(streams_url, params=params)
        streams_response.raise_for_status()
        return streams_response.json()
    
    def _pad_list(self, lst: List[Any], target_length: int) -> List[Any]:
        """Pad list with None values to reach target length."""
        return lst + [None] * (target_length - len(lst))
    
    def _extract_coordinates(self, latlng_data: List[Any]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Extract latitude and longitude from coordinate data.
        
        Args:
            latlng_data (List[Any]): List of [lat, lng] pairs
            
        Returns:
            Tuple[List[Optional[float]], List[Optional[float]]]: Latitude and longitude lists
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
    
    # def save_run_streams(self, run: Dict[str, Any], run_folder_path: str, date_str: str) -> bool:
    #     """
    #     Save detailed stream data as CSV.
        
    #     Args:
    #         run (Dict[str, Any]): Run data
    #         run_folder_path (str): Path to run folder
    #         date_str (str): Date string for filename
            
    #     Returns:
    #         bool: True if successful, False if no data available
    #     """
    #     run_id = run["id"]
    #     name = run.get("name", "Unnamed_run")
        
    #     # Fetch stream data
    #     streams = self.fetch_activity_streams(run_id)
        
    #     # Extract stream data with defaults
    #     time_data = streams.get("time", {}).get("data", [])
    #     if not time_data:
    #         print(f"⚠️ No time stream data for run '{name}' ({run_id}), skipping streams.")
    #         return False
        
    #     # Extract all stream types
    #     stream_data = {
    #         'heartrate': streams.get("heartrate", {}).get("data", []),
    #         'distance': streams.get("distance", {}).get("data", []),
    #         'cadence': streams.get("cadence", {}).get("data", []),
    #         'altitude': streams.get("altitude", {}).get("data", []),
    #         'watts': streams.get("watts", {}).get("data", []),
    #         'temp': streams.get("temp", {}).get("data", []),
    #         'velocity': streams.get("velocity_smooth", {}).get("data", []),
    #         'grade': streams.get("grade_smooth", {}).get("data", []),
    #         'moving': streams.get("moving", {}).get("data", []),
    #         'latlng': streams.get("latlng", {}).get("data", [])
    #     }
        
    #     # Pad all lists to match time_data length
    #     max_len = len(time_data)
    #     for key in stream_data:
    #         stream_data[key] = self._pad_list(stream_data[key], max_len)
        
    #     # Extract coordinates
    #     lat, lng = self._extract_coordinates(stream_data['latlng'])
        
    #     # Save to CSV
    #     csv_filename = f"{date_str}_streams.csv"
    #     csv_filepath = os.path.join(run_folder_path, csv_filename)
        
    #     with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow([
    #             "time_s", "heartrate_bpm", "cadence_rpm", "distance_m", "altitude_m", "watts",
    #             "temp_c", "velocity_mps", "grade_percent", "moving", "latitude", "longitude"
    #         ])
            
    #         for row in zip(
    #             time_data, stream_data['heartrate'], stream_data['cadence'], 
    #             stream_data['distance'], stream_data['altitude'], stream_data['watts'], 
    #             stream_data['temp'], stream_data['velocity'], stream_data['grade'], 
    #             stream_data['moving'], lat, lng
    #         ):
    #             writer.writerow(row)
        
    #     print(f"✅ Saved streams for '{name}' to {csv_filename}")
    #     return True
    
    def parse_to_csv(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse run data to CSV format.
        
        Args:
            run (Dict[str, Any]): Run data
            
        Returns:
            str: CSV formatted string of run data
        """

        run_id = run["id"]
        name = run.get("name", "Unnamed_run")
        
        # Fetch stream data
        streams = self.fetch_activity_streams(run_id)
        
        # Extract stream data with defaults
        time_data = streams.get("time", {}).get("data", [])
        if not time_data:
            print(f"⚠️ No time stream data for run '{name}' ({run_id}), skipping streams.")
            return False
        
        # Extract all stream types
        stream_data = {
            'time': time_data,
            'heartrate': streams.get("heartrate", {}).get("data", []),
            'distance': streams.get("distance", {}).get("data", []),
            'cadence': streams.get("cadence", {}).get("data", []),
            'altitude': streams.get("altitude", {}).get("data", []),
            'velocity': streams.get("velocity_smooth", {}).get("data", []),
            'grade': streams.get("grade_smooth", {}).get("data", []),
            'moving': streams.get("moving", {}).get("data", []),
            'latlng': streams.get("latlng", {}).get("data", [])
        }
        
        # Pad all lists to match time_data length
        max_len = len(time_data)
        for key in stream_data:
            stream_data[key] = self._pad_list(stream_data[key], max_len)
        
        # Extract coordinates
        lat, lng = self._extract_coordinates(stream_data['latlng'])

        # put this into the stream_data
        stream_data['latitude'] = lat
        stream_data['longitude'] = lng

        # remove latlng from stream_data
        del stream_data['latlng']

        return stream_data
    
    def parse_to_json(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse run data to JSON format.
        
        Args:
            run (Dict[str, Any]): Run data
            
        Returns:
            Dict[str, Any]: JSON formatted run data
        """
        # Extract relevant fields for JSON
        json_data = {
            # basic run information
            "id": run.get("id"),
            "name": run.get("name"),
            "type": run.get("type"),

            # time information
            "start_date": run.get("start_date"),
            "start_date_local": run.get("start_date_local"),
            "timezone": run.get("timezone"),
            "utc_offset": run.get("utc_offset"),

            # run statistics
            "distance": run.get("distance"),
            "moving_time": run.get("moving_time"),
            "elapsed_time": run.get("elapsed_time"),

            # elevations
            "total_elevation_gain": run.get("total_elevation_gain"),
            "elevation_high": run.get("elevation_high"),
            "elevation_low": run.get("elevation_low"),
            "start_latlng": run.get("start_latlng"),
            "end_latlng": run.get("end_latlng"),

            # pace and speed
            "average_speed": run.get("average_speed"),
            "max_speed": run.get("max_speed"),
            "average_cadence": run.get("average_cadence"),
        }
        
        return json_data
    
    # def process_run(self, run: Dict[str, Any]) -> None:
    #     """
    #     Process a single run: save overview and streams data.
        
    #     Args:
    #         run (Dict[str, Any]): Run data from Strava
    #     """
    #     name = run.get("name", "Unnamed_run")
    #     start_date = run.get("start_date_local", "")
    #     date_str = self._create_date_string(start_date)
        
    #     # Setup directory
    #     run_folder_path = self._setup_output_directory(date_str)
        
    #     # Save overview data
    #     self.save_run_overview(run, run_folder_path, date_str)
        
    #     # Save streams data
    #     self.save_run_streams(run, run_folder_path, date_str)
    
    # def retrieve_and_save_runs(self, num_runs: int = 10) -> None:
    #     """
    #     Main method to retrieve and save running data.
        
    #     Args:
    #         num_runs (int): Number of recent runs to process
    #     """
    #     try:
    #         # Authenticate
    #         self.authenticate()
            
    #         # Fetch activities
    #         activities = self.fetch_activities()
            
    #         # Filter runs
    #         runs = self.filter_runs(activities, limit=num_runs)
            
    #         # Setup output folder
    #         print(f"\n📁 Saving data to folder: {self.output_folder}\n")
            
    #         # Process each run
    #         for i, run in enumerate(runs, 1):
    #             print(f"\n--- Processing run {i}/{len(runs)} ---")
    #             self.process_run(run)
            
    #         print("\n✅ Done saving all running data!")
            
    #     except Exception as e:
    #         print(f"❌ Error: {e}")
    #         raise


# def main():
#     """Main function to run the data retrieval process."""
#     # You can customize the output folder here
#     retriever = StravaDataRetriever(output_folder="data")
    
#     # Retrieve and save the 10 most recent runs
#     retriever.retrieve_and_save_runs(num_runs=10)


# if __name__ == "__main__":
#     main()
