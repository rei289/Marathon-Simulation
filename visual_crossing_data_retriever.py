"""
This file retrieves weather data from OpenWeather API
"""
import requests
from datetime import datetime
import time
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

class VisualCrossingDataRetriever:
    """
    Class to retrieve weather data from Visual Crossing API.
    """
    def __init__(self, output_folder: str = "data"):
        """
        Initialize the data retriever and load environment variables.
        """
        self.output_folder = output_folder
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """
        Load Visual Crossing API credentials from environment variables.
        """
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        
        if not self.api_key:
            raise ValueError("API_KEY must be set in environment variables")
        
    def get_weather_openweather(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve weather data for a given latitude, longitude, and ISO datetime.
        """
        lat = json_data['start_latlng'][0]
        lon = json_data['start_latlng'][1]
        iso_datetime = json_data['start_date']

        # Convert ISO datetime to Unix timestamp
        dt = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
        unix_time = int(dt.timestamp())
        # Construct the API URL
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        url = f"{base_url}/{lat},{lon}/{unix_time}?unitGroup=metric&key={self.api_key}"
        # Make the API request
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            # we now want to extract the relevant data
            # get the local time of the run
            local_iso_datetime = json_data['start_date_local']
            local_dt = datetime.fromisoformat(local_iso_datetime.replace("Z", "+00:00"))
            # find the hour in the weather data
            weather_data = data["days"][0]["hours"][int(local_dt.strftime("%H"))]
            return weather_data
        else:
            print(f"Error retrieving weather data: {response.status_code}")
            return {}
   