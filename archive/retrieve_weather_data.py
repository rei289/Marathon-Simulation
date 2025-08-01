"""
This file retrieves weather data from OpenWeather API
"""
import requests
from datetime import datetime
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_weather_openweather(name: str, lat: float, lon: float, iso_datetime: str, api_key: str):
    # Convert ISO datetime to Unix timestamp
    dt = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
    # date_str = dt.strftime("%Y-%m-%dT%H:00:00")
    unix_time = int(dt.timestamp())

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{lat},{lon}/{unix_time}?key={api_key}"

    # url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&dt={dt}&appid={api_key}"
    # url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    # params = {
    #     "lat": lat,
    #     "lon": lon,
    #     "dt": unix_time,
    #     "appid": api_key,
    #     "units": "metric"
    # }

    response = requests.get(url)

    # response = requests.get(url, params=params)

    # json method of response object 
    # convert json format data into
    # python format data
    x = response.json()
    # get the weather data for that specific hour
    
    # Create a JSON file
    json_filename = f"test_overall.json"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)

    # json.dumps(x, indent=4)

# loop through the folder in which the weather data is stored
folder_path = "data"

for folder_name in os.listdir(folder_path):
    # get the json file
    for filename in os.listdir(folder_path+ "/" + folder_name):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path+"/"+folder_name, filename), "r") as f:
                weather_data = json.load(f)

            # get the lat and lon from the json file
            lat = weather_data.get("start_latlng")[0]
            lon = weather_data.get("start_latlng")[1]
            iso_datetime = weather_data.get("start_date")

            print(f"Retrieving weather data for {folder_name} at {iso_datetime} for coordinates ({lat}, {lon})")
            
            api_key = os.getenv("API_KEY")
            if api_key is None:
                print("Error: API_KEY environment variable not set!")
                continue
                
            get_weather_openweather(folder_name, lat, lon, iso_datetime, api_key)
            break
    break