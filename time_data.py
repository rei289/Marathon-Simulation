from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import os
import json
import csv
import re
from dateutil import parser

# === Load credentials ===
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
redirect_url = "https://localhost"

# === Create OAuth2 session ===
session = OAuth2Session(client_id=client_id, redirect_uri=redirect_url, scope=["activity:read_all"])

# === Get auth URL and prompt user ===
auth_url, _ = session.authorization_url("https://www.strava.com/oauth/authorize")
print(f"👉 Click this URL to authorize access:\n{auth_url}")

redirect_response = input("\n🔁 Paste the full redirect URL after approving: ")

# === Fetch access token ===
token_url = "https://www.strava.com/api/v3/oauth/token"
token = session.fetch_token(
    token_url=token_url,
    client_id=client_id,
    client_secret=client_secret,
    authorization_response=redirect_response,
    include_client_id=True,
)

# === Fetch activities ===
print("\n📥 Fetching your latest activities...")
activities_url = "https://www.strava.com/api/v3/athlete/activities"
params = {"per_page": 50, "page": 1}
response = session.get(activities_url, params=params)
activities = response.json()

# === Filter only the 10 most recent runs ===
runs = [act for act in activities if act.get("type") == "Run"][:10]

# === Create output folder ===
output_folder = "data"
# output_folder = "run_data"
os.makedirs(output_folder, exist_ok=True)

print(f"\n📁 Saving data to folder: {output_folder}\n")

def clean_filename(s):
    return re.sub(r"[^\w\-_\. ]", "_", s)

# === Save each run in 2 files, JSON and CSV file
for run in runs:
    run_id = run["id"]
    name = run.get("name", "Unnamed_run")
    start_date = run.get("start_date_local", "")
    date_str = parser.isoparse(start_date).strftime("%Y-%m-%d_%H-%M")

    ## create a folder under output_folder for each
    run_folder = f"{date_str}"
    os.makedirs(output_folder+"/"+run_folder, exist_ok=True)

    ## Create a JSON file
    json_filename = f"{date_str}.json"
    json_filepath = os.path.join(output_folder+"/"+run_folder, json_filename)

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    print(f"✅ Saved: {json_filename}")

    ## Create CSV file
    csv_filename = f"{date_str}_streams.csv"
    # filename = f"{clean_filename(name)}_{date_str}_streams.csv"
    csv_filepath = os.path.join(output_folder+"/"+run_folder, csv_filename)

    # Request multiple streams at once
    streams_url = f"https://www.strava.com/api/v3/activities/{run_id}/streams"
    params = {
        # "keys": "time,heartrate,distance,cadence,altitude",
        "keys": "time,heartrate,cadence,distance,altitude,watts,temp,velocity_smooth,grade_smooth,moving,latlng",
        "key_by_type": "true"
    }
    streams_response = session.get(streams_url, params=params)
    streams_response.raise_for_status()
    streams = streams_response.json()

    # Extract streams, default to empty list if not available
    time_data = streams.get("time", {}).get("data", [])
    heartrate = streams.get("heartrate", {}).get("data", [])
    distance = streams.get("distance", {}).get("data", [])
    cadence = streams.get("cadence", {}).get("data", [])
    altitude = streams.get("altitude", {}).get("data", [])
    watts = streams.get("watts", {}).get("data", [])
    temp = streams.get("temp", {}).get("data", [])
    velocity = streams.get("velocity_smooth", {}).get("data", [])
    grade = streams.get("grade_smooth", {}).get("data", [])
    moving = streams.get("moving", {}).get("data", [])
    latlng = streams.get("latlng", {}).get("data", [])
    
    if not time_data:
        print(f"No time stream data for run {name} ({run_id}), skipping.")
        continue

    # Pad shorter lists with None to align all columns
    max_len = len(time_data)
    def pad_list(lst):
        return lst + [None] * (max_len - len(lst))

    heartrate = pad_list(heartrate)
    cadence = pad_list(cadence)
    distance = pad_list(distance)
    altitude = pad_list(altitude)
    watts = pad_list(watts)
    temp = pad_list(temp)
    velocity = pad_list(velocity)
    grade = pad_list(grade)
    moving = pad_list(moving)
    latlng = pad_list(latlng)

    # latlng is a list of [lat, lng] pairs; separate them or put None
    lat = []
    lng = []
    for point in latlng:
        if point and isinstance(point, list) and len(point) == 2:
            lat.append(point[0])
            lng.append(point[1])
        else:
            lat.append(None)
            lng.append(None)

    # Save to CSV
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "time_s", "heartrate_bpm", "cadence_rpm", "distance_m", "altitude_m", "watts",
            "temp_c", "velocity_mps", "grade_percent", "moving", "latitude", "longitude"
        ])

        for row in zip(time_data, heartrate, cadence, distance, altitude, watts, temp, velocity, grade, moving, lat, lng):
            writer.writerow(row)

    print(f"Saved streams for '{name}' to {csv_filepath}")

print("✅ Done saving all streams data!")
