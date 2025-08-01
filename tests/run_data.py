from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import os
import json

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
params = {
    "per_page": 50,
    "page": 1
}

response = session.get(activities_url, params=params)
activities = response.json()

# === Filter only "Run" activities ===
runs = [act for act in activities if act.get("type") == "Run"][:10]

print(f"\n🏃 Found {len(runs)} recent runs:\n")

for i, run in enumerate(runs, 1):
    name = run["name"]
    distance_km = run["distance"] / 1000  # meters to km
    moving_time = run["moving_time"] / 60  # seconds to minutes
    pace = (run["moving_time"] / 60) / distance_km if distance_km else 0
    date = run["start_date_local"]

    print(f"{i}. {name}")
    print(f"   📅 Date       : {date}")
    print(f"   📏 Distance   : {distance_km:.2f} km")
    print(f"   ⏱️ Time        : {moving_time:.1f} min")
    print(f"   🐾 Avg Pace   : {pace:.2f} min/km")
    print("-" * 40)
