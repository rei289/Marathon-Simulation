from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import os
import json
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
output_folder = "run_data"
os.makedirs(output_folder, exist_ok=True)

print(f"\n📁 Saving data to folder: {output_folder}\n")

def clean_filename(s):
    return re.sub(r"[^\w\-_\. ]", "_", s)

# === Save each run as a JSON file ===
for i, run in enumerate(runs, 1):
    name = run.get("name", "Unnamed Run")
    date = run.get("start_date_local", "unknown_date")
    date_str = parser.isoparse(date).strftime("%Y-%m-%d_%H-%M")

    filename = f"{clean_filename(name)}_{date_str}.json"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    print(f"✅ Saved: {filename}")

print("\n🎉 All run data saved!")
