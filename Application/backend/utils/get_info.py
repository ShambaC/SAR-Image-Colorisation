# Load environment variables
from dotenv import load_dotenv
env_res = load_dotenv()

if not env_res :
    raise Exception("No .env file provided")

import os
import random
import requests
import json

import polars as pl
from datetime import datetime
import pytz

df = pl.read_parquet("../../Dataset Tools/season_calculator/data/country_season_boundaries.parquet")

def get_country(lat: float, lon: float) -> str :
    api_url = f'https://api.api-ninjas.com/v1/reversegeocoding?lat={lat}&lon={lon}'

    API_KEY_LIST = ["API_NINJA_TOKEN_1", "API_NINJA_TOKEN_2"]

    response = requests.get(api_url, headers={'X-Api-Key': os.environ[random.choice(API_KEY_LIST)]})
    if response.status_code == requests.codes.ok:
        try:
            if response.text == "[]" :
                return "error"
            
            x = json.loads(response.text[1:-1])
            return x["country"]
        except :
            print(f"Response: {response.text}")
            raise Exception("API Ninja error occured")

    else:
        print("Error:", response.status_code, response.text)
        return "error"
    
def calc_hemisphere(lat: float) -> str :
    return "north" if lat > 0 else "south"
    
def get_season(country: str, timestamp_iso: str, hemisphere: str) -> str:
    # Parse the timestamp (assume UTC if 'Z' is present)
    if timestamp_iso.endswith('Z'):
        timestamp = datetime.strptime(timestamp_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
    else:
        timestamp = datetime.fromisoformat(timestamp_iso)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)

    # Get the row for the country
    row = df.filter(pl.col("country") == country)
    if row.height == 0:
        # Use meteorological seasons if country not found
        month = timestamp.month
        if hemisphere.lower() == "north":
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            elif month in [9, 10, 11]:
                return "fall"
        elif hemisphere.lower() == "south":
            if month in [12, 1, 2]:
                return "summer"
            elif month in [3, 4, 5]:
                return "fall"
            elif month in [6, 7, 8]:
                return "winter"
            elif month in [9, 10, 11]:
                return "spring"
        return "unknown"
    row = row.row(0)

    # Map season names to their start/end columns
    seasons = [
        ("winter", row[1], row[2]),
        ("spring", row[3], row[4]),
        ("summer", row[5], row[6]),
        ("fall",   row[7], row[8]),
    ]

    # Compare only month and day
    ts_md = (timestamp.month, timestamp.day)
    for season, start, end in seasons:
        start_md = (start.month, start.day)
        end_md = (end.month, end.day)
        if start_md <= end_md:
            if start_md <= ts_md <= end_md:
                return season
        else:
            # Season crosses year boundary
            if ts_md >= start_md or ts_md <= end_md:
                return season
    return "unknown"

def get_region(lat: float) -> str :
    region = ""
    if lat <= 23.5 and lat >= -23.5 :
        region = "tropical"
    elif (lat > 23.5 and lat < 66.5) or (lat < -23.5 and lat > -66.5) :
        region = "temperate"
    else :
        region = "arctic"

    return region

if __name__ == "__main__" :
    get_country(76.16237104394712, -109.54512274485435)