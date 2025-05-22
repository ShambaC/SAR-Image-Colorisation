# process_country_temperatures.py

import json
import pandas as pd
from tqdm import tqdm
from utils.api_service import get_daily_mean_temperature

def process_all_countries(json_path: str, year: int, output_path: str = "../data/country_daily_avg.parquet"):
    with open(json_path, "r") as f:
        data = json.load(f)

    country_dfs = []

    for country, cities in tqdm(data.items(), desc="Processing countries"):
        city_dfs = []
        for city, coords in tqdm(cities.items(), leave=False):
            lat, lon = coords
            try:
                df = get_daily_mean_temperature(lat, lon, year)
                df = df.rename(columns={"temperature_2m_mean": f"{city}"})
                city_dfs.append(df.set_index("date"))
            except Exception as e:
                print(f"Error fetching data for {country} - {city}: {e}")
                continue

        if city_dfs:
            country_df = pd.concat(city_dfs, axis=1).mean(axis=1).reset_index()
            country_df.columns = ["date", "temperature_2m_mean"]
            country_df["country"] = country
            country_dfs.append(country_df)

    # Combine all countries
    all_countries_df = pd.concat(country_dfs)
    all_countries_df.to_parquet(output_path, index=False)
    print(f"Saved country-level data to {output_path}")

if __name__ == "__main__":
    process_all_countries("../data/city_coords.json", year=2024)
