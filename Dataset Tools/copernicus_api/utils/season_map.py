import polars as pl
from datetime import datetime
import pytz

df = pl.read_parquet("../../season_calculator/data/country_season_boundaries.parquet")

def classify_season(country: str, timestamp_iso: str, hemisphere: str) -> str:
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

# Example usage:
if __name__ == "__main__" :
    country = "FR"
    timestamp_iso = "2023-09-29T23:59:59Z"
    season = classify_season(country, timestamp_iso, "North")
    print(f"{timestamp_iso} in {country} is {season}")