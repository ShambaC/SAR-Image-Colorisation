import polars as pl
import numpy as np
from tqdm import tqdm
from datetime import datetime


def circular_index(arr_len, start_idx, direction="backward"):
    return [(start_idx - i) % arr_len if direction == "backward" else (start_idx + i) % arr_len for i in range(1, arr_len)]


def days_between(start_dt, end_dt):
    """Calculate number of days between two datetime objects, accounting for year wrap."""
    start_dt = start_dt.replace(tzinfo=None)
    end_dt = end_dt.replace(tzinfo=None)

    if end_dt >= start_dt:
        return (end_dt - start_dt).days
    else:
        # wrap around year
        start_of_next_year = datetime(start_dt.year + 1, 1, 1)
        end_start_of_year = datetime(end_dt.year, 1, 1)
        return (start_of_next_year - start_dt).days + (end_dt - end_start_of_year).days


def find_transition_date(temps, dates, ref_idx, threshold, direction="backward", tolerance=0.4, min_days=30):
    indices = circular_index(len(temps), ref_idx, direction)
    ref_date = dates[ref_idx]

    for idx in indices:
        if abs(temps[idx] - threshold) <= tolerance:
            candidate_date = dates[idx]
            duration = days_between(ref_date, candidate_date)
            if duration >= min_days:
                return candidate_date
            # Else: keep looking for a date further away

    raise ValueError("No matching threshold value found that satisfies the minimum duration condition.")


def detect_season_boundaries(df: pl.DataFrame, min_days=30, tolerance=0.3) -> dict:
    temps = df["temperature_2m_mean"].to_numpy()
    dates = df["date"].to_list()

    min_temp = temps.min()
    max_temp = temps.max()
    min_idx = np.argmin(temps)
    max_idx = np.argmax(temps)

    d = max_temp - min_temp
    lower_t = min_temp + d / 4
    upper_t = max_temp - d / 4

    # Winter
    winter_start = find_transition_date(temps, dates, min_idx, lower_t, "backward", tolerance=tolerance, min_days=min_days)
    winter_end   = find_transition_date(temps, dates, min_idx, lower_t, "forward", tolerance=tolerance, min_days=min_days)

    # Summer
    summer_start = find_transition_date(temps, dates, max_idx, upper_t, "backward", tolerance=tolerance, min_days=min_days)
    summer_end   = find_transition_date(temps, dates, max_idx, upper_t, "forward", tolerance=tolerance, min_days=min_days)

    # Infer spring and fall
    spring_start = winter_end
    spring_end = summer_start
    fall_start = summer_end
    fall_end = winter_start

    return {
        "country": df["country"][0],
        "winter_start": winter_start,
        "winter_end": winter_end,
        "spring_start": spring_start,
        "spring_end": spring_end,
        "summer_start": summer_start,
        "summer_end": summer_end,
        "fall_start": fall_start,
        "fall_end": fall_end
    }


def generate_season_boundaries(input_parquet: str, output_parquet: str):
    df = pl.read_parquet(input_parquet)
    country_codes = df["country"].unique().to_list()
    results = []

    for code in tqdm(country_codes, desc="Computing season boundaries"):
        df_country = df.filter(pl.col("country") == code).sort("date")
        try:
            season_data = detect_season_boundaries(df_country, min_days=15, tolerance=0.3)
            results.append(season_data)
        except Exception as e:
            print(f"Skipping {code}: {e}")

    df_out = pl.DataFrame(results)
    df_out.write_parquet(output_parquet)


if __name__ == "__main__":
    generate_season_boundaries(
        "../data/country_daily_avg_iso.parquet",
        "../data/country_season_boundaries.parquet"
    )