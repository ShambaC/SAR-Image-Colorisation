import polars as pl
import numpy as np
from tqdm import tqdm

def circular_index(arr_len, start_idx, direction="backward"):
    if direction == "backward":
        return [(start_idx - i) % arr_len for i in range(1, arr_len)]
    else:
        return [(start_idx + i) % arr_len for i in range(1, arr_len)]


def find_transition_date(temps, dates, ref_idx, threshold, direction="backward", tolerance=0.4):
    indices = circular_index(len(temps), ref_idx, direction)
    for idx in indices:
        if abs(temps[idx] - threshold) <= tolerance:
            return dates[idx]
    raise ValueError("No matching threshold value found in circular search.")


def detect_season_boundaries(df: pl.DataFrame) -> dict:
    temps = df["temperature_2m_mean"].to_numpy()
    dates = df["date"].to_numpy()

    min_temp = temps.min()
    max_temp = temps.max()
    min_idx = np.argmin(temps)
    max_idx = np.argmax(temps)

    d = max_temp - min_temp
    lower_t = min_temp + d / 4
    upper_t = max_temp - d / 4

    # Winter
    winter_start = find_transition_date(temps, dates, min_idx, lower_t, "backward")
    winter_end = find_transition_date(temps, dates, min_idx, lower_t, "forward")

    # Summer
    summer_start = find_transition_date(temps, dates, max_idx, upper_t, "backward")
    summer_end = find_transition_date(temps, dates, max_idx, upper_t, "forward")

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
            season_data = detect_season_boundaries(df_country)
            results.append(season_data)
        except Exception as e:
            print(f"Skipping {code}: {e}")

    df_out = pl.DataFrame(results)
    df_out.write_parquet(output_parquet)


if __name__ == "__main__":
    generate_season_boundaries("../data/country_daily_avg_iso.parquet", "../data/country_season_boundaries.parquet")