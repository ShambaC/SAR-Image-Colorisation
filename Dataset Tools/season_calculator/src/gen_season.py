import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from tqdm import tqdm
from datetime import datetime

def detect_season_boundaries(df: pl.DataFrame, smoothing_window: int = 15, polyorder: int = 2) -> pl.DataFrame:
    """
    Detect adaptive seasonal boundaries in daily temperature data.

    Parameters:
        df (pl.DataFrame): A Polars DataFrame with 'date' and 'temperature_2m_mean' columns.
        smoothing_window (int): Window size for Savitzky-Golay filter (must be odd).
        polyorder (int): Polynomial order for smoothing.

    Returns:
        pl.DataFrame: DataFrame with added 'season' column.
    """
    if smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be odd")

    df = df.sort("date")
    temperatures = df["temperature_2m_mean"].to_numpy()
    n = len(df)

    smoothed_temp = savgol_filter(temperatures, smoothing_window, polyorder)
    first_derivative = np.gradient(smoothed_temp)

    # Peak (summer) and trough (winter)
    peak_idx = np.argmax(smoothed_temp)
    trough_idx = np.argmin(smoothed_temp)

    def find_transition(start, direction, condition):
        step = 1 if direction == "forward" else -1
        i = start
        while 0 <= i < n:
            if condition(first_derivative[i]):
                return i
            i += step
        return start

    winter_start = find_transition(trough_idx, "backward", lambda x: x < -0.001)
    winter_end = find_transition(trough_idx, "forward", lambda x: x > 0.001)
    summer_start = find_transition(peak_idx, "backward", lambda x: x > 0.001)
    summer_end = find_transition(peak_idx, "forward", lambda x: x < -0.001)

    season_labels = [""] * n

    def label_range(start, end, label):
        if start <= end:
            for i in range(start, end):
                season_labels[i] = label
        else:
            for i in list(range(start, n)) + list(range(0, end)):
                season_labels[i] = label

    label_range(winter_start, winter_end, "Winter")
    label_range(winter_end, summer_start, "Spring")
    label_range(summer_start, summer_end, "Summer")
    label_range(summer_end, winter_start, "Fall")

    return df.with_columns(pl.Series("season", season_labels))


def process_all_countries(parquet_path: str, output_path: str):
    """
    Load ISO-country-level parquet, process season labels for each country, and save result.

    Parameters:
        parquet_path (str): Path to input parquet file with columns ['country', 'date', 'temperature_2m_mean'].
        output_path (str): Path to output parquet with added 'season' column.
    """
    df = pl.read_parquet(parquet_path)
    countries = df["country"].unique().to_list()

    results = []
    print(f"Processing {len(countries)} countries...")
    for country in tqdm(countries, desc="Processing countries"):
        df_country = df.filter(pl.col("country") == country)
        df_seasonal = detect_season_boundaries(df_country)
        results.append(df_seasonal)

    combined = pl.concat(results)
    combined.write_parquet(output_path)
    print(f"\n✅ Saved processed seasonal data to: {output_path}")


# Example usage
if __name__ == "__main__":
    process_all_countries("../data/country_daily_avg_iso.parquet", "../data/country_daily_with_seasons.parquet")