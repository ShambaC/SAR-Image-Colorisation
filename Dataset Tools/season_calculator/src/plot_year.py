import matplotlib.pyplot as plt
import pandas as pd
from utils.api_service import get_daily_mean_temperature
from scipy.signal import savgol_filter

def plot_temperature_year(lat: float, lon: float, year: int):
    df = get_daily_mean_temperature(lat, lon, year)

    plt.figure(figsize=(15, 6))
    plt.plot(df["date"], df["temperature_2m_mean"], color="tab:blue", label="Daily Mean Temp")

    plt.title(f"Daily Mean Temperature in {year} (Lat: {lat}, Lon: {lon})")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_country_temperature(country: str, data_path: str = "../data/country_daily_avg_iso.parquet"):
    df = pd.read_parquet(data_path)
    df_country = df[df["country"] == country]

    if df_country.empty:
        print(f"No data found for {country}")
        return

    plt.figure(figsize=(15, 6))
    plt.plot(df_country["date"], df_country["temperature_2m_mean"], label=f"{country}", color="tab:red")

    plt.title(f"Daily Mean Temperature in {df_country['date'].dt.year.iloc[0]} - {country}")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_country_smoothed_yearly_temp(parquet_path: str, country_name: str, window: int = 15, polyorder: int = 2):
    """
    Plot smoothed daily average temperature for a given country using pandas.

    Parameters:
        parquet_path (str): Path to the .parquet file with daily temperature data.
        country_name (str): Name of the country (non-ISO allowed).
        window (int): Smoothing window for Savitzky-Golay filter (must be odd).
        polyorder (int): Polynomial order for the filter.
    """
    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")

    df = pd.read_parquet(parquet_path)
    df_country = df[df["country"] == country_name].sort_values("date")

    if df_country.empty:
        raise ValueError(f"No data found for country: {country_name}")

    dates = pd.to_datetime(df_country["date"])
    temps = df_country["temperature_2m_mean"].values

    if len(temps) < window:
        raise ValueError(f"Not enough data points for smoothing (need > {window})")

    smoothed = savgol_filter(temps, window, polyorder)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, temps, color='lightgray', alpha=0.6, label="Raw")
    plt.plot(dates, smoothed, color='blue', label="Smoothed")
    plt.title(f"Daily Avg Temperature for {country_name}")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_country_temperature("Australia")
    plot_country_smoothed_yearly_temp("../data/country_daily_avg_iso.parquet", "India")