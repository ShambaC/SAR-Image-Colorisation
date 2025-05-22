import matplotlib.pyplot as plt
import pandas as pd
from utils.api_service import get_daily_mean_temperature
import pycountry

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

def get_iso_alpha2(country_name: str) -> str:
    """
    Convert a country name to its ISO 3166-1 alpha-2 code using pycountry.
    """
    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_2
    except LookupError:
        raise ValueError(f"Could not resolve country name to ISO alpha-2: {country_name}")

def plot_country_temperature(country: str, data_path: str = "../data/country_daily_avg_iso.parquet"):
    df = pd.read_parquet(data_path)
    df_country = df[df["country"] == get_iso_alpha2(country)]
    # df_country = df[df["country"] == country]

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

if __name__ == "__main__":
    # Example: plot a country
    plot_country_temperature("Australia")