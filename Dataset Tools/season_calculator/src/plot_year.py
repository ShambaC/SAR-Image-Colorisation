import matplotlib.pyplot as plt
from utils.api_service import get_daily_mean_temperature

def plot_temperature_year(lat: float, lon: float, year: int):
    df = get_daily_mean_temperature(lat, lon, year)

    plt.figure(figsize=(15, 6))
    plt.plot(df["date"], df["temperature_2m_mean"], color="tab:blue", label="Daily Mean Temp")

    plt.title(f"Daily Mean Temperature in {year} (Lat: {lat}, Lon: {lon})")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    # Improve x-axis formatting
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: Berlin
    latitude = 60.505
    longitude = 22.4583
    year = 2024

    plot_temperature_year(latitude, longitude, year)
