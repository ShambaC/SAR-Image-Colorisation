import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Set up Open-Meteo client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_daily_mean_temperature(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetches daily mean temperature for a given latitude, longitude, and year.
    Returns a pandas DataFrame with columns: date, temperature_2m_mean
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": "temperature_2m_mean",
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # First (and only) location

    daily = response.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    temps = daily.Variables(0).ValuesAsNumpy()

    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": temps
    })

    return df
