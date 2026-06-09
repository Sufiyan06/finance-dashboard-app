from datetime import datetime, timedelta

import openmeteo_requests
import pandas as pd
import requests_cache
import streamlit as st
from retry_requests import retry

from finance_dashboard.config import CACHE_DIR, WEATHER_ARCHIVE_MIN_YEAR

_cache_session = requests_cache.CachedSession(str(CACHE_DIR), expire_after=3600)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.2)
_weather_client = openmeteo_requests.Client(session=_retry_session)


def _clamp_archive_dates(start_date: str, end_date: str) -> tuple[str, str] | None:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    archive_min = datetime(WEATHER_ARCHIVE_MIN_YEAR, 1, 1).date()
    archive_max = (datetime.today() - timedelta(days=1)).date()

    start = max(start, archive_min)
    end = min(end, archive_max)
    if start > end:
        return None

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@st.cache_data(show_spinner="Loading weather archive…")
def fetch_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "temp_max", "temp_min", "precipitation", "wind", "solar_radiation"])
    clamped = _clamp_archive_dates(start_date, end_date)
    if clamped is None:
        return empty

    start_str, end_str = clamped
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_str,
        "end_date": end_str,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "shortwave_radiation_sum",
        ],
        "timezone": "auto",
    }

    responses = _weather_client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    data = responses[0].Daily()

    return pd.DataFrame(
        {
            "date": pd.date_range(start=start_str, end=end_str, freq="D"),
            "temp_max": data.Variables(0).ValuesAsNumpy(),
            "temp_min": data.Variables(1).ValuesAsNumpy(),
            "precipitation": data.Variables(2).ValuesAsNumpy(),
            "wind": data.Variables(3).ValuesAsNumpy(),
            "solar_radiation": data.Variables(4).ValuesAsNumpy(),
        }
    )
