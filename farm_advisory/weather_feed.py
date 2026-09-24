"""Live weather feed module.

Fetches current conditions and a short forecast from the Open-Meteo API
(no API key required). Results are cached on disk for ``ttl_seconds`` so a
long-running pipeline does not hammer the API. In case of a network failure
the module degrades gracefully to the last cached reading (or a conservative
local fallback) so downstream advisories always have weather to work with.
"""

import os
import json
import time
import datetime as dt
import urllib.request

from config import FARM_LATITUDE, FARM_LONGITUDE, WEATHER_TTL_SECONDS, WEATHER_CACHE_PATH

# WMO weather interpretation codes -> human-friendly labels
WMO_LABELS = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Depositing rime fog',
    51: 'Light drizzle', 53: 'Drizzle', 55: 'Dense drizzle',
    56: 'Freezing drizzle', 57: 'Dense freezing drizzle',
    61: 'Light rain', 63: 'Moderate rain', 65: 'Heavy rain',
    66: 'Freezing rain', 67: 'Freezing rain (heavy)',
    71: 'Light snow', 73: 'Moderate snow', 75: 'Heavy snow', 77: 'Snow grains',
    80: 'Light rain showers', 81: 'Rain showers', 82: 'Violent rain showers',
    85: 'Snow showers', 86: 'Heavy snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with hail', 99: 'Thunderstorm with heavy hail',
}


def weather_code_label(code):
    return WMO_LABELS.get(int(code), f'Unknown ({code})')


class LiveWeatherFeed:
    """Polls current + short-term weather and caches the latest snapshot."""

    def __init__(self, latitude=FARM_LATITUDE, longitude=FARM_LONGITUDE,
                 cache_path=WEATHER_CACHE_PATH, ttl_seconds=WEATHER_TTL_SECONDS):
        self.latitude = latitude
        self.longitude = longitude
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------- low level I/O
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as handle:
                    return json.load(handle)
            except Exception:
                return None
        return None

    def _save_cache(self, payload):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

    # ------------------------------------------------------------ fetching
    def fetch_current(self, force=False):
        """Return a normalized weather snapshot dict.

        Uses a fresh API call when the cache is stale (or ``force``), otherwise
        the cached snapshot. Falls back to old cache / placeholder values only
        when the network call itself fails and nothing fresh is available.
        """
        cached = self._load_cache()
        if not force and cached and cached.get('fetched_at'):
            age = time.time() - cached['fetched_at']
            if age < self.ttl_seconds:
                return cached['weather']

        try:
            url = (
                'https://api.open-meteo.com/v1/forecast'
                f'?latitude={self.latitude}&longitude={self.longitude}'
                '&current=temperature_2m,relative_humidity_2m,precipitation,'
                'weather_code,wind_speed_10m'
                '&daily=temperature_2m_max,temperature_2m_min,'
                'precipitation_probability_max,precipitation_sum'
                '&forecast_days=3&timezone=auto'
            )
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))

            current = data['current']
            daily = data.get('daily', {})
            weather = {
                'source': 'open-meteo',
                'latitude': self.latitude,
                'longitude': self.longitude,
                'observed_at': current.get('time'),
                'temperature': current.get('temperature_2m'),
                'humidity': current.get('relative_humidity_2m'),
                'precipitation_mm': current.get('precipitation'),
                'wind_speed_kmh': current.get('wind_speed_10m'),
                'weather_code': current.get('weather_code'),
                'weather_label': weather_code_label(current.get('weather_code')),
                'forecast': {
                    'dates': daily.get('time', []),
                    't_max': daily.get('temperature_2m_max', []),
                    't_min': daily.get('temperature_2m_min', []),
                    'precip_probability': daily.get('precipitation_probability_max', []),
                    'precip_sum': daily.get('precipitation_sum', []),
                },
            }
            self._save_cache({'fetched_at': time.time(), 'weather': weather})
            return weather

        except Exception as exc:
            print(f'[weather_feed] API error: {exc}')
            if cached and cached.get('weather'):
                print('[weather_feed] using last cached reading')
                weather = dict(cached['weather'])
                weather['source'] = 'cache'
                return weather
            # Conservative offline fallback (moderate, dry conditions).
            return {
                'source': 'offline-fallback',
                'latitude': self.latitude,
                'longitude': self.longitude,
                'observed_at': dt.datetime.now().isoformat(timespec='seconds'),
                'temperature': 26.0,
                'humidity': 60.0,
                'precipitation_mm': 0.0,
                'wind_speed_kmh': 8.0,
                'weather_code': 1,
                'weather_label': weather_code_label(1),
                'forecast': {'dates': [], 't_max': [], 't_min': [],
                             'precip_probability': [], 'precip_sum': []},
            }

    def fetch_forecast(self):
        return self.fetch_current().get('forecast', {})

    # ------------------------------------------------------------ summary
    def describe(self):
        w = self.fetch_current()
        return (
            f"{w['weather_label']} | {w['temperature']} C | {w['humidity']}% RH | "
            f"precip {w['precipitation_mm']} mm | wind {w['wind_speed_kmh']} km/h "
            f"({w['source']})"
        )


if __name__ == '__main__':
    feed = LiveWeatherFeed()
    print(feed.describe())