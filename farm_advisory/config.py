"""Central configuration for the farm advisory application layer.

The advisory layer glues the three trained subsystems together and adds
live weather, live IoT ingestion, a recommendation engine and automated
delivery. Edit the values below to point at a different farm or a
different set of model artifacts.
"""

import os

# ---------------------------------------------------------------------------
# Resolved paths (do not edit unless you relocate folders)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SENSOR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'sensor_model', 'rf_model.joblib')
SENSOR_THRESHOLDS_PATH = os.path.join(
    PROJECT_ROOT, 'sensor_model', 'dataset', 'selected_thresholds.json'
)
LEAF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'leaf_disease_model', 'best_leaf_disease_model.pth')
LEAF_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'leaf_disease_model', 'dataset')
SATELLITE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, 'satellite_weather_model', 'satellite_weather_model.pth'
)

SENSOR_FEATURES = [
    'Temperature', 'Humidity', 'Moisture', 'Nitrogen',
    'Phosphorus', 'Potassium', 'PH', 'Light_Intensity',
]

# ---------------------------------------------------------------------------
# Farm / location configuration (Open-Meteo takes decimal lat/lon)
# ---------------------------------------------------------------------------
FARM_LATITUDE = 28.6139
FARM_LONGITUDE = 77.2090
CROP_NAME = 'Wheat'

# ---------------------------------------------------------------------------
# Agronomic advisory thresholds (kept in sync with sensor_model thresholds)
# ---------------------------------------------------------------------------
SOIL_ADVISORY_THRESHOLDS = {
    'Moisture': {'low': 30.0, 'high': 70.0},
    'PH': {'low': 6.0, 'high': 7.5},
    'Nitrogen': {'low': 30.0},
    'Phosphorus': {'low': 20.0},
    'Potassium': {'low': 25.0},
    'Humidity': {'high': 85.0},
    'Temperature': {'high': 32.0},
}

# Weather-code driven advisories
RAIN_DELAY_SPRAYING = True          # delay spraying when rain is expected

# ---------------------------------------------------------------------------
# Live data feeds
# ---------------------------------------------------------------------------
WEATHER_TTL_SECONDS = 30 * 60       # cache fresh weather for 30 minutes
WEATHER_CACHE_PATH = os.path.join(BASE_DIR, 'dataset', 'weather_cache.json')
SENSOR_LOG_PATH = os.path.join(BASE_DIR, 'dataset', 'sensor_log.csv')

# ---------------------------------------------------------------------------
# Automated delivery (all disabled by default - set to enable)
# ---------------------------------------------------------------------------
REPORT_DIR = os.path.join(BASE_DIR, 'outputs')
DELIVERY = {
    'console': True,     # always print summary to console
    'file': True,        # always write markdown + json reports
    'email': {
        'enabled': False,
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': '',          # from-address
        'password': '',          # app password
        'to': [],                # list of recipient addresses
    },
    'telegram': {
        'enabled': False,
        'bot_token': '',
        'chat_id': '',
    },
}