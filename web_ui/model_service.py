"""Shared inference helpers for the Flask web UI.

Every trained subsystem is loaded lazily and cached in a module-level
singleton so web requests stay fast after the first call. All deep models
run on CPU inside the web layer to keep the pipeline simple and portable.

Artifacts resolved:
    sensor_model/rf_model.joblib
    leaf_disease_model/best_leaf_disease_model.pth
    satellite_weather_model/satellite_weather_model.pth
    farm_advisory/ (recommendation engine + weather feed)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cpu'

SENSOR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'sensor_model', 'rf_model.joblib')
LEAF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'leaf_disease_model', 'best_leaf_disease_model.pth')
SATELLITE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, 'satellite_weather_model', 'satellite_weather_model.pth'
)

SENSOR_FEATURES = [
    'Temperature', 'Humidity', 'Moisture', 'Nitrogen',
    'Phosphorus', 'Potassium', 'PH', 'Light_Intensity',
]

# Field guidance shown in HTML forms (matches sensor_model/test.py).
SENSOR_FIELD_GUIDE = [
    {'name': 'Temperature', 'label': 'Temperature', 'unit': '°C',
     'low': 15.0, 'high': 35.0, 'step': 0.1, 'default': 26.0},
    {'name': 'Humidity', 'label': 'Humidity', 'unit': '%',
     'low': 30.0, 'high': 95.0, 'step': 0.1, 'default': 62.0},
    {'name': 'Moisture', 'label': 'Soil Moisture', 'unit': '%',
     'low': 10.0, 'high': 80.0, 'step': 0.1, 'default': 48.0},
    {'name': 'Nitrogen', 'label': 'Nitrogen (N)', 'unit': 'mg/kg',
     'low': 10.0, 'high': 100.0, 'step': 0.1, 'default': 48.0},
    {'name': 'Phosphorus', 'label': 'Phosphorus (P)', 'unit': 'mg/kg',
     'low': 10.0, 'high': 80.0, 'step': 0.1, 'default': 42.0},
    {'name': 'Potassium', 'label': 'Potassium (K)', 'unit': 'mg/kg',
     'low': 10.0, 'high': 80.0, 'step': 0.1, 'default': 40.0},
    {'name': 'PH', 'label': 'Soil pH', 'unit': 'pH',
     'low': 4.5, 'high': 8.5, 'step': 0.05, 'default': 6.6},
    {'name': 'Light_Intensity', 'label': 'Light Intensity', 'unit': 'lux',
     'low': 200.0, 'high': 1000.0, 'step': 1.0, 'default': 620.0},
]


def _ensure_on_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Sensor model (disease classification + yield regressor)
# ---------------------------------------------------------------------------
_sensor_bundle = None


def _get_sensor_bundle():
    global _sensor_bundle
    if _sensor_bundle is None:
        import joblib
        if not os.path.exists(SENSOR_MODEL_PATH):
            raise FileNotFoundError(
                f'Sensor model not found: {SENSOR_MODEL_PATH}. '
                'Run sensor_model/train.py first.'
            )
        _sensor_bundle = joblib.load(SENSOR_MODEL_PATH)
    return _sensor_bundle


def sensor_predict(values):
    """Classify disease and forecast yield from an 8-feature reading."""
    bundle = _get_sensor_bundle()
    classifier = bundle.get('classifier', bundle.get('model'))
    regressor = bundle.get('regressor')
    scaler = bundle['scaler']
    label_encoder = bundle['label_encoder']

    import pandas as pd
    frame = pd.DataFrame([[float(values[f]) for f in SENSOR_FEATURES]],
                         columns=SENSOR_FEATURES)
    scaled = scaler.transform(frame)

    predicted = label_encoder.inverse_transform(classifier.predict(scaled))[0]
    probabilities = {
        name: float(prob)
        for name, prob in zip(label_encoder.classes_, classifier.predict_proba(scaled)[0])
    }
    yield_forecast = None
    if regressor is not None:
        yield_forecast = round(float(regressor.predict(scaled)[0]), 2)

    return {
        'prediction': predicted,
        'yield_forecast': yield_forecast,
        'probabilities': probabilities,
        'values': {f: float(values[f]) for f in SENSOR_FEATURES},
    }


# ---------------------------------------------------------------------------
# Leaf disease model (EfficientNet-B0 CNN)
# ---------------------------------------------------------------------------
_leaf_meta = None  # (model, class_names, transform)


def _get_leaf_model():
    global _leaf_meta
    if _leaf_meta is None:
        import torch
        from torchvision import models, transforms as T

        if not os.path.exists(LEAF_MODEL_PATH):
            raise FileNotFoundError(
                f'Leaf model not found: {LEAF_MODEL_PATH}. '
                'Run leaf_disease_model/train.py first.'
            )
        checkpoint = torch.load(LEAF_MODEL_PATH, map_location=DEVICE, weights_only=False)
        class_names = checkpoint['class_names']

        _ensure_on_path(os.path.dirname(LEAF_MODEL_PATH))
        from train import create_model
        model = create_model(len(class_names), model_name='efficientnet_b0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _leaf_meta = (model, class_names, transform)
    return _leaf_meta


def leaf_diagnose(image_path):
    """Diagnose a leaf from an image file -> {prediction, probability, topk}."""
    import numpy as np
    import torch
    from PIL import Image

    model, class_names, transform = _get_leaf_model()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    order = np.argsort(probs)[::-1]
    return {
        'prediction': class_names[int(order[0])],
        'probability': float(probs[order[0]]),
        'topk': [{'class': class_names[int(i)], 'probability': float(probs[i])}
                 for i in order[:5]],
        'all_probabilities': {cls: float(p) for cls, p in zip(class_names, probs)},
    }


# ---------------------------------------------------------------------------
# Satellite / weather-state model (image CNN)
# ---------------------------------------------------------------------------
_satellite_meta = None  # (model, class_names, transform)


def _get_satellite_model():
    global _satellite_meta
    if _satellite_meta is None:
        import torch

        if not os.path.exists(SATELLITE_MODEL_PATH):
            raise FileNotFoundError(
                f'Satellite model not found: {SATELLITE_MODEL_PATH}. '
                'Run satellite_weather_model/main.py first.'
            )
        checkpoint = torch.load(SATELLITE_MODEL_PATH, map_location=DEVICE, weights_only=True)
        class_names = checkpoint['class_names']

        _ensure_on_path(os.path.join(PROJECT_ROOT, 'satellite_weather_model'))
        from main import Net, EVAL_TRANSFORM
        model = Net(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        _satellite_meta = (model, class_names, EVAL_TRANSFORM)
    return _satellite_meta


def satellite_diagnose(image_path):
    """Classify the weather state of an image -> {prediction, probability, topk}."""
    import numpy as np
    import torch
    from PIL import Image

    model, class_names, transform = _get_satellite_model()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    order = np.argsort(probs)[::-1]
    return {
        'prediction': class_names[int(order[0])],
        'probability': float(probs[order[0]]),
        'topk': [{'class': class_names[int(i)], 'probability': float(probs[i])}
                 for i in order[:4]],
        'all_probabilities': {cls: float(p) for cls, p in zip(class_names, probs)},
    }


# ---------------------------------------------------------------------------
# Farm advisory layer (recommendation engine + weather feed)
# ---------------------------------------------------------------------------
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
        from recommendation_engine import RecommendationEngine
        _engine = RecommendationEngine(verbose=False)
    return _engine


def build_weather_feed(latitude, longitude):
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    from weather_feed import LiveWeatherFeed
    return LiveWeatherFeed(latitude, longitude)


_report_delivery = None


def get_report_delivery():
    """Shared ReportDelivery writing into farm_advisory/outputs/."""
    global _report_delivery
    if _report_delivery is None:
        _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
        from delivery import ReportDelivery
        _report_delivery = ReportDelivery()
    return _report_delivery


def get_report_dir():
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    import config
    return config.REPORT_DIR


def get_device_registry():
    """Shared ESP32 registry (latest reading per device + history)."""
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    import device_ingest
    return device_ingest.registry()


def model_status():
    """Report which trained artifacts are present (used by tests/CLI)."""
    return {
        'sensor_model': os.path.exists(SENSOR_MODEL_PATH),
        'leaf_model': os.path.exists(LEAF_MODEL_PATH),
        'satellite_model': os.path.exists(SATELLITE_MODEL_PATH),
    }


def read_telemetry(limit=40):
    """Most recent rows from the farm sensor log, oldest first."""
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    import csv
    from config import SENSOR_LOG_PATH

    if not os.path.exists(SENSOR_LOG_PATH):
        return []

    rows = []
    with open(SENSOR_LOG_PATH, 'r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({f: float(row.get(f)) for f in SENSOR_FEATURES})
            except (TypeError, ValueError):
                continue
    return rows[-limit:]


def live_reading(stress_profile='normal'):
    """A fresh simulated IoT reading; stable within a 30-second window."""
    import time as _time
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    from iot_ingestion import SimulatedSensorNode

    node = SimulatedSensorNode('WEB-LIVE', seed=int(_time.time() // 30),
                               stress_profile=stress_profile)
    return node.read()


def recent_reports(limit=6):
    """Most recently written advisory reports, newest first."""
    report_dir = get_report_dir()
    if not os.path.isdir(report_dir):
        return []
    entries = []
    for name in os.listdir(report_dir):
        if not name.endswith(('.md', '.json')):
            continue
        full = os.path.join(report_dir, name)
        try:
            entries.append({
                'name': name,
                'size_kb': round(os.path.getsize(full) / 1024, 1),
                'modified': os.path.getmtime(full),
            })
        except OSError:
            continue
    entries.sort(key=lambda item: item['modified'], reverse=True)
    return entries[:limit]


def get_farm_defaults():
    _ensure_on_path(os.path.join(PROJECT_ROOT, 'farm_advisory'))
    import config
    return {
        'latitude': config.FARM_LATITUDE,
        'longitude': config.FARM_LONGITUDE,
        'crop': config.CROP_NAME,
    }