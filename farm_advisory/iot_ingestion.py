"""Live IoT sensor ingestion module.

The agronomy platform has two ways to obtain soil/plant telemetry:

* ``SensorIngestion`` - a validation + persistence pipeline. Readings are
  range-checked, lightly clamped and appended to ``dataset/sensor_log.csv``
  exactly like the feature schema the Random Forest was trained on.
* ``SimulatedSensorNode`` - a realistic virtual IoT node that reproduces the
  observed sensor distributions (soil pH, moisture, N/P/K, temperature,
  humidity, light) with a daily cycle, random-walk drift and occasional
  stress episodes so the whole advisory pipeline can run end-to-end without
  physical hardware. It can be swapped for a real MQTT/HTTP node later.

A single reading uses the standard 8-feature schema:

    Temperature, Humidity, Moisture, Nitrogen,
    Phosphorus, Potassium, PH, Light_Intensity
"""

import os
import csv
import time
import math
import datetime as dt

import numpy as np

from config import SENSOR_LOG_PATH, SENSOR_FEATURES, CROP_NAME

# Physical bounds used when synthesizing realistic telemetry.
FEATURE_RANGES = {
    'Temperature': (15.0, 35.0),
    'Humidity': (30.0, 95.0),
    'Moisture': (10.0, 80.0),
    'PH': (4.5, 8.5),
    'Nitrogen': (10.0, 100.0),
    'Phosphorus': (10.0, 80.0),
    'Potassium': (10.0, 80.0),
    'Light_Intensity': (200.0, 1000.0),
}

# Baseline "healthy farm" values (observations lean toward these).
BASE_VALUES = {
    'Temperature': 26.0,
    'Humidity': 62.0,
    'Moisture': 48.0,
    'PH': 6.6,
    'Nitrogen': 48.0,
    'Phosphorus': 42.0,
    'Potassium': 40.0,
    'Light_Intensity': 620.0,
}


def _clip(value, feature):
    low, high = FEATURE_RANGES[feature]
    return float(min(max(value, low), high))


class SimulatedSensorNode:
    """Virtual IoT node emitting one telemetry reading at a time."""

    def __init__(self, node_id, base_values=None, seed=None, stress_profile='normal'):
        self.node_id = node_id
        self.rng = np.random.default_rng(seed if seed is not None else hash(node_id) % 2**32)
        self.base = dict(base_values or BASE_VALUES)
        self.drift = {f: 0.0 for f in SENSOR_FEATURES}
        self.t0 = time.time()
        self.stress_profile = stress_profile
        # A slow moving random walk gives the farm a gentle seasonal change.
        for f in SENSOR_FEATURES:
            self.drift[f] = self.rng.normal(0, 0.8)

    def read(self, when=None):
        """Produce a full 8-feature reading (in the sensor_model schema)."""
        elapsed_h = (time.time() - self.t0) / 3600.0
        when = when or dt.datetime.now()

        # Daily sinusoidal cycle for temperature & light.
        diurnal = math.sin(2 * math.pi * (elapsed_h - 6.0) / 24.0)

        reading = {}
        for f in SENSOR_FEATURES:
            if f == 'Temperature':
                value = self.base[f] + 4.0 * diurnal
            elif f == 'Light_Intensity':
                value = self.base[f] + 260.0 * diurnal
            elif f == 'Humidity':
                value = self.base[f] - 15.0 * diurnal
            else:
                value = self.base[f]
            # Random-walk drift + measurement noise (jitter like training data).
            value += self.drift[f]
            value += self.rng.normal(0, 0.02 * (FEATURE_RANGES[f][1] - FEATURE_RANGES[f][0]))
            reading[f] = _clip(value, f)

        self._apply_stress(reading)
        reading['PH'] = round(reading['PH'], 2)
        reading['Light_Intensity'] = round(reading['Light_Intensity'], 1)
        for f in ('Temperature', 'Humidity', 'Moisture', 'Nitrogen',
                  'Phosphorus', 'Potassium'):
            reading[f] = round(reading[f], 2)

        return {
            'timestamp': when.isoformat(timespec='seconds'),
            'node_id': self.node_id,
            'crop': CROP_NAME,
            **reading,
        }

    def _apply_stress(self, reading):
        """Occasionally push features into stress so advisories are meaningful."""
        roll = self.rng.random()
        profile = self.stress_profile
        if profile == 'moisture-stress' and roll < 0.18:
            reading['Moisture'] = _clip(reading['Moisture'] - 20.0, 'Moisture')
        elif profile == 'nutrient-stress' and roll < 0.15:
            reading['Nitrogen'] = _clip(reading['Nitrogen'] - 22.0, 'Nitrogen')
            reading['Potassium'] = _clip(reading['Potassium'] - 12.0, 'Potassium')
        elif profile == 'humidity-stress' and roll < 0.16:
            reading['Humidity'] = _clip(reading['Humidity'] + 18.0, 'Humidity')
            reading['Temperature'] = _clip(reading['Temperature'] + 2.0, 'Temperature')
        elif profile == 'rootrot' and roll < 0.12:
            reading['Moisture'] = _clip(reading['Moisture'] + 22.0, 'Moisture')
            reading['PH'] = _clip(reading['PH'] - 1.2, 'PH')


class SensorIngestion:
    """Validates, normalizes and persists sensor readings to a CSV log."""

    def __init__(self, log_path=SENSOR_LOG_PATH):
        self.log_path = log_path
        self.last_reading = None
        self._count = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=['timestamp', 'node_id', 'crop'] + SENSOR_FEATURES,
                )
                writer.writeheader()

    def ingest_reading(self, reading):
        """Validate a reading and append it to the log; return the stored row."""
        missing = [f for f in SENSOR_FEATURES if f not in reading]
        if missing:
            raise ValueError(f'Reading missing required sensor features: {missing}')
        row = {
            'timestamp': reading.get('timestamp') or dt.datetime.now().isoformat(timespec='seconds'),
            'node_id': reading.get('node_id', 'unknown'),
            'crop': reading.get('crop', CROP_NAME),
        }
        for f in SENSOR_FEATURES:
            try:
                value = float(reading[f])
            except (TypeError, ValueError):
                raise ValueError(f'Non-numeric value for feature {f}: {reading[f]!r}')
            row[f] = round(_clip(value, f), 2)

        with open(self.log_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)

        self._count += 1
        self.last_reading = row
        return row

    def simulate_stream(self, node, duration_minutes, interval_seconds=20):
        """Stream readings from a node over a duration; blocks and prints."""
        steps = int(duration_minutes * 60 / interval_seconds)
        for step in range(steps):
            reading = node.read()
            stored = self.ingest_reading(reading)
            print(f"[{stored['timestamp']}] node={stored['node_id']} "
                  f"T={stored['Temperature']:>5.2f} H={stored['Humidity']:>5.1f} "
                  f"M={stored['Moisture']:>5.1f} pH={stored['PH']:>4.2f} "
                  f"N={stored['Nitrogen']:>5.1f} P={stored['Phosphorus']:>5.1f} "
                  f"K={stored['Potassium']:>5.1f} L={stored['Light_Intensity']:>6.1f}")
            if step < steps - 1:
                time.sleep(interval_seconds)
        return self._count


if __name__ == '__main__':
    node = SimulatedSensorNode('SIM-01', seed=7, stress_profile='normal')
    pipeline = SensorIngestion()
    pipeline.simulate_stream(node, duration_minutes=0.1, interval_seconds=2)
    print('Streamed', pipeline._count, 'readings ->', pipeline.log_path)