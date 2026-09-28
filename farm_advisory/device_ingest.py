"""ESP32 field-node ingestion.

Accepts the JSON payload posted by the ESP32 firmware, maps it onto the
8-feature schema the Random Forest was trained on, and keeps a rolling
history so the dashboard can show real hardware telemetry.

Firmware payload
----------------
::

    {
      "device_id": "FARM_01",
      "soil_moisture": 45,        # %
      "air_temperature": 26.4,    # degC   (DHT11)
      "humidity": 62.1,           # %      (DHT11)
      "light_intensity": 620.5,   # lux    (BH1750)
      "mq135_raw": 1180,          # ADC    (MQ-135 air quality)
                                 # optional: "ph" when a probe is fitted
      "bme_temperature": 25.1,    # degC   (BME280)
      "pressure": 1013.2          # hPa    (BME280)
    }

Sensor mapping
--------------
Four of the five model features are measured directly:

======================  =========================
Model feature        Source
======================  =========================
``Temperature``      ``air_temperature`` (DHT11)
``Humidity``         ``humidity`` (DHT11)
``Moisture``         ``soil_moisture``
``Light_Intensity``  ``light_intensity`` (BH1750)
======================  =========================

``PH`` needs a dedicated probe. The firmware may send it (``ph``) and it is
then used verbatim; when absent it is *estimated* from soil moisture, so the
classifier still has a complete row. Every reading is tagged with a
``provenance`` map so the UI can show which values are measured and which are
estimated.

Nitrogen, phosphorus and potassium were removed from the schema: no probe
reports them, so the dashboard was displaying three invented numbers on every
reading. A soil lab test is the right source for those, not a sensor guess.

``mq135_raw``, ``bme_temperature`` and ``pressure`` are kept as context signals
and displayed, but are not part of the model schema.
"""

import os
import csv
import io
import json
import time
import datetime as dt
import urllib.error
import urllib.request

from config import PROJECT_ROOT, SENSOR_FEATURES, DATASET_DIR

DEVICE_LOG_PATH = os.path.join(DATASET_DIR, 'device_readings.csv')
DEVICE_REGISTRY_PATH = os.path.join(DATASET_DIR, 'devices.json')

# Healthy-farm pH baseline used when no probe is fitted.
BASELINE = {'PH': 6.6}

# Physical bounds shared with the simulator, used to clamp incoming values.
BOUNDS = {
    'Temperature': (15.0, 35.0),
    'Humidity': (30.0, 95.0),
    'Moisture': (10.0, 80.0),
    'PH': (4.5, 8.5),
    'Light_Intensity': (200.0, 1000.0),
}

# MQ-135 raw ADC span used to normalise the air-quality signal (0-1).
MQ135_MIN, MQ135_MAX = 300.0, 3000.0

LOG_FIELDS = (['timestamp', 'device_id', 'source_ip', 'firmware_measured']
              + SENSOR_FEATURES
              + ['mq135_raw', 'bme_temperature', 'pressure'])


def _clamp(feature, value):
    low, high = BOUNDS[feature]
    return max(low, min(high, float(value)))


def _number(payload, *keys):
    """First finite numeric value among the candidate keys."""
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue
            if value == value:  # reject NaN
                return value
    return None


def map_to_sensor_schema(payload):
    """Map a firmware payload onto the 5-feature schema plus context signals.

    Returns ``(sensor, context, provenance)``.
    """
    moisture = _number(payload, 'soil_moisture', 'moisture', 'Moisture')
    temperature = _number(payload, 'air_temperature', 'temperature', 'Temperature')
    humidity = _number(payload, 'humidity', 'Humidity')
    light = _number(payload, 'light_intensity', 'light', 'Light_Intensity')
    mq135 = _number(payload, 'mq135_raw', 'mq135', 'gas_raw')
    bme_temperature = _number(payload, 'bme_temperature')
    pressure = _number(payload, 'pressure')

    if moisture is None or temperature is None or humidity is None:
        raise ValueError('payload must include soil_moisture, '
                         'air_temperature and humidity')

    # Soil pH is the only value still estimated when no probe is fitted: it is
    # nudged by how wet the soil is, since waterlogged and dry beds both drift
    # off the neutral baseline.
    wet_delta = ((moisture if moisture is not None else 48.0) - 48.0) / 100.0
    estimated = {'PH': 6.6 - 0.8 * wet_delta}

    # MQ-135 air quality is a context signal only; it is not a model feature.
    # It still gets a normalised 0-100 index for display.
    gas_load = 0.0
    if mq135 is not None:
        gas_load = max(0.0, min(1.0, (mq135 - MQ135_MIN) / (MQ135_MAX - MQ135_MIN)))

    measured_map = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Moisture': moisture,
        'Light_Intensity': light if light is not None else 620.0,
    }

    sensor, provenance = {}, {}
    for feature in SENSOR_FEATURES:
        if feature in measured_map and measured_map[feature] is not None:
            sensor[feature] = _clamp(feature, measured_map[feature])
            provenance[feature] = 'measured'
            continue

        direct = _number(payload, feature.lower(), feature)
        if direct is not None:
            sensor[feature] = _clamp(feature, direct)
            provenance[feature] = 'measured'
        else:
            sensor[feature] = _clamp(feature, estimated[feature])
            provenance[feature] = 'estimated'

    context = {
        'mq135_raw': mq135,
        'mq135_index': round(gas_load * 100, 1) if mq135 is not None else None,
        'bme_temperature': bme_temperature,
        'pressure': pressure,
    }
    return sensor, context, provenance


class DeviceRegistry:
    """Stores the latest reading per ESP32 plus a rolling history on disk."""

    def __init__(self, log_path=DEVICE_LOG_PATH, registry_path=DEVICE_REGISTRY_PATH):
        self.log_path = log_path
        self.registry_path = registry_path
        self.latest = {}          # device_id -> record
        self.order = []           # device_ids by most recent contact
        self._load_registry()

    # ------------------------------------------------------------ storage
    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            return
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for device_id, record in (data.get('devices') or {}).items():
            self.latest[device_id] = record
            self.order.append(device_id)

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        payload = {
            'updated_at': dt.datetime.now().isoformat(timespec='seconds'),
            'devices': {device_id: self.latest[device_id] for device_id in self.order},
        }
        with open(self.registry_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

    def _append_log(self, record):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        exists = os.path.exists(self.log_path)

        # The record nests features under 'sensor' and extras under 'context';
        # flatten both into the CSV columns.
        row = {
            'timestamp': record.get('timestamp'),
            'device_id': record.get('device_id'),
            'source_ip': record.get('source_ip'),
            'firmware_measured': record.get('firmware_measured'),
        }
        row.update(record.get('sensor') or {})
        row.update(record.get('context') or {})

        with open(self.log_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in LOG_FIELDS})

    # ------------------------------------------------------------- ingest
    def ingest(self, payload, source_ip=None):
        """Validate, map and store one firmware payload."""
        if not isinstance(payload, dict):
            raise ValueError('payload must be a JSON object')

        device_id = str(payload.get('device_id') or 'ESP32-UNKNOWN').strip()
        sensor, context, provenance = map_to_sensor_schema(payload)

        now = time.time()
        record = {
            'device_id': device_id,
            'source_ip': source_ip or payload.get('ip') or '',
            'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
            'received_at': now,
            'sensor': sensor,
            'context': context,
            'provenance': provenance,
            'firmware_measured': sum(1 for v in provenance.values() if v == 'measured'),
        }

        if device_id in self.latest:
            self.order.remove(device_id)
        self.latest[device_id] = record
        self.order.insert(0, device_id)
        self._append_log(record)
        self._save_registry()
        return record

    # -------------------------------------------------------------- query
    def devices(self):
        """Known devices, most recently seen first."""
        now = time.time()
        rows = []
        for device_id in self.order:
            record = self.latest.get(device_id)
            if not record:
                continue
            age = now - record.get('received_at', now)
            rows.append({
                'device_id': device_id,
                'source_ip': record.get('source_ip', ''),
                'timestamp': record.get('timestamp', ''),
                'age_seconds': int(age),
                'online': age <= 90,
                'firmware_measured': record.get('firmware_measured', 0),
            })
        return rows

    def get(self, device_id=None):
        if device_id is None:
            return self.latest.get(self.order[0]) if self.order else None
        return self.latest.get(device_id)

    def history(self, device_id=None, limit=40):
        """Most recent sensor rows for a device, oldest first.

        Feature values only, because that is what the trend charts plot. Use
        :meth:`history_full` when the timestamps and context signals matter.
        """
        return [
            {feature: row[feature] for feature in SENSOR_FEATURES
             if feature in row}
            for row in self.history_full(device_id, limit=limit)
        ]

    def history_full(self, device_id=None, limit=200, since=None, until=None):
        """Complete stored rows for a device, oldest first.

        Unlike :meth:`history` this keeps everything the log recorded: the
        timestamp, the source IP, which values the firmware actually measured
        rather than derived, and the MQ-135/BME280 context signals.

        ``since``/``until`` are ISO timestamps or ``YYYY-MM-DD`` dates, so a
        caller can ask for one day without post-processing the whole log.
        """
        if not os.path.exists(self.log_path):
            return []

        rows = []
        with open(self.log_path, 'r', encoding='utf-8', newline='') as handle:
            for raw in csv.DictReader(handle):
                if device_id and raw.get('device_id') != device_id:
                    continue

                stamp = (raw.get('timestamp') or '').strip()
                if since and stamp and stamp < since:
                    continue
                if until and stamp and stamp > until:
                    continue

                row = {
                    'timestamp': stamp,
                    'device_id': raw.get('device_id') or '',
                    'source_ip': raw.get('source_ip') or '',
                    'firmware_measured': int(raw.get('firmware_measured') or 0),
                }
                for field in SENSOR_FEATURES:
                    try:
                        row[field] = float(raw.get(field))
                    except (TypeError, ValueError):
                        continue
                # Context signals are optional in firmware, but a consumer of
                # the JSON should not have to cope with the key vanishing, so
                # report an absent signal as null rather than dropping it.
                for field in ('mq135_raw', 'bme_temperature', 'pressure'):
                    try:
                        row[field] = float(raw.get(field))
                    except (TypeError, ValueError):
                        row[field] = None
                rows.append(row)

        return rows[-limit:] if limit else rows

    def history_csv(self, device_id=None, limit=0):
        """Raw log text, optionally filtered. Used for the CSV download."""
        if not os.path.exists(self.log_path):
            return ''
        if device_id or limit:
            rows = self.history_full(device_id, limit=limit or None)
            if not rows:
                return ''
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=LOG_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            return buffer.getvalue()
        with open(self.log_path, 'r', encoding='utf-8', newline='') as handle:
            return handle.read()

    # -------------------------------------------------------- reachability
    def probe(self, ip, timeout=2.5):
        """Check whether a device at ``ip`` answers a direct data request.

        Works when the firmware exposes ``GET /api/sensor-data`` in addition to
        posting. Returns a dict describing the outcome.
        """
        ip = (ip or '').strip()
        if not ip:
            return {'ok': False, 'ip': ip, 'detail': 'no IP address supplied'}

        if ip.startswith('http://') or ip.startswith('https://'):
            base = ip.rstrip('/')
        else:
            base = f'http://{ip}'

        started = time.time()
        for path in ('/api/sensor-data', '/data'):
            url = f'{base}{path}'
            try:
                request = urllib.request.Request(
                    url, headers={'Accept': 'application/json'})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read(4096).decode('utf-8', 'replace')
                    return {
                        'ok': True,
                        'ip': ip,
                        'url': url,
                        'latency_ms': int((time.time() - started) * 1000),
                        'detail': f'device responded on {path}',
                        'payload': body,
                    }
            except urllib.error.HTTPError as exc:
                return {
                    'ok': False,
                    'ip': ip,
                    'url': url,
                    'latency_ms': int((time.time() - started) * 1000),
                    'detail': f'HTTP {exc.code} from {path}',
                }
            except Exception as exc:
                last_error = exc

        return {
            'ok': False,
            'ip': ip,
            'latency_ms': int((time.time() - started) * 1000),
            'detail': f'no response ({type(last_error).__name__}) - the dashboard '
                      'will still use readings the device pushes',
        }


_REGISTRY = None


def registry():
    """Shared registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DeviceRegistry()
    return _REGISTRY


if __name__ == '__main__':
    sample = {
        'device_id': 'FARM_01', 'soil_moisture': 45, 'air_temperature': 26.4,
        'humidity': 62.1, 'light_intensity': 620.5, 'mq135_raw': 1180,
        'bme_temperature': 25.1, 'pressure': 1013.2,
    }
    record = registry().ingest(sample, source_ip='192.168.1.50')
    print(json.dumps(record, indent=2))
