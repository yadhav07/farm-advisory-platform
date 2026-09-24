"""Flask web dashboard for the Farm Advisory Platform.

Serves live field-node telemetry (ESP32) and the trained models:

* ``POST /api/sensor-data``  ingest the JSON posted by the ESP32 firmware
* ``GET  /api/devices``      list known field nodes
* ``GET  /api/latest``       latest reading, mapped to the model schema
* ``/``                      overview dashboard for the live node
* ``/device``                field-node connection panel (IP address input)
* ``/sensor``                disease diagnosis + yield forecast
* ``/leaf`` ``/satellite``   image classification
* ``/weather``               live conditions + forecast
* ``/advisory``              full pipeline with report downloads

Every visual is rendered by the browser as HTML, CSS or inline SVG. Python only
supplies coordinates and percentages through the ``viz`` helpers, so there is no
server-side image generation and no charting library.

Run:  python app.py            then open http://127.0.0.1:5001
"""

import os
import sys
import shutil
import datetime as dt

# Windows consoles default to cp1252; force UTF-8 for the report output.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import (
    Flask, render_template, request, abort, send_from_directory, jsonify,
)
from werkzeug.utils import secure_filename

import viz
import model_service
from model_service import SENSOR_FIELD_GUIDE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app = Flask(__name__)
app.jinja_env.globals.update(viz.VIZ)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _name_suffix():
    import uuid
    return uuid.uuid4().hex[:8]


def save_upload(file_storage):
    """Persist an uploaded image to a unique name under uploads/."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    original = secure_filename(file_storage.filename or 'image.jpg')
    ext = original.rsplit('.', 1)[1].lower() if '.' in original else 'jpg'
    name = f'{dt.datetime.now():%Y%m%d_%H%M%S}_{_name_suffix()}.{ext}'
    path = os.path.join(UPLOAD_DIR, name)
    file_storage.save(path)
    return path


def parse_sensor_form(form):
    """Pull the eight telemetry features from a form; returns (values, error)."""
    values = {}
    for field in model_service.SENSOR_FEATURES:
        raw = (form.get(field) or '').strip()
        if not raw:
            return None, f'Missing reading for "{field}".'
        try:
            values[field] = float(raw)
        except ValueError:
            return None, f'"{raw}" is not a valid number for {field}.'
    return values, None


def sensor_values_for_defaults():
    return {f['name']: f['default'] for f in SENSOR_FIELD_GUIDE}


def format_error(msg):
    return f'Error: {msg}'


def top_risk(probabilities):
    """The highest-probability non-Healthy class, as (name, probability)."""
    if not probabilities:
        return (None, 0.0)
    return max(((k, v) for k, v in probabilities.items() if k != 'Healthy'),
               key=lambda kv: kv[1], default=(None, 0.0))


def _at(sequence, index):
    if not sequence or index >= len(sequence):
        return None
    return sequence[index]


def build_forecast_rows(snapshot):
    """Zip the weather feed's parallel forecast lists into row dicts."""
    rows = []
    if not snapshot or not snapshot.get('forecast'):
        return rows
    forecast = snapshot['forecast']
    for i, day in enumerate(forecast.get('dates', [])):
        rows.append({
            'date': day,
            't_min': _at(forecast.get('t_min'), i),
            't_max': _at(forecast.get('t_max'), i),
            'precip_prob': _at(forecast.get('precip_probability'), i),
            'precip_sum': _at(forecast.get('precip_sum'), i),
        })
    return rows


def disease_items(probabilities):
    return [(k.replace('_', ' ').title(), float(v)) for k, v in (probabilities or {}).items()]


def vision_items(topk):
    """(label, probability) pairs for a classifier's top-k list."""
    return [(item['class'].replace('_', ' ').title(), float(item['probability']))
            for item in (topk or [])]


def priority_summary(recommendations):
    """Count recommendations per priority for the priority-mix donut."""
    order = ['High', 'Medium', 'Low']
    counts = {name: 0 for name in order}
    for rec in recommendations or []:
        if rec.get('priority') in counts:
            counts[rec['priority']] += 1
    active = [name for name in order if counts[name] > 0]
    return {
        # key is 'pairs' not 'items': in templates `d.items` would resolve to
        # the dict method rather than this list.
        'pairs': [(name, counts[name]) for name in active],
        'colors': [viz.PRIORITY_COLORS[name] for name in active],
        'total': sum(counts.values()),
        'high': counts['High'],
    }


def resolve_live_reading(device_id=None):
    """Latest field-node reading if one exists, otherwise the simulated node.

    Returns a dict with ``values``, ``source`` and display metadata.
    """
    registry = model_service.get_device_registry()
    record = registry.get(device_id)

    if record:
        values = {k: float(v) for k, v in record['sensor'].items()}
        return {
            'sensor': values,
            'source': 'device',
            'device_id': record.get('device_id', 'ESP32'),
            'source_ip': record.get('source_ip', ''),
            'timestamp': record.get('timestamp', ''),
            'context': record.get('context', {}),
            'provenance': record.get('provenance', {}),
            'online': (dt.datetime.now().timestamp() - record.get('received_at', 0)) <= 90,
        }

    reading = model_service.live_reading()
    values = {f: float(reading[f]) for f in model_service.SENSOR_FEATURES}
    return {
        'sensor': values,
        'source': 'simulated',
        'device_id': reading.get('node_id', 'SIM'),
        'source_ip': '',
        'timestamp': reading.get('timestamp', ''),
        'context': {},
        'provenance': {f: 'measured' for f in model_service.SENSOR_FEATURES},
        'online': False,
    }


def resolve_history(live, limit=40):
    """Telemetry history for the live reading: device log or simulated log."""
    if live['source'] == 'device':
        return model_service.get_device_registry().history(live['device_id'], limit)
    return model_service.read_telemetry(limit)


# Template helpers exposed to Jinja alongside the viz geometry functions.
app.jinja_env.globals['disease_items'] = disease_items
app.jinja_env.globals['vision_items'] = vision_items
app.jinja_env.globals['priority_summary'] = priority_summary


# ---------------------------------------------------------------------------
# Field-node API (consumed by the ESP32 firmware)
# ---------------------------------------------------------------------------
@app.route('/api/sensor-data', methods=['POST'])
def api_sensor_data():
    """Accept a JSON payload from the ESP32 and return the model assessment."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'ok': False, 'error': 'expected a JSON body'}), 400
    try:
        record = model_service.get_device_registry().ingest(
            payload, source_ip=request.remote_addr)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400

    result = model_service.sensor_predict(
        {k: float(v) for k, v in record['sensor'].items()})
    return jsonify({
        'ok': True,
        'device_id': record['device_id'],
        'received_at': record['timestamp'],
        'crop_state': result['prediction'],
        'yield_forecast': result['yield_forecast'],
        'provenance': record['provenance'],
    })


@app.route('/api/devices')
def api_devices():
    registry = model_service.get_device_registry()
    return jsonify({
        'ok': True,
        'devices': registry.devices(),
        'sensor_endpoint': '/api/sensor-data',
    })


@app.route('/api/latest')
def api_latest():
    live = resolve_live_reading(request.args.get('device_id'))
    if live['source'] != 'device':
        return jsonify({'ok': True, 'source': 'simulated', 'sensor': live['sensor']})
    return jsonify({
        'ok': True,
        'source': 'device',
        'device_id': live['device_id'],
        'source_ip': live['source_ip'],
        'timestamp': live['timestamp'],
        'online': live['online'],
        'sensor': live['sensor'],
        'context': live['context'],
        'provenance': live['provenance'],
    })


# ---------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    error = None
    result = None
    live = None
    weather = None
    rows = []
    risk = (None, 0.0)
    values = sensor_values_for_defaults()

    try:
        live = resolve_live_reading()
        values = live['sensor']
        result = model_service.sensor_predict(values)
        risk = top_risk(result['probabilities'])

        defaults = model_service.get_farm_defaults()
        weather = model_service.build_weather_feed(
            defaults['latitude'], defaults['longitude']).fetch_current()
        rows = build_forecast_rows(weather)
    except Exception as exc:
        error = format_error(str(exc))

    return render_template(
        'index.html', result=result, live=live, weather=weather,
        forecast_rows=rows, telemetry=resolve_history(live or {'source': 'simulated',
                                                               'device_id': None}),
        fields=SENSOR_FIELD_GUIDE, values=values, risk=risk, error=error,
    )


# ---------------------------------------------------------------------------
# Field-node connection panel
# ---------------------------------------------------------------------------
@app.route('/device', methods=['GET', 'POST'])
def device():
    registry = model_service.get_device_registry()
    devices = registry.devices()
    probe = None
    error = None
    selected = (request.values.get('device_id') or
                (devices[0]['device_id'] if devices else ''))

    if request.method == 'POST':
        ip = (request.form.get('ip') or '').strip()
        if not ip:
            error = 'Enter the IP address of the field node.'
        else:
            probe = registry.probe(ip)
            if not probe.get('ok') and probe.get('payload'):
                error = None
            selected = (request.form.get('device_id') or selected).strip()

    live = resolve_live_reading(selected or None)
    result = None
    if live['source'] == 'device' or request.values.get('demo'):
        try:
            result = model_service.sensor_predict(live['sensor'])
        except Exception as exc:
            error = format_error(str(exc))

    return render_template('device.html', devices=devices, probe=probe, error=error,
                           live=live, result=result, selected=selected,
                           fields=SENSOR_FIELD_GUIDE, values=live['sensor'])


# ---------------------------------------------------------------------------
# Sensor analytics
# ---------------------------------------------------------------------------
@app.route('/sensor', methods=['GET', 'POST'])
def sensor():
    error = None
    result = None
    risk = (None, 0.0)
    is_live = request.method == 'GET'

    if is_live:
        try:
            live = resolve_live_reading(request.args.get('device_id'))
            values = live['sensor']
        except Exception as exc:
            error = format_error(str(exc))
            values = sensor_values_for_defaults()
    else:
        values, error = parse_sensor_form(request.form)

    if error is None:
        try:
            result = model_service.sensor_predict(values)
            risk = top_risk(result['probabilities'])
        except Exception as exc:
            error = format_error(str(exc))

    return render_template('sensor.html', fields=SENSOR_FIELD_GUIDE, values=values,
                           result=result, error=error, is_live=is_live, risk=risk)


# ---------------------------------------------------------------------------
# Leaf disease detection
# ---------------------------------------------------------------------------
@app.route('/leaf', methods=['GET', 'POST'])
def leaf():
    result = None
    error = None

    if request.method == 'POST':
        upload = request.files.get('image')
        if upload is None or upload.filename == '':
            error = format_error('Please choose a leaf image to upload.')
        elif not allowed_file(upload.filename):
            error = format_error('Unsupported image type. Use PNG, JPG, JPEG, WebP or BMP.')
        else:
            path = save_upload(upload)
            try:
                result = model_service.leaf_diagnose(path)
            except Exception as exc:
                error = format_error(str(exc))
            finally:
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)  # keep the tree tidy

    return render_template('leaf.html', result=result, error=error)


# ---------------------------------------------------------------------------
# Satellite / weather-state classification
# ---------------------------------------------------------------------------
@app.route('/satellite', methods=['GET', 'POST'])
def satellite():
    result = None
    error = None

    if request.method == 'POST':
        upload = request.files.get('image')
        if upload is None or upload.filename == '':
            error = format_error('Please choose a sky or field image to upload.')
        elif not allowed_file(upload.filename):
            error = format_error('Unsupported image type. Use PNG, JPG, JPEG, WebP or BMP.')
        else:
            path = save_upload(upload)
            try:
                result = model_service.satellite_diagnose(path)
            except Exception as exc:
                error = format_error(str(exc))
            finally:
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

    return render_template('satellite.html', result=result, error=error)


# ---------------------------------------------------------------------------
# Live weather feed
# ---------------------------------------------------------------------------
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    defaults = model_service.get_farm_defaults()
    latitude = defaults['latitude']
    longitude = defaults['longitude']
    snapshot = None
    rows = []
    error = None

    try:
        if request.method == 'POST':
            latitude = float(request.form.get('latitude', defaults['latitude']))
            longitude = float(request.form.get('longitude', defaults['longitude']))
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                raise ValueError('coordinates out of range')
            snapshot = model_service.build_weather_feed(
                latitude, longitude).fetch_current(force=True)
        else:
            snapshot = model_service.build_weather_feed(
                latitude, longitude).fetch_current()
        rows = build_forecast_rows(snapshot)
    except ValueError:
        error = format_error('Please enter valid decimal latitude and longitude.')
    except Exception as exc:
        error = format_error(str(exc))

    return render_template('weather.html', snapshot=snapshot, forecast_rows=rows,
                           error=error, latitude=latitude, longitude=longitude)


# ---------------------------------------------------------------------------
# Full advisory pipeline
# ---------------------------------------------------------------------------
@app.route('/advisory', methods=['GET', 'POST'])
def advisory():
    result = None
    error = None
    report_links = None
    values = sensor_values_for_defaults()
    use_weather = False
    latitude = None
    longitude = None
    is_live = request.method == 'GET'

    if is_live:
        try:
            values = resolve_live_reading(request.args.get('device_id'))['values']
        except Exception as exc:
            error = format_error(str(exc))
    else:
        values, error = parse_sensor_form(request.form)
        use_weather = (request.form.get('use_weather') == 'on')

    if error is None and request.method == 'POST':
        try:
            sensor_row = {**values, 'node_id': 'WEB', 'crop': 'Wheat',
                          'timestamp': dt.datetime.now().isoformat(timespec='seconds')}

            weather = None
            if use_weather:
                try:
                    lat = float(request.form.get('latitude', 0) or 0)
                    lon = float(request.form.get('longitude', 0) or 0)
                except ValueError:
                    lat = lon = None
                if not lat or not lon:
                    defaults = model_service.get_farm_defaults()
                    lat, lon = defaults['latitude'], defaults['longitude']
                latitude, longitude = lat, lon
                weather = model_service.build_weather_feed(lat, lon).fetch_current(force=True)

            leaf_path = sky_path = None
            uploads = []
            try:
                leaf_upload = request.files.get('leaf_image')
                sky_upload = request.files.get('sky_image')
                if leaf_upload and leaf_upload.filename and allowed_file(leaf_upload.filename):
                    leaf_path = save_upload(leaf_upload)
                    uploads.append(leaf_path)
                if sky_upload and sky_upload.filename and allowed_file(sky_upload.filename):
                    sky_path = save_upload(sky_upload)
                    uploads.append(sky_path)

                analysis = model_service.get_engine().analyze(
                    sensor_row, weather=weather,
                    leaf_image=leaf_path, sky_image=sky_path,
                )

                saved = model_service.get_report_delivery().deliver_file(analysis)
                report_links = {
                    'markdown': os.path.basename(saved['markdown']),
                    'json': os.path.basename(saved['json']),
                }
                result = analysis
            finally:
                for path in uploads:
                    if os.path.exists(path):
                        os.remove(path)
        except Exception as exc:
            error = format_error(str(exc))

    return render_template('advisory.html', fields=SENSOR_FIELD_GUIDE, values=values,
                           result=result, error=error, report_links=report_links,
                           use_weather=use_weather, is_live=is_live,
                           latitude=latitude, longitude=longitude)


@app.route('/advisory/report/<filename>')
def advisory_report(filename):
    """Serve an advisory report that was written to farm_advisory/outputs/."""
    report_dir = model_service.get_report_dir()
    safe = secure_filename(filename)
    if safe != filename or not os.path.isfile(os.path.join(report_dir, safe)):
        abort(404)
    return send_from_directory(report_dir, safe)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
