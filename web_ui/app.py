"""Flask web UI for the Multi-disciplinary Farm Advisory Platform.

A chart-driven dashboard over every trained subsystem:

* Overview    - live telemetry, model output, KPI tiles and the full pipeline map
* Sensor      - disease diagnosis + yield forecast with donut / gauge / radar charts
* Leaf        - leaf disease detection from an uploaded photo
* Satellite   - weather-state classification from a sky/field photo
* Weather     - live conditions + 3-day forecast chart (Open-Meteo)
* Advisory    - full pipeline: sensor + optional photos + live weather ->
                prioritized recommendations + downloadable reports

Charts are rendered server-side with matplotlib (Agg) and served as PNG from
``/chart/<kind>``; the pages themselves are dependency-free HTML/CSS.

Run:  python app.py            then open http://127.0.0.1:5001
"""

import os
import sys
import shutil
import datetime as dt
import uuid
import threading
from collections import OrderedDict

# Windows consoles default to cp1252 which cannot encode the emoji used below.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import (
    Flask, render_template, request, abort, send_from_directory, Response,
)
from werkzeug.utils import secure_filename

import charts
import model_service
from model_service import SENSOR_FIELD_GUIDE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Tiny in-memory result store: pages POST a payload once, then the <img> tags
# for each chart fetch it back by id. Keeps model inference off the chart path.
# ---------------------------------------------------------------------------
RESULT_STORE = OrderedDict()
_STORE_LOCK = threading.Lock()
_STORE_LIMIT = 80


def store_result(payload):
    """Persist a chart payload and return its short id."""
    result_id = uuid.uuid4().hex[:16]
    with _STORE_LOCK:
        RESULT_STORE[result_id] = payload
        RESULT_STORE.move_to_end(result_id)
        while len(RESULT_STORE) > _STORE_LIMIT:
            RESULT_STORE.popitem(last=False)
    return result_id


def load_result(result_id):
    with _STORE_LOCK:
        return RESULT_STORE.get(result_id) or {}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage):
    """Persist an uploaded image to a unique name under uploads/."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    original = secure_filename(file_storage.filename or 'image.jpg')
    ext = original.rsplit('.', 1)[1].lower() if '.' in original else 'jpg'
    name = f'{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.{ext}'
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
    return f'⚠️ {msg}'


def pct(value):
    try:
        return f'{float(value) * 100:.1f}%'
    except (TypeError, ValueError):
        return '—'


def build_chart(kind, payload):
    """Dispatch a chart request to the right charts.py function."""
    payload = payload or {}
    if kind == 'disease':
        return charts.disease_donut(payload.get('probabilities') or {})
    if kind == 'yield':
        return charts.yield_gauge(payload.get('yield_forecast'))
    if kind == 'radar':
        return charts.sensor_radar(payload.get('values') or {}, SENSOR_FIELD_GUIDE)
    if kind == 'bars':
        return charts.sensor_bars(payload.get('values') or {}, SENSOR_FIELD_GUIDE)
    if kind == 'vision':
        return charts.vision_donut(payload.get('vision_items') or [],
                                    payload.get('vision_title') or
                                    'Classification confidence')
    if kind == 'priority':
        return charts.priority_donut(payload.get('recommendations') or [])
    if kind == 'forecast':
        return charts.forecast(payload.get('forecast_rows') or [])
    raise KeyError(kind)


# ---------------------------------------------------------------------------
# Chart delivery
# ---------------------------------------------------------------------------
@app.route('/chart/<kind>')
def chart(kind):
    """Render a dashboard chart as a PNG (result data comes from the store)."""
    result_id = request.args.get('id')
    payload = load_result(result_id) if result_id else {}

    try:
        if kind == 'telemetry':
            png = charts.render(charts.telemetry, model_service.read_telemetry(40))
        else:
            png = charts.render(build_chart, kind, payload)
    except KeyError:
        abort(404)
    except Exception as exc:  # never take the page down over one chart
        app.logger.warning('chart %s failed: %s', kind, exc)
        abort(500)

    return Response(png, mimetype='image/png',
                    headers={'Cache-Control': 'no-store'})


# ---------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Live telemetry + model output dashboard."""
    error = None
    result = None
    reading = None
    weather = None
    result_id = None
    rows = []

    try:
        reading = model_service.live_reading()
        values = {f: float(reading[f]) for f in model_service.SENSOR_FEATURES}
        result = model_service.sensor_predict(values)

        feed = model_service.build_weather_feed(
            model_service.get_farm_defaults()['latitude'],
            model_service.get_farm_defaults()['longitude'],
        )
        weather = feed.fetch_current()
        rows = build_forecast_rows(weather)

        risk = max(
            ((k, v) for k, v in result['probabilities'].items() if k != 'Healthy'),
            key=lambda kv: kv[1], default=(None, 0.0),
        )
        result_id = store_result({
            'probabilities': result['probabilities'],
            'yield_forecast': result['yield_forecast'],
            'values': values,
            'forecast_rows': rows,
        })
    except Exception as exc:
        error = format_error(str(exc))

    return render_template(
        'index.html', result=result, reading=reading, weather=weather,
        result_id=result_id, error=error, fields=SENSOR_FIELD_GUIDE,
        values=values if reading else sensor_values_for_defaults(),
        top_risk=risk if reading else (None, 0.0),
    )


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


def _at(sequence, index):
    if not sequence or index >= len(sequence):
        return None
    return sequence[index]


# ---------------------------------------------------------------------------
# Sensor analytics
# ---------------------------------------------------------------------------
@app.route('/sensor', methods=['GET', 'POST'])
def sensor():
    error = None
    result = None
    result_id = None
    is_live = request.method == 'GET'

    if request.method == 'GET':
        try:
            reading = model_service.live_reading()
            values = {f: float(reading[f]) for f in model_service.SENSOR_FEATURES}
        except Exception as exc:
            error = format_error(str(exc))
            values = sensor_values_for_defaults()
    else:
        values, error = parse_sensor_form(request.form)

    if error is None:
        try:
            result = model_service.sensor_predict(values)
            result_id = store_result({
                'probabilities': result['probabilities'],
                'yield_forecast': result['yield_forecast'],
                'values': values,
            })
        except Exception as exc:
            error = format_error(str(exc))

    risk = None
    if result:
        risk = max(
            ((k, v) for k, v in result['probabilities'].items() if k != 'Healthy'),
            key=lambda kv: kv[1], default=(None, 0.0),
        )

    return render_template('sensor.html', fields=SENSOR_FIELD_GUIDE, values=values,
                           result=result, result_id=result_id, error=error,
                           is_live=is_live, risk=risk)


# ---------------------------------------------------------------------------
# Leaf disease detection
# ---------------------------------------------------------------------------
@app.route('/leaf', methods=['GET', 'POST'])
def leaf():
    result = None
    error = None
    result_id = None

    if request.method == 'POST':
        upload = request.files.get('image')
        if upload is None or upload.filename == '':
            error = format_error('Please choose a leaf image to upload.')
        elif not allowed_file(upload.filename):
            error = format_error('Unsupported image type. Use PNG/JPG/JPEG/WebP/BMP.')
        else:
            path = save_upload(upload)
            try:
                result = model_service.leaf_diagnose(path)
                result_id = store_result({
                    'vision_items': [(i['class'], i['probability'])
                                     for i in result['topk']],
                    'vision_title': 'Leaf disease confidence',
                })
            except Exception as exc:
                error = format_error(str(exc))
            finally:
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)  # keep the tree tidy

    return render_template('leaf.html', result=result, result_id=result_id, error=error)


# ---------------------------------------------------------------------------
# Satellite / weather-state classification
# ---------------------------------------------------------------------------
@app.route('/satellite', methods=['GET', 'POST'])
def satellite():
    result = None
    error = None
    result_id = None

    if request.method == 'POST':
        upload = request.files.get('image')
        if upload is None or upload.filename == '':
            error = format_error('Please choose a sky / field image to upload.')
        elif not allowed_file(upload.filename):
            error = format_error('Unsupported image type. Use PNG/JPG/JPEG/WebP/BMP.')
        else:
            path = save_upload(upload)
            try:
                result = model_service.satellite_diagnose(path)
                result_id = store_result({
                    'vision_items': [(i['class'], i['probability'])
                                     for i in result['topk']],
                    'vision_title': 'Weather-state confidence',
                })
            except Exception as exc:
                error = format_error(str(exc))
            finally:
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

    return render_template('satellite.html', result=result, result_id=result_id,
                           error=error)


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
    result_id = None
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
        result_id = store_result({'forecast_rows': rows})
    except ValueError:
        error = format_error('Please enter valid decimal latitude/longitude.')
    except Exception as exc:
        error = format_error(str(exc))

    return render_template('weather.html', snapshot=snapshot, forecast_rows=rows,
                           result_id=result_id, error=error,
                           latitude=latitude, longitude=longitude)


# ---------------------------------------------------------------------------
# Full advisory pipeline
# ---------------------------------------------------------------------------
@app.route('/advisory', methods=['GET', 'POST'])
def advisory():
    result = None
    error = None
    report_links = None
    result_id = None
    values = sensor_values_for_defaults()
    use_weather = False
    latitude = None
    longitude = None
    is_live = request.method == 'GET'

    if request.method == 'GET':
        try:
            reading = model_service.live_reading()
            values = {f: float(reading[f]) for f in model_service.SENSOR_FEATURES}
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
                result_id = store_result({
                    'probabilities': analysis['sensor'].get('disease_probabilities') or {},
                    'yield_forecast': analysis['sensor'].get('yield_forecast'),
                    'values': values,
                    'recommendations': analysis.get('recommendations') or [],
                    'forecast_rows': build_forecast_rows(weather),
                })
            finally:
                for path in uploads:
                    if os.path.exists(path):
                        os.remove(path)
        except Exception as exc:
            error = format_error(str(exc))

    return render_template('advisory.html', fields=SENSOR_FIELD_GUIDE, values=values,
                           result=result, error=error, report_links=report_links,
                           result_id=result_id, use_weather=use_weather, is_live=is_live,
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
    app.run(host='127.0.0.1', port=5001, debug=False)
