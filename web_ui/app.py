"""Flask web UI for the Multi-disciplinary Farm Advisory Platform.

Exposes every trained subsystem as a small web tool:

* Dashboard    - overview of the available tools
* Sensor       - disease diagnosis + yield forecast from an 8-feature reading
* Leaf         - leaf disease detection from an uploaded photo
* Satellite    - weather-state classification from a sky/field photo
* Weather      - live conditions + 3-day forecast (Open-Meteo)
* Advisory     - full pipeline: sensor + optional photos + live weather ->
                 prioritized recommendations + downloadable reports

Run:  python web_ui/app.py            then open http://127.0.0.1:5001
"""

import os
import sys
import shutil
import datetime as dt
import uuid

# Windows consoles default to cp1252 which cannot encode the emoji used below.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import (
    Flask, render_template, request, abort, send_from_directory,
)
from werkzeug.utils import secure_filename

import model_service
from model_service import SENSOR_FIELD_GUIDE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app = Flask(__name__)


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


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Sensor disease diagnosis + yield forecast
# ---------------------------------------------------------------------------
@app.route('/sensor', methods=['GET', 'POST'])
def sensor():
    result = None
    error = None
    values = sensor_values_for_defaults()

    if request.method == 'POST':
        values, error = parse_sensor_form(request.form)
        if error is None:
            try:
                result = model_service.sensor_predict(values)
            except Exception as exc:
                error = format_error(str(exc))

    return render_template('sensor.html', fields=SENSOR_FIELD_GUIDE,
                           values=values, result=result, error=error)


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
            error = format_error('Unsupported image type. Use PNG/JPG/JPEG/WebP/BMP.')
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
            error = format_error('Please choose a sky / field image to upload.')
        elif not allowed_file(upload.filename):
            error = format_error('Unsupported image type. Use PNG/JPG/JPEG/WebP/BMP.')
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
    feed_defaults = model_service.get_farm_defaults()
    latitude = feed_defaults['latitude']
    longitude = feed_defaults['longitude']
    snapshot = None
    error = None

    if request.method == 'POST':
        try:
            latitude = float(request.form.get('latitude', feed_defaults['latitude']))
            longitude = float(request.form.get('longitude', feed_defaults['longitude']))
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                raise ValueError('coordinates out of range')
            feed = model_service.build_weather_feed(latitude, longitude)
            snapshot = feed.fetch_current(force=True)
        except ValueError:
            error = format_error('Please enter valid decimal latitude/longitude.')
        except Exception as exc:
            error = format_error(str(exc))
    else:
        try:
            feed = model_service.build_weather_feed(latitude, longitude)
            snapshot = feed.fetch_current()
        except Exception as exc:
            error = format_error(str(exc))

    forecast_rows = []
    if snapshot and snapshot.get('forecast'):
        f = snapshot['forecast']
        dates = f.get('dates', [])
        for i, day in enumerate(dates):
            forecast_rows.append({
                'date': day,
                't_min': f['t_min'][i] if i < len(f.get('t_min', [])) else None,
                't_max': f['t_max'][i] if i < len(f.get('t_max', [])) else None,
                'precip_prob': (f['precip_probability'][i]
                                if i < len(f.get('precip_probability', [])) else None),
                'precip_sum': (f['precip_sum'][i]
                               if i < len(f.get('precip_sum', [])) else None),
            })

    return render_template('weather.html', snapshot=snapshot,
                           forecast_rows=forecast_rows, error=error,
                           latitude=latitude, longitude=longitude)


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

    if request.method == 'POST':
        values, error = parse_sensor_form(request.form)
        use_weather = (request.form.get('use_weather') == 'on')

        if error is None:
            try:
                sensor = {**values, 'node_id': 'WEB', 'crop': 'Wheat',
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

                    engine = model_service.get_engine()
                    analysis = engine.analyze(sensor, weather=weather,
                                              leaf_image=leaf_path, sky_image=sky_path)

                    # Persist the advisory reports and expose download links.
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

    return render_template('advisory.html', fields=SENSOR_FIELD_GUIDE,
                           values=values, result=result, error=error,
                           report_links=report_links, use_weather=use_weather,
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