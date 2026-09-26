"""Farm advisory platform entry point.

``python app.py``  ->  Flask web UI (default)
``python app.py --cli``  ->  end-to-end live CLI pipeline

Web UI (default)
----------------
    python app.py                    # start browser UI at http://127.0.0.1:5001
    python app.py --port 8080        # custom port
    python app.py --no-browser       # do not auto-open the browser

CLI pipeline (--cli, or any pipeline option passed)
---------------------------------------------------
    python app.py --cli                      # 1 reading, live weather
    python app.py --steps 5 --interval 3     # 5 readings, 3s apart
    python app.py --leaf "C:/leaf.jpg"       # include leaf disease image
    python app.py --sky  "C:/sky.jpg"        # include weather/sky image
    python app.py --offline                  # skip the live weather API call
"""

import os
import sys
import json
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import FARM_LATITUDE, FARM_LONGITUDE, SENSOR_LOG_PATH, REPORT_DIR
from weather_feed import LiveWeatherFeed
from iot_ingestion import SimulatedSensorNode, SensorIngestion
from recommendation_engine import RecommendationEngine
from delivery import ReportDelivery

DEFAULT_STEPS = 1
DEFAULT_INTERVAL = 5.0

# Values matching "no options passed" - used to auto-detect the CLI pipeline.
PIPELINE_DEFAULTS = {
    'lat': FARM_LATITUDE,
    'lon': FARM_LONGITUDE,
    'steps': DEFAULT_STEPS,
    'interval': DEFAULT_INTERVAL,
    'node_id': 'SIM-01',
    'seed': 7,
    'stress': 'normal',
    'leaf': None,
    'sky': None,
    'offline': False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Farm advisory platform (web UI by default, --cli for pipeline)',
    )
    # Mode switches -----------------------------------------------------
    parser.add_argument('--cli', action='store_true',
                        help='Run the end-to-end live CLI pipeline instead of the web UI')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port for the web UI (default: 5001)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open the web UI in the browser')
    # Pipeline options ---------------------------------------------------
    parser.add_argument('--lat', type=float, default=FARM_LATITUDE, help='Farm latitude')
    parser.add_argument('--lon', type=float, default=FARM_LONGITUDE, help='Farm longitude')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS,
                        help='Number of telemetry readings to ingest/run')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help='Seconds between readings (simulates live streaming)')
    parser.add_argument('--node-id', default='SIM-01', help='Sensor node identifier')
    parser.add_argument('--seed', type=int, default=7, help='Seed for the simulated node')
    parser.add_argument('--stress', default='normal',
                        choices=['normal', 'moisture-stress', 'nutrient-stress',
                                 'humidity-stress', 'rootrot'],
                        help='Stress profile used by the simulated node')
    parser.add_argument('--leaf', default=None, help='Optional leaf image path')
    parser.add_argument('--sky', default=None, help='Optional sky/weather image path')
    parser.add_argument('--offline', action='store_true',
                        help='Skip live weather API call (use cache/fallback)')
    return parser.parse_args(argv)


def _pipeline_requested(args):
    """True when any pipeline option differs from its default value."""
    if args.cli:
        return True
    for name, default in PIPELINE_DEFAULTS.items():
        if getattr(args, name) != default:
            return True
    return False


def lan_ip():
    """Best-effort LAN address of this machine, for the ESP32 SERVER_URL."""
    import socket
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.3)
        # No packet is sent; this just asks the OS which local interface
        # would be used to reach the outside world.
        probe.connect(('8.8.8.8', 80))
        address = probe.getsockname()[0]
        probe.close()
        return address
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def launch_web_ui(port=5001, open_browser=True, host=None):
    """Start the Flask web UI (``web_ui/app.py``) as the default mode.

    ``host`` defaults to the ``HOST`` environment variable and then to
    ``0.0.0.0`` so an ESP32 on the same network can reach the ingest API.
    ``PORT`` is honoured too, so the same entry point works unchanged behind a
    PaaS that assigns the port.
    """
    import importlib.util
    import threading
    import webbrowser

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(project_root, 'web_ui')
    web_app_path = os.path.join(web_dir, 'app.py')
    if not os.path.exists(web_app_path):
        raise SystemExit(f'[app] web UI not found: {web_app_path}')

    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)

    try:
        port = int(os.environ.get('PORT', port))
    except (TypeError, ValueError):
        pass
    host = host or os.environ.get('HOST', '0.0.0.0')

    # Import web_ui/app.py under a distinct name so its `__main__` block
    # never runs - we start the server ourselves below.
    spec = importlib.util.spec_from_file_location('farm_advisory_webapp', web_app_path)
    web_module = importlib.util.module_from_spec(spec)
    sys.modules['farm_advisory_webapp'] = web_module
    spec.loader.exec_module(web_module)

    local_url = f'http://127.0.0.1:{port}'
    lan_url = f'http://{lan_ip()}:{port}' if host == '0.0.0.0' else local_url

    print('=' * 72)
    print(' FARM ADVISORY WEB UI '.center(72, '='))
    print('=' * 72)
    print(f'[app] local    {local_url}')
    if lan_url != local_url:
        print(f'[app] network  {lan_url}   <- point the ESP32 SERVER_URL here')
    print('[app] health   ' + f'{lan_url}/healthz')
    print('[app] ingest   ' + f'{lan_url}/api/sensor-data')
    print('[app] press CTRL+C to stop the server')

    if open_browser:
        threading.Timer(1.2, webbrowser.open, args=(local_url,)).start()

    web_module.app.run(host=host, port=port, debug=False)
    return None


def run(args):
    print('=' * 72)
    print(' FARM ADVISORY PIPELINE '.center(72, '='))
    print('=' * 72)

    # 1. Components -------------------------------------------------------
    ingestion = SensorIngestion(log_path=SENSOR_LOG_PATH)
    node = SimulatedSensorNode(args.node_id, seed=args.seed, stress_profile=args.stress)
    engine = RecommendationEngine(verbose=True)
    delivery = ReportDelivery(report_dir=REPORT_DIR)

    if args.offline:
        weather = LiveWeatherFeed(args.lat, args.lon).fetch_current()
        weather['source'] = 'offline/cached'
    else:
        feed = LiveWeatherFeed(args.lat, args.lon)
        print(f'[app] fetching live weather for {args.lat}, {args.lon} ...')
        weather = feed.fetch_current(force=True)
        print(f'[app] weather: {weather["weather_label"]} {weather["temperature"]}C '
              f'{weather["humidity"]}% ({weather["source"]})')

    delivered = []

    # 2. Stream telemetry + advise ---------------------------------------
    import time
    for step in range(max(1, args.steps)):
        reading = node.read()
        stored = ingestion.ingest_reading(reading)
        print(f'[app] ingested reading {step + 1}/{args.steps} from {stored["node_id"]}')

        analysis = engine.analyze(
            stored,
            weather=weather,
            leaf_image=args.leaf,
            sky_image=args.sky,
        )
        log = delivery.deliver(analysis)
        delivered.append(log)

        if step < args.steps - 1:
            time.sleep(max(0.5, args.interval))

    print(f'\n[app] completed {len(delivered)} advisory cycle(s)')
    print(f'[app] sensor log   -> {SENSOR_LOG_PATH}')
    print(f'[app] report dir   -> {REPORT_DIR}')
    return delivered


def main(argv=None):
    args = parse_args(argv)
    if _pipeline_requested(args):
        return run(args)
    return launch_web_ui(port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()