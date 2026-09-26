"""WSGI entry point for production servers.

Every PaaS looks for a module that exposes a WSGI callable, conventionally
named ``application``::

    gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 wsgi:application
    waitress-serve --listen=0.0.0.0:$PORT wsgi:application

``HOST`` and ``PORT`` environment variables are honoured so the same command
works locally, on Render, Railway, Fly or Heroku without editing anything.

The heavy model artifacts (EfficientNet-B0, the weather CNN and the Random
Forest bundle) total about 130 MB and are loaded lazily on first use, so the
process starts cheaply and only the page you open pays for its model. Keep the
worker count at 1 and raise threads instead: a second worker would load a second
copy of every model into memory.
"""

import os
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_UI_DIR = os.path.join(BASE_DIR, 'web_ui')
FARM_ADVISORY_DIR = os.path.join(BASE_DIR, 'farm_advisory')

# The web layer resolves farm-advisory modules (config, device_ingest,
# recommendation_engine) by path, and imports its own siblings (viz,
# model_service) as top-level modules. Both directories go on the path; the
# web_ui one comes first so a sibling `app.py` never shadows it.
for _path in (FARM_ADVISORY_DIR, WEB_UI_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Both web_ui/app.py and farm_advisory/app.py are called `app`, so a plain
# `from app import app` would import whichever one sys.path finds first. Load
# the web app from its file under a unique name instead.
_MODULE_NAME = 'farm_advisory_webapp'
if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, os.path.join(WEB_UI_DIR, 'app.py'))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _module
    _spec.loader.exec_module(_module)

#: The WSGI callable every PaaS and application server looks for.
application = _module.app

#: Aliases some hosts probe for.
app_instance = application
server = application


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main():
    """Run the production server directly: ``python wsgi.py``."""
    host = os.environ.get('HOST', '0.0.0.0')
    port = _int_env('PORT', 5001)
    threads = _int_env('THREADS', 8)

    from waitress import serve
    print(f'[wsgi] serving on http://{host}:{port} with {threads} threads')
    print('[wsgi] press CTRL+C to stop')
    serve(application, host=host, port=port, threads=threads)


if __name__ == '__main__':
    main()
