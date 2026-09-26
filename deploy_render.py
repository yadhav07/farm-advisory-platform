"""Deploy the FarmIQ workspace to Render via its REST API.

Usage (PowerShell, from the repo root):

    $env:RENDER_API_KEY = "rnd_..."
    python deploy_render.py

The key is read from the environment and never written to disk. Get one at
<https://render.com/docs/api>.

What it does:
  1. authenticates and resolves the workspace
  2. reuses the service if it exists, otherwise creates it
  3. triggers a deploy and polls until the build finishes
  4. prints the public URL and waits for /healthz to answer

API shape notes, confirmed against the live API rather than the docs, because
the error text is misleading in places:

  * ``GET /owners`` returns ``[{"cursor": ..., "owner": {...}}]``, so the
    workspace id is nested one level down.
  * ``type`` must be ``web_service``; ``web`` is rejected.
  * ``runtime``, ``plan``, ``buildCommand``, ``startCommand`` and ``envVars``
    must be nested inside a ``serviceDetails`` object, otherwise the API
    answers "must include serviceDetails when creating a non-static service".
  * several malformed shapes return a bare "invalid JSON", which does not
    mean the JSON was malformed.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API = 'https://api.render.com/v1'
REPO = 'github.com/yadhav07/farm-advisory-platform'
NAME = 'farm-advisory-platform'
KEY = os.environ.get('RENDER_API_KEY', '').strip()

BUILD = ('pip install --index-url https://download.pytorch.org/whl/cpu '
         'torch==2.13.0+cpu torchvision==0.28.0+cpu '
         '&& pip install -r requirements.txt')
START = ('gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 '
         '--timeout 120 wsgi:application')


def call(method, path, body=None, timeout=90):
    """One authenticated Render API call."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f'{API}{path}', data=data, method=method)
    req.add_header('Authorization', f'Bearer {KEY}')
    req.add_header('Accept', 'application/json')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode('utf-8', 'replace')
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode('utf-8', 'replace')
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {'raw': raw[:400]}


def step(msg):
    print(f'\n=== {msg} ===')


def workspace():
    status, payload = call('GET', '/owners')
    if status >= 400:
        raise RuntimeError(f'auth failed (HTTP {status}): {payload}')
    for entry in (payload if isinstance(payload, list) else []):
        inner = entry.get('owner', entry)
        if inner.get('id'):
            return inner
    raise RuntimeError(f'no usable workspace in {payload}')


def find_service():
    status, payload = call('GET', '/services?limit=100')
    if status >= 400 or not isinstance(payload, list):
        return None
    for item in payload:
        svc = item.get('service', item)
        if svc.get('name') == NAME:
            return svc
    return None


def main():
    if not KEY:
        print('RENDER_API_KEY is not set.')
        print('Create a key at https://render.com/docs/api, then:')
        print('  $env:RENDER_API_KEY = "rnd_..."')
        print('  python deploy_render.py')
        return 2

    step('authenticating')
    try:
        owner = workspace()
    except RuntimeError as exc:
        print(f'  {exc}')
        return 1
    print(f'  workspace "{owner.get("name")}" / {owner.get("email")}')

    step('checking for an existing service')
    existing = find_service()
    if existing:
        svc_id = existing['id']
        print(f'  reusing {NAME} ({svc_id}) {existing.get("url")}')
    else:
        step('creating the web service')
        status, created = call('POST', '/services', {
            'ownerID': owner['id'],
            'name': NAME,
            'repo': REPO,
            'branch': 'main',
            'type': 'web_service',
            'serviceDetails': {
                'runtime': 'python',
                'plan': 'free',
                'buildCommand': BUILD,
                'startCommand': START,
                'envVars': [
                    {'key': 'PYTHON_VERSION', 'value': '3.12'},
                    {'key': 'THREADS', 'value': '4'},
                ],
            },
            'healthCheckPath': '/healthz',
        })
        if status == 429:
            print('  RATE LIMITED by the Render API. Service creation has a tight')
            print('  write quota. Either wait and re-run, or create it from the')
            print('  dashboard, which does not consume the API quota:')
            print('    https://render.com/deploy  -> pick this repo -> Apply')
            return 1
        if status >= 400:
            print(f'  create failed (HTTP {status}): {json.dumps(created)[:600]}')
            return 1
        svc = created.get('service', created)
        svc_id = svc.get('id')
        print(f'  created {svc.get("name")} ({svc_id})')
        print(f'  build: {" ".join(svc.get("buildCommand", "").split())[:95]}')
        print(f'  start: {" ".join(svc.get("startCommand", "").split())[:95]}')

    step('triggering a deploy')
    status, deploy = call('POST', f'/services/{svc_id}/deploys',
                          {'clearCache': 'do_not_clear'})
    if status >= 400:
        print(f'  trigger failed (HTTP {status}): {json.dumps(deploy)[:400]}')
        return 1
    deploy_id = (deploy.get('deploy') or {}).get('id')
    print(f'  deploy id: {deploy_id}')

    step('watching the build (pulling torch, so allow a few minutes)')
    final = None
    for _ in range(180):
        time.sleep(10)
        status, d = call('GET', f'/services/{svc_id}/deploys/{deploy_id}')
        info = d.get('deploy', d)
        state = info.get('status')
        print(f'  {state}', end='\r')
        if state in ('live', 'build_failed', 'update_failed', 'canceled',
                     'pre_deploy_failed'):
            final = state
            break
    print()
    if final != 'live':
        print(f'  build ended as: {final}')
        print('  Read the log at Dashboard -> your service -> Events')
        return 1

    step('result')
    status, svc_wrap = call('GET', f'/services/{svc_id}')
    details = svc_wrap.get('service', svc_wrap)
    url = details.get('url') or f'https://{NAME}.onrender.com'
    print(f'  live at: {url}')
    print(f'  health:  {url}/healthz')

    step('waiting for the health check')
    for _ in range(30):
        try:
            with urllib.request.urlopen(f'{url}/healthz', timeout=30) as r:
                print(f'  {url}/healthz -> {r.status}')
                print(' ', r.read().decode('utf-8', 'replace')[:200])
                return 0
        except Exception:
            time.sleep(10)
    print('  health check has not answered yet; the first build may still be')
    print('  fetching the model artifacts. Try the URL again in a minute.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
