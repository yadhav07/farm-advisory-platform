"""Deploy the FarmIQ workspace to Render via its REST API.

Usage (PowerShell, from the repo root):

    $env:RENDER_API_KEY = "rnd_..."
    python deploy_render.py

The key is read from the environment and never written to disk. Get one at
https://render.com/docs/api (Dashboard -> API Keys -> Create API Key).

What it does:
  1. finds the GitHub repo connection
  2. creates (or reuses) a web service from render.yaml
  3. triggers a deploy and polls the build log until it finishes
  4. prints the public URL and waits for /healthz to answer
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API = 'https://api.render.com/v1'
REPO = 'https://github.com/yadhav07/farm-advisory-platform'
NAME = 'farm-advisory-platform'
KEY = os.environ.get('RENDER_API_KEY', '').strip()


def call(method, path, body=None, timeout=90):
    """One authenticated Render API call."""
    url = f'{API}{path}'
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {KEY}')
    req.add_header('Accept', 'application/json')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode('utf-8', 'replace')
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', 'replace')
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {'raw': body[:400]}


def step(msg):
    print(f'\n=== {msg} ===')


def main():
    if not KEY:
        print('RENDER_API_KEY is not set.')
        print('Create a key at https://render.com/docs/api, then:')
        print('  $env:RENDER_API_KEY = "rnd_..."')
        print('  python deploy_render.py')
        return 2

    step('authenticating')
    status, owner = call('GET', '/owners')
    if status >= 400:
        print(f'  auth failed (HTTP {status}): {owner}')
        return 1
    print(f'  ok, authenticated as {owner.get("email") or owner.get("username")}')
    owner_id = owner.get('id') or owner.get('owner', {}).get('id')

    step('finding the GitHub repo connection')
    status, conns = call('GET', f'/owners/{owner_id}/repo-connections')
    if status >= 400:
        print(f'  could not list repo connections (HTTP {status}): {conns}')
        print('  In the Render dashboard, connect the GitHub repo once, then retry.')
        return 1

    wanted = REPO.lower().rstrip('.git')
    match = None
    for c in (conns if isinstance(conns, list) else []):
        if wanted in json.dumps(c).lower():
            match = c
            break
    if match is None:
        print('  no repo connection matches', REPO)
        print('  Connect it in the Render dashboard first:')
        print('    Dashboard -> Repos -> Connect -> pick yadhav07/farm-advisory-platform')
        return 1
    print(f'  found: {match.get("repo")}')

    step('looking for an existing service')
    status, services = call('GET', f'/services?limit=100')
    existing = None
    for s in (services if isinstance(services, list) else []):
        if s.get('name') == NAME:
            existing = s
            break

    if existing:
        svc_id = existing['service']['id']
        print(f'  service already exists: {existing["service"]["name"]} ({svc_id})')
        print('  reusing it and triggering a fresh deploy')
    else:
        step('creating the web service from render.yaml')
        status, created = call('POST', '/services', {
            'name': NAME,
            'repo': match.get('repo'),
            'branch': 'main',
            'type': 'web',
            'runtime': 'python',
            'plan': 'starter',
            'healthCheckPath': '/healthz',
            'autoDeploy': True,
            'envVars': [
                {'key': 'PYTHON_VERSION', 'value': '3.12'},
                {'key': 'THREADS', 'value': '4'},
            ],
        })
        if status >= 400:
            print(f'  create failed (HTTP {status}): {created}')
            return 1
        svc = created.get('service', created)
        svc_id = svc.get('id')
        print(f'  created: {svc.get("name")} ({svc_id})')
        print('  NOTE: render.yaml settings are applied on the Render side;')
        print('        confirm the build/start commands in the dashboard.')

    step('triggering a deploy')
    status, deploy = call('POST', f'/services/{svc_id}/deploys', {
        'clearCache': 'do_not_clear',
    })
    if status >= 400:
        print(f'  deploy trigger failed (HTTP {status}): {deploy}')
        return 1
    deploy_id = (deploy.get('deploy') or {}).get('id')
    print(f'  deploy id: {deploy_id}')

    step('watching the build (this pulls torch, so give it a few minutes)')
    final = None
    for _ in range(180):
        time.sleep(10)
        status, d = call('GET', f'/services/{svc_id}/deploys/{deploy_id}')
        info = d.get('deploy', d)
        state = info.get('status')
        print(f'  {state}', end='\r')
        if state in ('live', 'build_failed', 'update_failed', 'canceled', 'pre_deploy_failed'):
            final = state
            break
    print()
    if final != 'live':
        print(f'  build ended as: {final}')
        print('  Read the log in the dashboard: Dashboard -> your service -> Events')
        return 1

    step('result')
    status, svc = call('GET', f'/services/{svc_id}')
    details = svc.get('service', svc)
    url = details.get('url') or f'https://{NAME}.onrender.com'
    print(f'  live at: {url}')
    print(f'  health:  {url}/healthz')

    step('waiting for the health check to answer')
    for _ in range(30):
        try:
            with urllib.request.urlopen(f'{url}/healthz', timeout=30) as r:
                print(f'  {url}/healthz -> {r.status}')
                print(' ', r.read().decode('utf-8', 'replace')[:200])
                return 0
        except Exception as exc:
            print(f'  waiting... ({type(exc).__name__})')
            time.sleep(10)
    print('  health check did not answer yet; the first build may still be')
    print('  pulling the model artifacts. Check the URL above in a minute.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
