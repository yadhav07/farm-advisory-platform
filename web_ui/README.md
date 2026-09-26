# Web UI (Flask)

Four pages, one job each, behind a fixed sidebar. Every trained subsystem is
visible from the same dashboard.

The dashboard reports **real hardware only**. There is no simulated node, so
with nothing reporting the overview still renders in full - the metric row, the
pie chart, the reading log and the trends - with empty values rather than a
placeholder screen. The page **polls for a node** and reloads itself the moment
one reports, so it never needs a manual refresh. The advisory page works
without a node, starting from the documented field-guide defaults.

## Vision pages

Each page is a two-card row: the upload form on the left, the result on the
right. After a classification a third card appears with the class distribution
as a donut. The result card and the distribution legend show the same
percentages, so there are deliberately no extra top-k bars.

| Page | Model | Result label |
| :--- | :--- | :--- |
| `/vision/leaf` | EfficientNet-B0 | Diagnosis |
| `/vision/sky` | Weather CNN | Weather state |

## What an empty overview looks like

The layout is always the real one, so the design can be reviewed with or without
hardware attached:

| Panel | No reading |
| :--- | :--- |
| Crop state | `No data` |
| Yield forecast, top risk, soil moisture | an em dash, with the caption saying why |
| Disease split | the card keeps its header, the body says the classifier has nothing to score |
| Reading log | the table renders with a `No readings logged yet.` row |
| Trends | the four panels render with a `No telemetry logged yet.` note |

The sidebar node card and the top bar pill both read `No node reporting`, so it
is always clear that the empty values mean "no hardware", not "no data model".

## Run

```bash
cd web_ui
pip install -r requirements.txt
python app.py
```

Or from the advisory layer (the project's unified entry point):

```bash
cd farm_advisory
python app.py          # launches the web UI and opens the browser
python app.py --cli    # instead runs the live CLI advisory pipeline
```

Then open <http://127.0.0.1:5001>. The server binds to `0.0.0.0` so field nodes
on the same network can reach it.

## Pages

| Route          | What it does                                                          |
|----------------|-----------------------------------------------------------------------|
| `/`            | Overview: live metrics, the disease pie chart, the reading log, trends |
| `/vision/leaf` | Leaf Vision: classify a leaf photo as a disease                        |
| `/vision/sky`  | Sky Vision: classify a sky photo as a weather state                     |
| `/advisory`    | Run the full pipeline and get ranked actions plus a report            |

`/vision` redirects to `/vision/leaf`, so older links still land somewhere sensible.
An unknown model segment (`/vision/nonsense`) returns 404.

Navigation is the sidebar: brand, four items, and a node-status card that shows
the real device as online, idle, or not reporting. The sticky top bar carries
the breadcrumb and the same status as a pill.

Leaf Vision and Sky Vision share one Flask endpoint (`/vision/<kind>`) and one
template. Everything model-specific - title, heading, blurb, model name, icon,
colour, empty-state copy - lives in the `VISION_MODELS` dict in `app.py`, so
`vision.html` has no conditionals and adding a third image model is a dict
entry plus a sidebar row.

Because both tabs share the `vision` endpoint, the sidebar compares
`request.view_args['kind']` as well as `request.endpoint`; otherwise both tabs
would light up at the same time.

## Overview page

The dashboard answers three questions and nothing else:

1. **What state is the field in?** — crop state, yield forecast, top disease risk
2. **Why?** — the disease pie chart from the Random Forest classifier
3. **How is it changing?** — the last 10 readings as a table, plus trend sparklines

The metric row sits above an equal-width two-column row (pie chart and reading
log), with the trends card full width below.

Weather is not on the dashboard; it is folded into the advisory run, where the
live forecast actually changes a recommendation.

## Field-node API

The ESP32 firmware in `../hardware/esp32_farm_node/` pushes readings here:

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/sensor-data` | POST | ingest a firmware payload; returns crop state and yield forecast |
| `/api/devices` | GET | known nodes with IP, last reading and online state |
| `/api/latest` | GET | latest reading mapped to the 8-feature model schema |

`/api/latest` answers `{"source": "none"}` until the first reading arrives, so a
client can tell "no data yet" from "a node is reporting".

Payload keys: `device_id`, `soil_moisture`, `air_temperature`, `humidity`,
`light_intensity`, `mq135_raw`, `bme_temperature`, `pressure`. Optional `ph`,
`nitrogen`, `phosphorus` and `potassium` replace the estimated values.

There is no connection page. `DeviceRegistry.probe(ip)` in
`../farm_advisory/device_ingest.py` still probes a node directly
(`GET http://<ip>/api/sensor-data`, which the firmware serves) and is available
to call from the backend; the UI does not expose it.

Post a reading by hand to see the dashboard fill in:

```bash
curl -X POST http://127.0.0.1:5001/api/sensor-data \
  -H 'Content-Type: application/json' \
  -d '{"device_id":"FARM_01","soil_moisture":44,"air_temperature":27.3,
       "humidity":64.8,"light_intensity":585.2}'
```

## How the charts are drawn

Every visual is rendered by the **browser** as HTML, CSS or inline SVG. There is
no server-side image generation, no matplotlib and no JavaScript charting
library, so the dashboard ships no chart assets and has no chart CDN dependency.

`viz.py` contains pure geometry helpers (coordinates, percentages, CSS gradient
strings) registered as Jinja globals. `_charts.html` turns them into macros:

| Macro         | Technique                                    | Visual                                     |
|---------------|-----------------------------------------------|--------------------------------------------|
| `donut`       | CSS `conic-gradient` with a masked centre     | disease / confidence / priority mix         |
| `radar`       | inline SVG polygons, rings and axes          | sensor profile against ideal range          |
| `rangebars`   | HTML/CSS bars over a track                   | reading position inside each range          |
| `sparkgrid`   | inline SVG polylines                         | telemetry trends                            |
| `suggestions` | HTML cards                                   | ranked advisory actions                     |

Typography and icons come from Manrope and Font Awesome via CDN; the layout
degrades cleanly if they fail to load.

## Layout

```
web_ui/
├── app.py            Flask routes + field-node API
├── viz.py            chart geometry helpers (no image output)
├── model_service.py  lazy-loaded, cached inference helpers
├── templates/        base, index, vision, advisory + _charts
├── static/style.css  workspace design system
├── requirements.txt
└── uploads/          runtime folder for image uploads (auto-cleaned)
```

## Alignment contract

The stylesheet documents the five rules that keep every page aligned, in a
comment at the top of `static/style.css`:

1. `.app-shell` is a two-column grid. The sidebar is its own sticky column and
   the content column is `minmax(0, 1fr)`, so the sidebar can never push the
   cards out of alignment. Inside the content column, `.topbar-inner` and
   `.content-wrap` share one `--shell-width` and one `--shell-pad`, so the top
   bar and every card start at the same x. Below 980px the sidebar becomes a
   horizontal bar and is deliberately *not* sticky, so the top bar keeps
   `top: 0` and there is no double-sticky offset to get wrong.
2. `.view` is a single-column flex with one `--gutter` gap, so every card row on
   every page is separated by the same space and no template needs an inline
   margin for it.
3. Cards inside a grid row stretch to the tallest card, and `.panel-body` /
   `.table-wrap` claim the leftover height, so headers, charts and tables line up.
4. `.panel-header` has one height (`--header-h`) everywhere, so card content
   always starts on the same baseline.
5. `.metric-card p` uses `margin-top: auto`, so the captions sit on one baseline
   across the metric row, and `.metric-value` has a single size (`.metric-value-text`
   is the smaller step used when the value is a word rather than a number).

## Notes

* All deep models run on CPU in the web layer and are cached after the first
  call, so subsequent requests are fast.
* Uploaded images are removed after inference to keep the folder tidy.
* Advisory reports are written to `farm_advisory/outputs/` and offered as
  downloadable `.md` and `.json` files.
* Use a WSGI server in production (e.g. `waitress`, `gunicorn`) - the built-in
  server is for development only.
