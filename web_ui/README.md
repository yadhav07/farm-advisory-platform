# Web UI (Flask)

Four pages, one job each. Every trained subsystem and the ESP32 field node are
visible from the same dashboard.

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

| Route       | What it does                                                             |
|-------------|--------------------------------------------------------------------------|
| `/`         | Overview: live metrics, the disease pie chart, the reading log, trends, the pipeline and the weather outlook |
| `/device`   | Field node: connect an ESP32 by IP address and see its latest reading      |
| `/vision`   | Two image classifiers: leaf disease and weather state                     |
| `/advisory` | Run the full pipeline and get ranked actions plus a downloadable report    |

## Overview page

The dashboard answers four questions and nothing else:

1. **What state is the field in?** — crop state, yield forecast, top disease risk
2. **Why?** — the disease pie chart from the Random Forest classifier
3. **How is it changing?** — the last 10 readings as a table, plus trend sparklines
4. **What does the weather do?** — a three-day outlook with a location field

The pipeline lanes show how a reading becomes advice, so the flow is visible
without a separate page.

## Field-node API

The ESP32 firmware in `../hardware/esp32_farm_node/` pushes readings here:

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/sensor-data` | POST | ingest a firmware payload; returns crop state and yield forecast |
| `/api/devices` | GET | known nodes with IP, last reading and online state |
| `/api/latest` | GET | latest reading mapped to the 8-feature model schema |

Payload keys: `device_id`, `soil_moisture`, `air_temperature`, `humidity`,
`light_intensity`, `mq135_raw`, `bme_temperature`, `pressure`. Optional `ph`,
`nitrogen`, `phosphorus` and `potassium` replace the estimated values.

The **Field node** page takes the node's IP address and tries
`GET http://<ip>/api/sensor-data`; the firmware serves that route, so a direct
pull works when the node is reachable. If it is not, the dashboard still uses
whatever the node has pushed.

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
| `forecast`    | HTML/CSS positioned columns                  | 3-day temperature range and rain            |
| `flow`        | HTML/CSS nodes with arrow connectors         | signal-flow lanes                           |
| `suggestions` | HTML cards                                   | ranked advisory actions                     |

Typography and icons come from Manrope and Font Awesome via CDN; the layout
degrades cleanly if they fail to load.

## Layout

```
web_ui/
├── app.py            Flask routes + field-node API
├── viz.py            chart geometry helpers (no image output)
├── model_service.py  lazy-loaded, cached inference helpers
├── templates/        base, index, device, vision, advisory + _charts
├── static/style.css  workspace design system
├── requirements.txt
└── uploads/          runtime folder for image uploads (auto-cleaned)
```

## Notes

* All deep models run on CPU in the web layer and are cached after the first
  call, so subsequent requests are fast.
* Uploaded images are removed after inference to keep the folder tidy.
* Advisory reports are written to `farm_advisory/outputs/` and offered as
  downloadable `.md` and `.json` files.
* Use a WSGI server in production (e.g. `waitress`, `gunicorn`) - the built-in
  server is for development only.
