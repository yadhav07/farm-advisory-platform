# Web UI (Flask)

A workspace dashboard over the Farm Advisory Platform. Every trained subsystem and
the ESP32 field node are visualised, and the full advisory pipeline runs
end-to-end from a single page.

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

## Interface

A fixed sidebar groups the workspace, a sticky top bar shows breadcrumbs and the
node's live state, and every page is built from the same primitives: metric
cards, panel cards, flow lanes, data tables and suggestion lists.

The sensor and advisory pages use **range sliders** with live value readouts
rather than number boxes, so a scenario can be dialled in by dragging.

## Pages

| Route        | What it does                                                                        |
|--------------|-------------------------------------------------------------------------------------|
| `/`          | Overview: live telemetry, metric cards, donut, signal flow, history, summary         |
| `/device`    | Field node: IP connection, reachability probe, known nodes, firmware payload contract |
| `/sensor`    | Sensor model: slider simulator, disease donut, yield gauge, radar, range bars        |
| `/leaf`      | Leaf vision: upload a photo for EfficientNet-B0 diagnosis + confidence donut          |
| `/satellite` | Sky vision: classify cloud / rain / shine / sunrise + confidence donut                |
| `/weather`   | Live conditions, metric cards and a 3-day forecast chart                             |
| `/advisory`  | Full pipeline: priority mix, disease donut, ranked suggestions, report downloads      |

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

| Macro         | Technique                                       | Visual                                        |
|---------------|-------------------------------------------------|-----------------------------------------------|
| `donut`       | CSS `conic-gradient` with a masked centre        | disease / confidence / priority mix            |
| `gauge`       | inline SVG arc (`stroke-dasharray`) + needle    | Yield_Rate forecast                            |
| `radar`       | inline SVG polygons, rings and axes             | sensor profile against ideal range             |
| `rangebars`   | HTML/CSS bars over a track                      | reading position inside each admissible range  |
| `sparkgrid`   | inline SVG polylines                           | small-multiple telemetry time series           |
| `forecast`    | HTML/CSS positioned columns                     | 3-day temperature range and precipitation      |
| `flow`        | HTML/CSS nodes with arrow connectors            | signal-flow lanes                              |
| `suggestions` | HTML cards                                      | ranked advisory actions                        |

Typography and icons come from Manrope and Font Awesome via CDN; the layout
degrades cleanly if they fail to load.

## Layout

```
web_ui/
├── app.py            Flask routes + field-node API
├── viz.py            chart geometry helpers (no image output)
├── model_service.py  lazy-loaded, cached inference helpers
├── templates/        Jinja2 templates (pages + _charts)
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
