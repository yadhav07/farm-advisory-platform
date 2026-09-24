# Web UI (Flask)

A chart-driven dashboard over the entire Farm Advisory Platform. Every trained
subsystem is visualised, and the full advisory pipeline runs end-to-end from a
single page.

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

Then open <http://127.0.0.1:5001>.

## Pages

| Route        | What it does                                                                      |
|--------------|-----------------------------------------------------------------------------------|
| `/`          | Overview dashboard — live telemetry, KPI tiles, model charts, pipeline flowchart   |
| `/sensor`    | Sensor analytics — disease donut, yield gauge, profile radar, range bars          |
| `/leaf`      | Leaf vision — upload a photo for EfficientNet-B0 diagnosis + confidence donut      |
| `/satellite` | Sky vision — classify cloud / rain / shine / sunrise + confidence donut            |
| `/weather`   | Live conditions, KPI tiles and a 3-day forecast chart                              |
| `/advisory`  | Full pipeline — priority mix, disease donut, ranked actions, report downloads       |

## Charts

Charts are rendered **server-side** with matplotlib (Agg backend) and served as
PNG from `/chart/<kind>`, so there is no JavaScript charting library and no CDN.

| Chart kind  | Visual                                                    |
|-------------|-----------------------------------------------------------|
| `disease`   | donut of disease class probabilities                      |
| `yield`     | semicircular gauge for the Yield_Rate forecast             |
| `radar`     | sensor profile against each feature's ideal range          |
| `bars`      | reading position inside each admissible range              |
| `telemetry` | small-multiple time series of the logged sensor readings   |
| `vision`    | donut of leaf / weather-state classifier confidences      |
| `forecast`  | 3-day temperature range bars + precipitation probability  |
| `priority`  | donut of advisory actions grouped by priority             |

Pages store their inference result once in a small in-memory store and pass an
`id` to each `<img>`, so charts never re-run a model.

## Pipeline flowchart

`templates/_flowchart.html` renders the five-stage data-flow diagram (inputs →
preprocessing → models → fusion → output) as inline SVG. Pages highlight the
stage they exercise. No external diagram library is required.

## Layout

```
web_ui/
├── app.py            Flask routes + chart dispatch
├── charts.py         matplotlib chart factory (thread-safe)
├── model_service.py  lazy-loaded, cached inference helpers
├── templates/        Jinja2 templates (base + pages + _flowchart)
├── static/style.css  dashboard stylesheet
├── requirements.txt
└── uploads/          runtime folder for image uploads (auto-cleaned)
```

## Notes

* All deep models run on CPU in the web layer and are cached after the first
  call, so subsequent requests are fast.
* Uploaded images are removed after inference to keep the folder tidy.
* Advisory reports are written to `farm_advisory/outputs/` and offered as
  downloadable `.md` and `.json` files.
* Use a WSGI server in production (e.g. `waitress`, `gunicorn`) — the built-in
  server is for development only.
