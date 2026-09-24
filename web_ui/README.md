# Web UI (Flask)

A browser interface for the entire Farm Advisory Platform. Every trained
subsystem is exposed as a web tool, and the full advisory pipeline can be
run end-to-end from a single page.

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

| Route     | What it does                                                        |
|-----------|---------------------------------------------------------------------|
| `/`       | Dashboard overview of the available tools                       |
| `/sensor` | Disease diagnosis + yield forecast from an 8-feature reading        |
| `/leaf`   | Leaf disease detection from an uploaded photo (EfficientNet-B0)     |
| `/satellite` | Weather-state classification from a sky/field photo              |
| `/weather`  | Live conditions + 3-day forecast (Open-Meteo), any lat/lon         |
| `/advisory` | Full pipeline: sensor + optional photos + live weather -> prioritized recommendations, with markdown/JSON report download |

## Layout

```
web_ui/
├── app.py            Flask routes (one per subsystem)
├── model_service.py  lazy-loaded, cached inference helpers
├── templates/        Jinja2 templates (base + one per page)
├── static/style.css  dependency-free stylesheet
├── requirements.txt
└── uploads/          runtime folder for image uploads (auto-cleaned)
```

## Notes

* All deep models run on CPU in the web layer and are cached after the
  first call, so subsequent requests are fast.
* Uploaded images are removed after inference to keep the folder tidy.
* Advisory reports are written to `farm_advisory/outputs/` and offered as
  downloadable `.md` and `.json` files.
* Use `FLASK` in production behind a WSGI server (e.g. `waitress`,
  `gunicorn`) — the built-in server is for development only.