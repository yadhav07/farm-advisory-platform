# Farm Advisory Platform

A multi-modal agronomy machine-learning system that combines field telemetry,
plant imagery and live weather into a single actionable farm advisory — with a
Flask web interface on top.

The repository contains four model/integration subsystems plus a web front end,
each self-contained and runnable on its own.

---

## What it does

| Subsystem | Model | Task | Measured performance |
| :--- | :--- | :--- | :--- |
| `sensor_model` | Random Forest classifier + regressor | Crop disease state from 8 soil/air readings, plus a continuous `Yield_Rate` forecast | Classifier CV F1 **0.94** · Regressor R² **0.64**, MAE **3.4** |
| `leaf_disease_model` | EfficientNet-B0 (transfer learning) | Leaf disease diagnosis from field photography | Test accuracy **97.62%** · F1 **98.33%** |
| `satellite_weather_model` | Custom CNN (`Net`) | Weather-state classification (cloud / rain / shine / sunrise) from optical imagery | Test accuracy **84.88%** |
| `farm_advisory` | Fusion layer | Fuses all models + live weather into prioritized agronomic actions and delivers reports | — |

The advisory layer turns model output into concrete instructions — irrigation
timing, pH correction, disease control and safe spraying
windows — each with a plain-language rationale, ranked High / Medium / Low.

---

## Quick start

### 1. Requirements

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn pillow torch torchvision flask
```

### 2. Run the dashboard

```bash
cd farm_advisory
python app.py
```

The browser opens automatically at <http://127.0.0.1:5001>. The server binds to
`0.0.0.0` and prints its LAN address on startup, so field nodes on the same
network can reach it:

```
[app] local    http://127.0.0.1:5001
[app] network  http://10.1.132.248:5001   <- point the ESP32 SERVER_URL here
[app] health   http://10.1.132.248:5001/healthz
[app] ingest   http://10.1.132.248:5001/api/sensor-data
```

Four pages behind a sidebar: **overview** (live metrics, the disease pie chart,
the reading log and the trend charts), **leaf vision** and **sky vision** (the
two image classifiers, one page each), and **advisory** (run the pipeline with
live weather, get ranked actions and a report). There is no simulated node: with
nothing reporting the overview still renders in full, with empty values instead
of a placeholder screen, and it polls for a node so it fills itself in the
moment one appears.

| Route | Page |
| :--- | :--- |
| `/` | Overview |
| `/vision/leaf` | Leaf Vision - EfficientNet-B0 disease classification |
| `/vision/sky` | Sky Vision - weather CNN state classification |
| `/advisory` | Advisory - full pipeline, ranked actions, report |

Both vision pages share one template; `/vision` redirects to `/vision/leaf`.

`GET /healthz` is a liveness probe that loads no model, so it answers while the
artifacts are still cold.

### 2b. Host sizing

This app is memory-hungry and a small instance will crash it. Measured
resident memory, one step at a time:

| Stage | Resident |
| :--- | ---: |
| python + flask | 18 MB |
| + Random Forest bundle (the overview needs it) | 203 MB |
| + `import torch` | 659 MB |
| + EfficientNet-B0 | 836 MB |
| + weather CNN | 899 MB |

Importing `torch` alone costs about 460 MB, more than every model in the
project put together. No amount of caching changes that, so the service
needs **at least 2 GB**.

On Render that means the **standard** instance. Note that **starter is
also 512 MB**, so upgrading one tier will not help - it has more CPU but
the same memory as free.

Two mitigations are in the code, and they buy headroom on a host that is
big enough rather than making a small host work:

- Only **one image model is resident at a time** by default. The LRU evicts
  the other before loading a new one, which keeps roughly 180 MB free. Set
  `IMAGE_MODEL_CACHE=2` to keep both and pay a reload when a user
  alternates between the two vision pages.
- `torch` is pinned to **one thread**. It reserves a work arena per
  thread, so the default thread count is a real memory cost, and inference
  is a single 224x224 image where extra threads buy nothing.

If you must stay on a 512 MB instance, the overview and advisory work
(203 MB, no torch), but any request to a vision page will pull torch in
and take the whole process down with it - the OOM is process-wide, so the
overview goes down too.

### 2c. Run it as a production server

`python app.py` uses the Flask development server, which is fine locally but is
not meant for anything else. For a real deployment use the WSGI entry point:

```bash
pip install -r requirements.txt
python wsgi.py                      # waitress on 0.0.0.0:5001
```

`HOST`, `PORT` and `THREADS` are read from the environment, so the same command
works unchanged on a host that assigns the port:

```bash
PORT=8080 THREADS=16 python wsgi.py
```

Under gunicorn or any PaaS:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 wsgi:application
```

**Keep the worker count at 1 and raise threads instead.** The three trained
artifacts total about 130 MB and a second worker would load a second copy of
every model into memory. They load lazily, so the process starts cheaply and only
the page you open pays for its model.

`Procfile` and `render.yaml` are already set up for Heroku/Railway and Render.
The `render.yaml` build installs CPU-only torch, which is much smaller than the
default CUDA wheel; the web layer runs every model on CPU anyway. Budget at least
2 GB of RAM on the host, because the first request to a vision page pulls
EfficientNet-B0 and the weather CNN into memory.

### 3. Connect an ESP32 field node

The full contract is in **[`docs/esp32_api.md`](docs/esp32_api.md)** - endpoint,
JSON format, response shape, accepted ranges and firmware notes. In short:

```json
POST https://farm-advisory-platform.onrender.com/api/sensor-data
Content-Type: application/json

{
  "device_id": "FARM_01",
  "soil_moisture": 65,
  "air_temperature": 29.5,
  "humidity": 72,
  "light_intensity": 540,
  "bme_temperature": 29.2,
  "pressure": 1008,
  "mq135_raw": 820
}
```

No API key. HTTPS. `soil_moisture`, `air_temperature` and `humidity` are
required; everything else is optional. The reply carries the predicted crop
state and yield score, so the node gets an assessment back for free.

The firmware in `hardware/esp32_farm_node/` posts readings to this server. Set
`SERVER_URL` at the top of the sketch. To see it working without hardware, post
a reading by hand:

```bash
curl -X POST https://farm-advisory-platform.onrender.com/api/sensor-data \
  -H 'Content-Type: application/json' \
  -d '{"device_id":"FARM_01","soil_moisture":44,"air_temperature":27.3,
       "humidity":64.8,"light_intensity":585.2}'
```

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/sensor-data` | ingest a firmware payload, returns crop state and yield forecast |
| `GET /api/devices` | known nodes with IP, last reading and online state |
| `GET /api/latest` | latest reading mapped to the 5-feature model schema (`source: none` until a node reports) |
| `GET /api/history` | the recorded readings, as JSON or `?format=csv` for a file download |
| `GET /healthz` | liveness probe, loads no model |

Three features map straight from the hardware (temperature, humidity, moisture)
and `light_intensity` is used when the node has an LDR. `ph` is **estimated**
from soil moisture and the MQ-135 reading until a pH probe is fitted; send `ph`
in the payload and it is used verbatim instead. The UI labels each value as
measured or estimated.

Nitrogen, phosphorus and potassium were **removed** from the schema. No probe
reports them, so the dashboard would have been showing three invented numbers on
every reading. A soil lab test is the right source for those.

### Where the reading history lives

Every accepted reading is appended to
`farm_advisory/dataset/device_readings.csv` (12 columns: `timestamp`,
`device_id`, `source_ip`, `firmware_measured`, the 5 features, then
`mq135_raw`, `bme_temperature`, `pressure`). `devices.json` alongside it holds
the newest reading per node.

Read it back over HTTP with `GET /api/history`:

```bash
curl 'https://farm-advisory-platform.onrender.com/api/history?device_id=FARM_01&limit=50'
curl 'https://farm-advisory-platform.onrender.com/api/history?format=csv' -o readings.csv
curl 'https://farm-advisory-platform.onrender.com/api/history?since=2026-09-28'
```

**This file is not durable on the current hosting plan.** It is written inside
the container filesystem, so it is discarded on every restart, redeploy or
out-of-memory kill - and the free tier does all three routinely. Readings
survive only until the next restart. To keep them, move the path onto a
persistent disk (`FARM_DATA_DIR`) and attach one; see *Host sizing* above.

### 4. Or run the live CLI pipeline

```bash
cd farm_advisory
python app.py --cli                    # one reading + live weather
python app.py --cli --steps 5 --interval 3
python app.py --cli --leaf "C:/leaf.jpg" --sky "C:/sky.jpg"
python app.py --cli --offline          # use cached weather, no API call
```

### 5. Retrain from scratch (optional)

Trained weights are committed, so this is only needed to reproduce them:

```bash
python sensor_model/train.py            # Random Forest + yield regressor
python sensor_model/visualize.py        # regenerate report plots
python leaf_disease_model/train.py      # EfficientNet-B0 fine-tuning
python satellite_weather_model/main.py  # weather CNN
```

---

## Project structure

```
Multidisciplinary_Project/
├── sensor_model/              # Telemetry ML — Random Forest + yield regressor
│   ├── dataset/               # telemetry CSVs + tuned thresholds
│   ├── outputs/               # confusion matrix, feature importance, yield plots
│   ├── train.py, test.py, visualize.py
│   └── rf_model.joblib
├── leaf_disease_model/        # Leaf disease CNN (EfficientNet-B0)
│   ├── dataset/{train,valid,test}/
│   ├── outputs/               # confusion matrix PNG/CSV
│   ├── train.py, test.py
│   └── best_leaf_disease_model.pth
├── satellite_weather_model/   # Weather-state CNN
│   ├── dataset/{train,valid,test}/{cloud,rain,shine,sunrise}/
│   ├── outputs/               # training curves + confusion matrix
│   ├── main.py, test.py
│   └── satellite_weather_model.pth
├── farm_advisory/             # Integration layer
│   ├── config.py              # paths, farm location, thresholds, delivery settings
│   ├── weather_feed.py        # live weather (Open-Meteo, cached, offline-safe)
│   ├── iot_ingestion.py       # telemetry validation + simulated sensor node
│   ├── device_ingest.py       # ESP32 field-node ingestion and device registry
│   ├── recommendation_engine.py
│   ├── delivery.py            # markdown/JSON reports, optional email/Telegram
│   ├── app.py                 # entry point: web UI (default) or --cli pipeline
│   ├── dataset/, outputs/
│   └── test.py
├── web_ui/                    # Flask dashboard
│   ├── app.py, viz.py, model_service.py
│   ├── templates/, static/
│   └── README.md
├── hardware/
│   └── esp32_farm_node/       # ESP32 firmware (posts + serves readings)
└── project_architecture.md    # full technical specification
wsgi.py                   # production WSGI entry point (gunicorn / waitress / PaaS)
requirements.txt          # pinned runtime dependencies
Procfile                  # Heroku / Railway / Render start command
render.yaml               # one-click Render blueprint
```

`project_architecture.md` contains the complete design write-up: feature
definitions, the yield-rate formulation, threshold tuning methodology, model
architecture details and comparisons.

---

## Data sources

* **Sensor telemetry** — field-style datasets for temperature, humidity, soil
  moisture, pH and light intensity, with a synthesized `Yield_Rate` target.
* **Leaf imagery** — multi-class leaf disease photographs (pepper, potato,
  tomato) with healthy counterparts.
* **Weather imagery** — labelled sky/field photographs across cloud, rain, shine
  and sunrise conditions (1,124 images, split 70/15/15).

Licensing and provenance notes ship alongside each dataset.

---

## Notes

* The advisory layer degrades gracefully: if the weather API is unreachable it
  falls back to the last cached snapshot, and if no cache exists it uses a
  conservative offline default.
* Report delivery works out of the box (markdown + JSON files, console output).
  SMTP email and Telegram push are optional and disabled until credentials are
  filled into `farm_advisory/config.py`.
* The Flask server bundled here is the development server; put it behind a WSGI
  server (waitress, gunicorn) for real deployment.
