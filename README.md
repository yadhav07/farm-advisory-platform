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
timing, nutrient top-dressing, pH correction, disease control and safe spraying
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
`0.0.0.0` so field nodes on the same network can reach it.

Four pages: **overview** (live metrics, the disease pie chart, the reading log
and the trend charts), **field node** (connect an ESP32 by IP), **vision**
(leaf and sky photo classifiers) and **advisory** (run the pipeline with live
weather, get ranked actions and a downloadable report).

### 3. Connect an ESP32 field node

The firmware in `hardware/esp32_farm_node/` posts readings to this server:

```json
POST http://<your-pc-ip>:5001/api/sensor-data
{
  "device_id": "FARM_01",
  "soil_moisture": 44,
  "air_temperature": 27.3,
  "humidity": 64.8,
  "light_intensity": 585.2,
  "mq135_raw": 1240,
  "bme_temperature": 26.1,
  "pressure": 1008.4
}
```

Open the **Field node** page, type the ESP32's IP address (for example
`192.168.1.50`) and press **Connect**. The dashboard lists known nodes, shows
online/idle state, and the overview then renders that node's live telemetry
through the Random Forest heads.

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/sensor-data` | ingest a firmware payload, returns crop state and yield forecast |
| `GET /api/devices` | known nodes with IP, last reading and online state |
| `GET /api/latest` | latest reading mapped to the 8-feature model schema |

Four features map straight from the hardware (temperature, humidity, moisture,
light). `ph`, `nitrogen`, `phosphorus` and `potassium` are **estimated** from soil
moisture and the MQ-135 reading until those probes are fitted; send them in the
payload and they are used verbatim instead. The UI labels each value as
measured or estimated.

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
```

`project_architecture.md` contains the complete design write-up: feature
definitions, the yield-rate formulation, threshold tuning methodology, model
architecture details and comparisons.

---

## Data sources

* **Sensor telemetry** — field-style datasets for temperature, humidity, soil
  moisture, pH and NPK levels, with a synthesized `Yield_Rate` target.
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
