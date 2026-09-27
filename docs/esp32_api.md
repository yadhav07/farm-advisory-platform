# ESP32 field node API

Everything an ESP32 needs to push sensor readings to the FarmIQ dashboard.
Every response shown here was captured from the live deployment.

---

## 1. The endpoint

| | |
| :--- | :--- |
| **API URL** | `https://farm-advisory-platform.onrender.com/api/sensor-data` |
| **Method** | `POST` |
| **Header** | `Content-Type: application/json` |
| **API key** | None. No authentication is required. |
| **HTTPS** | Yes, with a valid certificate. Use the `https://` URL as-is. |

Base URL: `https://farm-advisory-platform.onrender.com`

---

## 2. What to send

```json
{
  "device_id": "FARM_01",
  "soil_moisture": 65,
  "air_temperature": 29.5,
  "humidity": 72,
  "light_intensity": 540,
  "bme_temperature": 29.2,
  "pressure": 1008,
  "mq135_raw": 820,
  "ph": 6.4
}
```

### Required

The request is rejected with `400` if any of these three is missing.

| Key | Unit | Notes |
| :--- | :--- | :--- |
| `soil_moisture` | % | Capacitive soil probe, 0-100 |
| `air_temperature` | °C | DHT11, float |
| `humidity` | % | DHT11, float |

### Optional

| Key | Unit | If omitted |
| :--- | :--- | :--- |
| `device_id` | string | Defaults to `ESP32-UNKNOWN`. **Set this** so the node is identifiable. |
| `light_intensity` | lux | Defaults to 620 (BH1750) |
| `ph` | pH | Estimated from soil moisture instead |
| `bme_temperature` | °C | Not used by the models; shown for context only |
| `pressure` | hPa | Not used by the models; shown for context only |
| `mq135_raw` | ADC 0-4095 | Not used by the models; shown as an air-quality index |

The model reads five values: `soil_moisture`, `air_temperature`, `humidity`,
`light_intensity` and `ph`. Everything else is context for the display.

### Minimum working payload

If the node only has a soil probe and a DHT11, this is enough:

```json
{"device_id": "FARM_01", "soil_moisture": 65, "air_temperature": 29.5, "humidity": 72}
```

---

## 3. What you get back

`200 OK`:

```json
{
  "ok": true,
  "device_id": "FARM_01",
  "received_at": "2026-09-27T15:03:57",
  "crop_state": "Healthy",
  "yield_forecast": 57.64,
  "provenance": {
    "Temperature": "measured",
    "Humidity": "measured",
    "Moisture": "measured",
    "PH": "estimated",
    "Light_Intensity": "measured"
  }
}
```

| Field | Meaning |
| :--- | :--- |
| `ok` | `true` when the reading was stored |
| `device_id` | The id the reading was filed under |
| `received_at` | Server timestamp, `YYYY-MM-DDTHH:MM:SS` (UTC, no timezone suffix) |
| `crop_state` | The disease the classifier predicted |
| `yield_forecast` | Predicted yield score, 10-100 |
| `provenance` | Per feature, `measured` or `estimated` |

`crop_state` is one of: `Healthy`, `Early_Blight`, `Root_Rot`, `Powdery_Mildew`,
`Rust`, `Bacterial_Leaf_Spot`.

On failure, `400 Bad Request`:

```json
{"ok": false, "error": "payload must include soil_moisture, air_temperature and humidity"}
```

Note: a field sent as a non-numeric string produces the same
"must include ..." message rather than a type-specific one, because the value
is treated as absent. The request still fails correctly with `400`.

---

## 4. Out-of-range values are clamped, not rejected

The server clamps readings into a safe range rather than refusing them, so a
miswired probe cannot break a request. There is no need to clamp in firmware.

| Field | Clamped to |
| :--- | :--- |
| `soil_moisture` | 10 - 80 |
| `air_temperature` | 15 - 35 |
| `humidity` | 30 - 95 |
| `light_intensity` | 200 - 1000 |
| `ph` | 4.5 - 8.5 |

---

## 5. Read-back endpoints

Useful for debugging a node. All are `GET` and need no key.

| Endpoint | Returns |
| :--- | :--- |
| `/api/latest` | The latest reading, mapped to the model schema. `{"source": "none"}` until a node reports. |
| `/api/devices` | Every node that has reported, with IP, last-seen time and online state |
| `/healthz` | Liveness probe. Answers fast and loads no model. |

`/api/latest` accepts `?device_id=FARM_01` to pick a specific node.

---

## 6. Practical notes for the firmware

**The site sleeps when idle.** It runs on a free hosting tier, so after about
15 minutes with no traffic it powers down. The first request after that can
take **30-60 seconds** while it wakes up.

- Set the HTTP client timeout to **90 seconds**, not the usual 10.
- Treat a timeout as **non-fatal**. Log it and post again on the next cycle.
  A wake-up timeout is normal, not a fault.
- Do not retry immediately in a tight loop. One retry per cycle is enough.

**Recommended cadence:** post every 15-60 seconds. The dashboard keeps a rolling
history, and a reading every 30s gives the trend charts a useful shape.

**Use HTTPS with the default certificate.** There is no need to pin a
certificate or install a root CA; the platform's certificate is valid.

**Retries:** if you get no response at all, just send again on the next cycle.
The server is stateless per request, so a retry is always safe. Duplicate
readings are harmless, they simply add another row to the history.

**Identify the node.** Set `device_id` to something stable and unique, such as
the last six digits of the MAC address. It is how a reading is traced back to a
physical board in the dashboard.

---

## 7. Copy-paste test

Before wiring the firmware, confirm the endpoint works from a laptop:

```bash
curl -X POST https://farm-advisory-platform.onrender.com/api/sensor-data \
  -H 'Content-Type: application/json' \
  -d '{"device_id":"FARM_01","soil_moisture":65,"air_temperature":29.5,
       "humidity":72,"light_intensity":540,"mq135_raw":820,
       "bme_temperature":29.2,"pressure":1008}'
```

A `200` with `"ok":true` means the endpoint is live. The reading appears on the
dashboard Overview page immediately.

---

## 8. Firmware sketch

`hardware/esp32_farm_node/esp32_farm_node.ino` in this repository is a working
implementation of everything above. The constant to change is near the top:

```cpp
#define SERVER_URL "https://farm-advisory-platform.onrender.com/api/sensor-data"
```

The sketch also serves `GET /api/sensor-data` on port 8080, so a dashboard on
the same network can pull readings directly instead of waiting for the next
push.
