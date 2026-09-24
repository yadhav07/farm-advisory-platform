/*
 * Farm Advisory Platform - ESP32 field node
 * ---------------------------------------------------------------------------
 * Based on the original sketch, with three additions:
 *
 *   1. the JSON payload is also served over HTTP (GET /api/sensor-data), so a
 *      dashboard can be pointed at this node by IP address and pull readings
 *      directly instead of waiting for the next push;
 *   2. optional soil-pH / NPK probes are read when present and sent in the
 *      payload, replacing the dashboard's estimated values;
 *   3. a MAC-derived device id is used unless DEVICE_ID is overridden.
 *
 * Wiring (unchanged):
 *   DHT11      GPIO4
 *   soil       GPIO35  (ADC1_CH7)
 *   MQ-135     GPIO34  (ADC1_CH6)
 *   BH1750     I2C     SDA 21 / SCL 22
 *   BME280     I2C     SDA 21 / SCL 22
 *   SSD1306    I2C     SDA 21 / SCL 22
 *
 * Board: ESP32 DevKit v1 (Arduino core 3.x)
 * Libraries: DHT sensor, BH1750, Adafruit BME280, Adafruit GFX, Adafruit SSD1306
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <Wire.h>

#include <DHT.h>
#include <BH1750.h>
#include <Adafruit_BME280.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// =====================================================
//                CONFIGURATION
// =====================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Address of the machine running the dashboard, e.g. 192.168.1.10
const char* SERVER_URL = "http://192.168.1.10:5001/api/sensor-data";

// Port the node listens on for direct pulls from the dashboard.
const uint16_t NODE_PORT = 8080;

// Leave as-is to derive the id from the board MAC address.
#define DEVICE_ID "FARM_01"

#define DHT_PIN 4
#define DHT_TYPE DHT11

#define SOIL_MOISTURE_PIN 35
#define MQ135_PIN 34

#define I2C_SDA 21
#define I2C_SCL 22

// Optional probes - set to a GPIO to enable, or -1 to leave them out.
#define SOIL_PH_PIN -1
#define NITROGEN_PIN -1
#define PHOSPHORUS_PIN -1
#define POTASSIUM_PIN -1

// Calibration from testing the actual sensor.
int SOIL_DRY_VALUE = 3000;
int SOIL_WET_VALUE = 1200;

const unsigned long SEND_INTERVAL = 10000;  // 10 seconds

// =====================================================
//                OBJECTS
// =====================================================

DHT dht(DHT_PIN, DHT_TYPE);
BH1750 lightMeter;
Adafruit_BME280 bme;
WebServer server(NODE_PORT);

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

bool bmeFound = false;
bool bh1750Found = false;

String deviceId = DEVICE_ID;

// Cached reading, shared by the push and pull paths.
struct Reading {
  float temperature = 0;
  float humidity = 0;
  int soilMoisture = 0;
  float light = 0;
  int mq135 = 0;
  float bmeTemperature = 0;
  float pressure = 0;
  float ph = -1;
  float nitrogen = -1;
  float phosphorus = -1;
  float potassium = -1;
};
Reading current;

// =====================================================
//                WIFI
// =====================================================

void connectWiFi()
{
  Serial.println();
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30)
  {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println("Wi-Fi connected.");
    Serial.print("Node IP address: ");
    Serial.println(WiFi.localIP());
  }
  else
  {
    Serial.println("Wi-Fi connection failed.");
  }
}

// =====================================================
//                SENSOR READS
// =====================================================

int readSoilMoisture()
{
  int rawValue = analogRead(SOIL_MOISTURE_PIN);

  int moisture = map(rawValue, SOIL_DRY_VALUE, SOIL_WET_VALUE, 0, 100);
  return constrain(moisture, 0, 100);
}

float readOptionalAnalog(int pin, int dryValue, int wetValue)
{
  if (pin < 0) return -1;
  int raw = analogRead(pin);
  if (dryValue == wetValue) return -1;
  float scaled = map(raw, dryValue, wetValue, 1000, 0) / 10.0f;
  return constrain(scaled, 0.0f, 14.0f);   // pH range
}

Reading readSensors()
{
  Reading r;

  r.humidity = dht.readHumidity();
  r.temperature = dht.readTemperature();

  if (isnan(r.humidity) || isnan(r.temperature))
  {
    Serial.println("DHT11 read failed; using previous values.");
    r.humidity = current.humidity;
    r.temperature = current.temperature;
  }

  r.soilMoisture = readSoilMoisture();
  r.mq135 = analogRead(MQ135_PIN);
  r.light = bh1750Found ? lightMeter.readLightLevel() : 0;

  if (bmeFound)
  {
    r.bmeTemperature = bme.readTemperature();
    r.pressure = bme.readPressure() / 100.0F;
  }

  r.ph = readOptionalAnalog(SOIL_PH_PIN, 3000, 1000);
#if NITROGEN_PIN >= 0
  r.nitrogen = analogRead(NITROGEN_PIN);
#endif
#if PHOSPHORUS_PIN >= 0
  r.phosphorus = analogRead(PHOSPHORUS_PIN);
#endif
#if POTASSIUM_PIN >= 0
  r.potassium = analogRead(POTASSIUM_PIN);
#endif

  return r;
}

void printSensors(const Reading& r)
{
  Serial.println();
  Serial.println("--------- SENSOR DATA ---------");
  Serial.print("Soil Moisture : "); Serial.print(r.soilMoisture); Serial.println(" %");
  Serial.print("DHT11 Temp    : "); Serial.print(r.temperature); Serial.println(" C");
  Serial.print("DHT11 Humidity: "); Serial.print(r.humidity); Serial.println(" %");
  Serial.print("Light         : "); Serial.print(r.light); Serial.println(" lux");
  Serial.print("MQ135         : "); Serial.println(r.mq135);
  Serial.print("BME Temp      : "); Serial.print(r.bmeTemperature); Serial.println(" C");
  Serial.print("Pressure      : "); Serial.print(r.pressure); Serial.println(" hPa");
  if (r.ph >= 0) { Serial.print("Soil pH       : "); Serial.println(r.ph); }
}

// =====================================================
//                OLED
// =====================================================

void showDisplay(const Reading& r)
{
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  display.print("Soil: ");  display.print(r.soilMoisture); display.println(" %");
  display.print("Temp: ");  display.print(r.temperature, 1); display.println(" C");
  display.print("Hum:  ");  display.print(r.humidity, 1); display.println(" %");
  display.print("Light:");  display.print(r.light, 0); display.println(" lx");
  display.print("MQ135:");  display.println(r.mq135);
  if (bmeFound) { display.print("Press:"); display.print(r.pressure, 1); display.println(" hPa"); }
  display.display();
}

// =====================================================
//                PAYLOAD
// =====================================================

String buildJson(const Reading& r)
{
  String json = "{";
  json += "\"device_id\":\"" + deviceId + "\",";
  json += "\"soil_moisture\":" + String(r.soilMoisture) + ",";
  json += "\"air_temperature\":" + String(r.temperature, 2) + ",";
  json += "\"humidity\":" + String(r.humidity, 2) + ",";
  json += "\"light_intensity\":" + String(r.light, 2) + ",";
  json += "\"mq135_raw\":" + String(r.mq135) + ",";
  json += "\"bme_temperature\":" + String(r.bmeTemperature, 2) + ",";
  json += "\"pressure\":" + String(r.pressure, 2);

  if (r.ph >= 0)          json += ",\"ph\":" + String(r.ph, 2);
  if (r.nitrogen >= 0)    json += ",\"nitrogen\":" + String(r.nitrogen, 1);
  if (r.phosphorus >= 0)  json += ",\"phosphorus\":" + String(r.phosphorus, 1);
  if (r.potassium >= 0)   json += ",\"potassium\":" + String(r.potassium, 1);

  json += "}";
  return json;
}

// =====================================================
//                PUSH TO DASHBOARD
// =====================================================

void sendDataToServer(const Reading& r)
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("Wi-Fi disconnected; reconnecting.");
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED)
    {
      Serial.println("Cannot send data.");
      return;
    }
  }

  String json = buildJson(r);

  Serial.println();
  Serial.println("========== JSON ==========");
  Serial.println(json);
  Serial.println("==========================");

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(6000);

  int code = http.POST(json);
  Serial.print("HTTP Response Code: ");
  Serial.println(code);

  if (code > 0)
  {
    Serial.print("Server response: ");
    Serial.println(http.getString());
  }

  http.end();
}

// =====================================================
//                PULL ENDPOINTS (for IP-based dashboard access)
// =====================================================

void handleRoot()
{
  String body = "Farm node " + deviceId + " - use /api/sensor-data\n";
  server.send(200, "text/plain", body);
}

void handleSensorData()
{
  String body = buildJson(current);
  server.send(200, "application/json", body);
}

void startNodeServer()
{
  server.on("/", HTTP_GET, handleRoot);
  server.on("/api/sensor-data", HTTP_GET, handleSensorData);
  server.on("/data", HTTP_GET, handleSensorData);
  server.begin();
  Serial.print("Node API listening on port ");
  Serial.println(NODE_PORT);
}

// =====================================================
//                SETUP
// =====================================================

void setup()
{
  Serial.begin(115200);
  delay(1000);

  // Derive a stable id from the board MAC when DEVICE_ID is left as default.
  if (String(DEVICE_ID) == "FARM_01")
  {
    uint8_t mac[6];
    WiFi.macAddress(mac);
    deviceId = "FARM-" + String(mac[4], HEX) + String(mac[5], HEX);
  }

  Serial.println();
  Serial.println("================================");
  Serial.println("   FARM ADVISORY - ESP32 NODE");
  Serial.println("================================");

  Wire.begin(I2C_SDA, I2C_SCL);

  dht.begin();
  Serial.println("DHT11 initialised.");

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
  {
    Serial.println("OLED not found.");
  }
  else
  {
    Serial.println("OLED initialised.");
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(8, 20);
    display.println("FARM NODE");
    display.setCursor(8, 36);
    display.println("starting...");
    display.display();
    delay(1500);
  }

  bh1750Found = lightMeter.begin();
  Serial.println(bh1750Found ? "BH1750 detected." : "BH1750 not detected.");

  if (bme.begin(0x76))          { bmeFound = true; Serial.println("BME280 detected at 0x76."); }
  else if (bme.begin(0x77))     { bmeFound = true; Serial.println("BME280 detected at 0x77."); }
  else                          { Serial.println("BME280 not detected."); }

  analogReadResolution(12);
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(MQ135_PIN, INPUT);

  connectWiFi();
  startNodeServer();

  current = readSensors();

  Serial.println();
  Serial.print("Device id: ");
  Serial.println(deviceId);
  Serial.println("System ready.");
}

// =====================================================
//                LOOP
// =====================================================

unsigned long lastSendTime = 0;

void loop()
{
  server.handleClient();

  current = readSensors();

  printSensors(current);
  showDisplay(current);

  if (millis() - lastSendTime >= SEND_INTERVAL)
  {
    lastSendTime = millis();
    sendDataToServer(current);
  }

  delay(1000);
}
