"""Recommendation engine for the farm advisory platform.

The engine fuses all trained subsystems into a single decision layer:

* ``sensor_model`` Random Forest  -> live disease state + yield forecast
* ``leaf_disease_model``          -> disease detected on a leaf photo
* ``satellite_weather_model``     -> weather state from a sky/field photo
* live weather feed               -> current conditions + short forecast

It returns a structured analysis with prioritized, actionable agronomic
recommendations (who, what, why) ready for the delivery layer.
"""

import os
import sys
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib

from config import (
    SENSOR_MODEL_PATH, SENSOR_THRESHOLDS_PATH, LEAF_MODEL_PATH,
    LEAF_DATASET_ROOT, SATELLITE_MODEL_PATH, SENSOR_FEATURES,
    SOIL_ADVISORY_THRESHOLDS, CROP_NAME,
)

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    pass


class RecommendationEngine:
    """Loads models lazily and produces farm advisories."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._sensor_bundle = None
        self._thresholds = None
        self._leaf_pipeline = None
        self._satellite_pipeline = None

    # ------------------------------------------------------------- loading
    def _load_sensor_bundle(self):
        if self._sensor_bundle is None:
            if not os.path.exists(SENSOR_MODEL_PATH):
                raise FileNotFoundError(
                    f'Sensor model missing: {SENSOR_MODEL_PATH}. '
                    'Run sensor_model/train.py first.'
                )
            self._sensor_bundle = joblib.load(SENSOR_MODEL_PATH)
        return self._sensor_bundle

    def _load_thresholds(self):
        if self._thresholds is None and os.path.exists(SENSOR_THRESHOLDS_PATH):
            with open(SENSOR_THRESHOLDS_PATH, 'r', encoding='utf-8') as handle:
                self._thresholds = json.load(handle)
        return self._thresholds

    def _load_leaf_pipeline(self):
        if self._leaf_pipeline is not None:
            return self._leaf_pipeline
        if not TORCH_AVAILABLE or not os.path.exists(LEAF_MODEL_PATH):
            return None
        try:
            import sys
            leaf_dir = os.path.dirname(LEAF_MODEL_PATH)
            if leaf_dir not in sys.path:
                sys.path.insert(0, leaf_dir)
            from test import infer_image as _leaf_infer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._leaf_pipeline = (_leaf_infer, LEAF_DATASET_ROOT, LEAF_MODEL_PATH, device)
        except Exception as exc:
            if self.verbose:
                print(f'[engine] leaf model unavailable: {exc}')
            return None
        return self._leaf_pipeline

    def _load_satellite_pipeline(self):
        if self._satellite_pipeline is not None:
            return self._satellite_pipeline
        if not TORCH_AVAILABLE or not os.path.exists(SATELLITE_MODEL_PATH):
            return None
        try:
            import sys
            sat_dir = os.path.dirname(SATELLITE_MODEL_PATH)
            if sat_dir not in sys.path:
                sys.path.insert(0, sat_dir)
            from test import infer_image as _sat_infer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._satellite_pipeline = (_sat_infer, device)
        except Exception as exc:
            if self.verbose:
                print(f'[engine] satellite model unavailable: {exc}')
            return None
        return self._satellite_pipeline

    # ------------------------------------------------------------- signals
    def _classify_sensor(self, readings):
        bundle = self._load_sensor_bundle()
        classifier = bundle.get('classifier', bundle.get('model'))
        regressor = bundle.get('regressor')
        scaler = bundle['scaler']
        label_encoder = bundle['label_encoder']

        row = pd.DataFrame(
            [[float(readings[f]) for f in SENSOR_FEATURES]],
            columns=SENSOR_FEATURES,
        )
        scaled = scaler.transform(row)

        label = label_encoder.inverse_transform(classifier.predict(scaled))[0]
        probs = {
            name: float(p)
            for name, p in zip(label_encoder.classes_, classifier.predict_proba(scaled)[0])
        }
        yield_forecast = None
        if regressor is not None:
            yield_forecast = round(float(regressor.predict(scaled)[0]), 2)

        # Find the single most likely disease (ignoring Healthy).
        risks = {k: v for k, v in probs.items() if k != 'Healthy'}
        top_disease = max(risks, key=risks.get) if risks else None
        top_disease_prob = risks.get(top_disease, 0.0) if top_disease else 0.0

        return {
            'disease_prediction': label,
            'disease_probabilities': probs,
            'top_disease_risk': {
                'disease': top_disease, 'probability': top_disease_prob,
            } if top_disease else None,
            'yield_forecast': yield_forecast,
        }

    def _classify_leaf(self, image_path):
        pipeline = self._load_leaf_pipeline()
        if pipeline is None:
            return None
        infer, dataset_root, model_path, device = pipeline
        try:
            result = infer(image_path, dataset_root, model_path, device)
            return {
                'image': image_path,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'all_probabilities': result.get('all_probabilities'),
            }
        except Exception as exc:
            if self.verbose:
                print(f'[engine] leaf inference failed: {exc}')
            return None

    def _classify_sky(self, image_path):
        pipeline = self._load_satellite_pipeline()
        if pipeline is None:
            return None
        infer, device = pipeline
        try:
            result = infer(image_path, device=device)
            return {
                'image': image_path,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'all_probabilities': result.get('all_probabilities'),
            }
        except Exception as exc:
            if self.verbose:
                print(f'[engine] satellite inference failed: {exc}')
            return None

    # --------------------------------------------------------- advisories
    def _sensor_advisories(self, readings, classified):
        adv = []
        thr = SOIL_ADVISORY_THRESHOLDS

        # 1. Disease risk from the Random Forest probabilities.
        risk = classified.get('top_disease_risk')
        if risk and risk['probability'] >= 0.40:
            adv.append({
                'priority': 'High',
                'category': 'Pest & Disease',
                'action': f"Inspect canopy for {risk['disease'].replace('_', ' ')} symptoms and "
                          'apply an approved fungicide/bactericide if confirmed.',
                'rationale': f"High-risk probability {risk['probability']*100:.0f}% for "
                             f"{risk['disease'].replace('_', ' ')} from live telemetry.",
            })
        elif risk and risk['probability'] >= 0.25:
            adv.append({
                'priority': 'Medium',
                'category': 'Pest & Disease',
                'action': f'Increase scouting frequency for {risk["disease"].replace("_", " ")}.',
                'rationale': f'Elevated risk probability {risk["probability"]*100:.0f}% for '
                             f'{risk["disease"].replace("_", " ")}.',
            })

        # 2. Soil moisture.
        m = readings['Moisture']
        if m < thr['Moisture']['low']:
            adv.append({'priority': 'High', 'category': 'Irrigation',
                        'action': 'Start scheduled irrigation (drip preferred, 10-15 mm).',
                        'rationale': f'Soil moisture {m}% is below the {thr["Moisture"]["low"]}% '
                                     'threshold - crops risk drought stress.'})
        elif m > thr['Moisture']['high']:
            adv.append({'priority': 'Medium', 'category': 'Drainage',
                        'action': 'Defer irrigation and check drainage; aerate compacted beds.',
                        'rationale': f'Soil moisture {m}% is above the {thr["Moisture"]["high"]}% '
                                     'threshold - waterlogging / root-rot risk.'})

        # 3. pH.
        ph = readings['PH']
        if ph < thr['PH']['low']:
            adv.append({'priority': 'Medium', 'category': 'Soil Amendment',
                        'action': f'Apply agricultural lime (~250-400 kg/acre) to raise pH.',
                        'rationale': f'Soil pH {ph} is acidic (below {thr["PH"]["low"]}); nutrients '
                                     'like phosphorus fixate in acidic soil.'})
        elif ph > thr['PH']['high']:
            adv.append({'priority': 'Medium', 'category': 'Soil Amendment',
                        'action': 'Apply elemental sulfur / ammonium-based fertilizers to lower pH.',
                        'rationale': f'Soil pH {ph} is alkaline (above {thr["PH"]["high"]}); iron and '
                                     'zinc availability drops in alkaline soil.'})

        # 4. Macro nutrients.
        for nutrient, label in (('Nitrogen', 'N'), ('Phosphorus', 'P'), ('Potassium', 'K')):
            low = thr[nutrient].get('low')
            if low and readings[nutrient] < low:
                doses = {'Nitrogen': '40-60 kg N/ha urea topdressing',
                         'Phosphorus': '30-40 kg P2O5/ha (DAP/SSP)',
                         'Potassium': '25-35 kg K2O/ha (MOP)'}
                adv.append({'priority': 'Medium', 'category': 'Fertilization',
                            'action': doses[nutrient],
                            'rationale': f'{label} level {readings[nutrient]:.1f} is below the '
                                         f'{low} threshold - deficiency detected.'})

        # 5. Temperature / humidity stress.
        temp, hum = readings['Temperature'], readings['Humidity']
        if temp > thr['Temperature']['high']:
            adv.append({'priority': 'Medium', 'category': 'Heat Management',
                        'action': 'Schedule irrigation in early morning; provide temporary shade nets '
                                  'for seedlings.',
                        'rationale': f'Temperature {temp} C exceeds {thr["Temperature"]["high"]} C - '
                                     'heat-stress window.'})
        if hum > thr['Humidity']['high']:
            adv.append({'priority': 'Medium', 'category': 'Pest & Disease',
                        'action': 'Ventilate greenhouse / prune dense canopy to lower humidity.',
                        'rationale': f'Relative humidity {hum}% exceeds {thr["Humidity"]["high"]}% - '
                                     'favourable for fungal spore germination.'})

        # 6. Yield outlook.
        y = classified.get('yield_forecast')
        if y is not None and y < 40:
            adv.append({'priority': 'Medium', 'category': 'Yield Management',
                        'action': 'Revisit nutrient + irrigation plan - projected yield is low.',
                        'rationale': f'Projected Yield_Rate {y} is well below a healthy range (60-90).'})

        return adv

    def _weather_advisories(self, weather):
        adv = []
        if not weather:
            return adv
        forecast = weather.get('forecast', {})
        t_max_next = forecast.get('t_max') or []
        precip_prob = forecast.get('precip_probability') or []
        precip_sum = forecast.get('precip_sum') or []

        rain_expected = bool(precip_sum) and any(float(v) > 5.0 for v in precip_sum[:2])
        rain_prob_high = bool(precip_prob) and any(float(v) >= 60 for v in precip_prob[:2])
        heat_expected = bool(t_max_next) and any(float(v) >= 38 for v in t_max_next[:2])

        if rain_expected or rain_prob_high:
            adv.append({'priority': 'High', 'category': 'Irrigation & Spraying',
                        'action': 'Delay irrigation and avoid foliar spraying over the next 24-48h.',
                        'rationale': 'Rainfall is expected in the forecast window - spraying would be '
                                     'washed off and irrigation would be wasteful.'})
        if heat_expected:
            adv.append({'priority': 'Medium', 'category': 'Heat Management',
                        'action': 'Shift irrigation to night/early-morning and raise water volume 10%.',
                        'rationale': f'Maximum temperature forecast reaches {max(t_max_next[:2])} C.'})

        label = weather.get('weather_label', '')
        if 'Thunderstorm' in label:
            adv.append({'priority': 'High', 'category': 'Crop Safety',
                        'action': 'Secure trellises and temporary structures; hold spraying.',
                        'rationale': f'Thunderstorm conditions reported ({label}).'})
        if weather.get('wind_speed_kmh', 0) and weather['wind_speed_kmh'] > 25:
            adv.append({'priority': 'Medium', 'category': 'Spraying',
                        'action': 'Avoid spraying today - wind will cause drift and poor coverage.',
                        'rationale': f"Wind speed {weather['wind_speed_kmh']} km/h exceeds the safe "
                                     'spray window (<25 km/h).'})
        if heat_expected is False and weather.get('temperature', 0) and weather['temperature'] <= 10:
            adv.append({'priority': 'Low', 'category': 'Crop Protection',
                        'action': 'Consider frost protection (irrigation mist / row covers) at night.',
                        'rationale': f"Current temperature {weather['temperature']} C approaches "
                                     'frost-risk levels.'})

        return adv

    def _leaf_advisory(self, leaf):
        if not leaf:
            return []
        prediction = leaf['prediction']
        if prediction == 'pepper bell healthy' or prediction == 'potato healthy':
            return []
        clean = prediction.replace('_', ' ').title()
        prob = leaf['probability'] * 100
        adv = {
            'High': [
                'pepper bell bacterial spot', 'potato early blight', 'potato late blight',
                'tomato target spot',
            ],
        }
        priority = 'High' if prediction in adv['High'] else 'Medium'
        return [{
            'priority': priority,
            'category': 'Leaf Disease',
            'action': f'{clean} detected on the leaf image - isolate affected plants, remove '
                      'infected foliage and apply a registered treatment.',
            'rationale': f'Leaf image classifier reports {clean} with {prob:.1f}% confidence.',
        }]

    def _sky_advisory(self, sky):
        if not sky:
            return []
        if sky['prediction'] == 'sunrise' or sky['prediction'] == 'shine':
            return []
        if sky['prediction'] == 'rain':
            return [{'priority': 'Medium', 'category': 'Field Operations',
                     'action': 'Plan harvest/machinery around the wet spell indicated by sky image.',
                     'rationale': f'Sky-image classifier reports rain with '
                                  f'{sky["probability"]*100:.1f}% confidence.'}]
        if sky['prediction'] == 'cloud':
            return [{'priority': 'Low', 'category': 'Observation',
                     'action': 'Consider supplemental lighting for greenhouse crops.',
                     'rationale': f'Sky-image classifier reports overcast conditions '
                                  f'({sky["probability"]*100:.1f}%).'}]
        return []

    # -------------------------------------------------------------- public
    def analyze(self, sensor, weather=None, leaf_image=None, sky_image=None):
        """Run the full advisory pipeline for one telemetry reading."""
        classified = self._classify_sensor(sensor)

        leaf = self._classify_leaf(leaf_image) if leaf_image else None
        sky = self._classify_sky(sky_image) if sky_image else None

        recommendations = []
        recommendations += self._sensor_advisories(sensor, classified)
        recommendations += self._weather_advisories(weather)
        recommendations += self._leaf_advisory(leaf)
        recommendations += self._sky_advisory(sky)
        recommendations.sort(key=lambda r: {'High': 0, 'Medium': 1, 'Low': 2}[r['priority']])

        healthy = classified['disease_prediction'] == 'Healthy'
        risk = classified['top_disease_risk']
        event_free = (healthy and (risk is None or risk['probability'] < 0.25)
                      and not recommendations and not leaf)

        summary = (
            f"{'Farm is healthy' if event_free else 'Action recommended'} - "
            f"crop state: {classified['disease_prediction']}; "
            f"yield forecast: {classified['yield_forecast']}; "
            f"{len(recommendations)} advisory action(s)."
        )

        return {
            'crop': sensor.get('crop', CROP_NAME),
            'node_id': sensor.get('node_id', 'unknown'),
            'timestamp': sensor.get('timestamp') or dt.datetime.now().isoformat(timespec='seconds'),
            'sensor': {**{f: sensor[f] for f in SENSOR_FEATURES}, **classified},
            'weather': weather,
            'leaf': leaf,
            'sky': sky,
            'recommendations': recommendations,
            'summary': summary,
        }


if __name__ == '__main__':
    # Quick sanity check with a healthy reading.
    engine = RecommendationEngine(verbose=True)
    sample = {
        'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
        'node_id': 'SIM-CHECK',
        'crop': CROP_NAME,
        'Temperature': 26.0, 'Humidity': 60.0, 'Moisture': 48.0,
        'Nitrogen': 48.0, 'Phosphorus': 42.0, 'Potassium': 40.0,
        'PH': 6.6, 'Light_Intensity': 620.0,
    }
    result = engine.analyze(sample, weather=None)
    print(json.dumps(result, indent=2))