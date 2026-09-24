"""Self-test for the farm advisory application layer.

Verifies the whole chain without requiring live hardware or a network:

    1. builds a simulated IoT reading (normal + nutrient-stressed)
    2. runs the recommendation engine (sensor RF + yield regressor)
    3. attaches a cached/fallback weather snapshot
    4. delivers a report (console + file channels always enabled)

Run:  python test.py
"""

import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import SENSOR_LOG_PATH, REPORT_DIR
from iot_ingestion import SimulatedSensorNode, SensorIngestion
from recommendation_engine import RecommendationEngine
from delivery import ReportDelivery

SAMPLE_LEAF = os.path.join(
    os.path.dirname(__file__), '..', 'leaf_disease_model', 'dataset', 'test'
)


def _first_leaf_image():
    root = os.path.abspath(SAMPLE_LEAF)
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, name)
    return None


def main():
    print('=' * 72)
    print(' FARM ADVISORY SELF-TEST '.center(72, '='))
    print('=' * 72)

    ingestion = SensorIngestion(log_path=SENSOR_LOG_PATH)
    engine = RecommendationEngine(verbose=True)
    delivery = ReportDelivery(report_dir=REPORT_DIR)

    # 1) Healthy farm reading.
    healthy_node = SimulatedSensorNode('SIM-HEALTHY', seed=1, stress_profile='normal')
    healthy = ingestion.ingest_reading(healthy_node.read())

    # 2) A nutrient/moisture stressed reading so recommendations trigger.
    stressed_node = SimulatedSensorNode('SIM-STRESS', seed=3,
                                        stress_profile='nutrient-stress')
    stressed = ingestion.ingest_reading(stressed_node.read())
    stressed['Node'] = 'SIM-STRESS'
    stressed.pop('Node', None)

    # 3) Offline-safe weather snapshot (cached fetch; fallback if no network).
    from weather_feed import LiveWeatherFeed
    weather = LiveWeatherFeed().fetch_current()

    leaf_image = _first_leaf_image()
    print(f'[test] leaf sample: {leaf_image}')

    both = [
        ('healthy', healthy),
        ('stressed', stressed),
    ]
    passed = True
    for label, readings in both:
        print(f'\n--- analyze: {label} ---')
        analysis = engine.analyze(readings, weather=weather, leaf_image=leaf_image)
        nrecs = len(analysis['recommendations'])
        print(f"[test] {label}: state={analysis['sensor']['disease_prediction']} "
              f"yield={analysis['sensor']['yield_forecast']} recs={nrecs}")
        print(f"[test] summary: {analysis['summary']}")
        # Delivery (writes a real report into outputs/).
        delivery.deliver(analysis)
        assert 'recommendations' in analysis and 'sensor' in analysis
        if label == 'stressed':
            # The stressed reading should produce at least one action.
            if nrecs == 0:
                passed = False
                print('[test] WARNING: stressed reading produced no recommendations!')

    print('\n' + '=' * 72)
    print('PASSED ✅' if passed else 'FAILED ❌', '— self-test result')
    print('=' * 72)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())