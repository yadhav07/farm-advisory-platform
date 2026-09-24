"""Geometry helpers for the HTML/SVG dashboard charts.

These functions only compute coordinates, percentages and CSS gradient
strings. All rendering happens in the browser through plain HTML, CSS and
inline SVG - there is no server-side image generation and no charting
library.

The helpers are registered as Jinja globals in ``app.py`` so templates can
call them directly, e.g. ``donut_segments(items)``.
"""

import math

# Palette shared with the stylesheet.
CATEGORY_COLORS = [
    '#2e7d32', '#1565c0', '#f9a825', '#c62828',
    '#00838f', '#5e35b1', '#66bb6a', '#78909c',
]
PRIORITY_COLORS = {'High': '#c62828', 'Medium': '#f9a825', 'Low': '#fdd835'}

FEATURE_LABELS = {
    'Temperature': 'Temp',
    'Humidity': 'Humidity',
    'Moisture': 'Moisture',
    'Nitrogen': 'Nitrogen',
    'Phosphorus': 'Phosphorus',
    'Potassium': 'Potassium',
    'PH': 'pH',
    'Light_Intensity': 'Light',
}

GUARD_COLORS = {'low': '#c62828', 'mid': '#f9a825', 'high': '#2e7d32'}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def fmt(value, digits=1):
    """Format a number, passing through anything non-numeric."""
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return '-'


def pct(value, digits=1):
    """Format a 0-1 fraction as a percentage string."""
    try:
        return f'{float(value) * 100:.{digits}f}%'
    except (TypeError, ValueError):
        return '-'


def bar_position(value, low, high):
    """Where a reading sits inside its admissible range, as 0-100."""
    try:
        span = float(high) - float(low)
        if span <= 0:
            return 50.0
        pos = (float(value) - float(low)) / span * 100.0
    except (TypeError, ValueError):
        return 50.0
    return max(0.0, min(100.0, pos))


def in_range(value, low, high):
    try:
        return float(low) <= float(value) <= float(high)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# donut / pie chart
# ---------------------------------------------------------------------------
def donut_segments(values, colors=None, center_label=None, center_value=None):
    """Build a CSS ``conic-gradient`` donut from (label, fraction) pairs."""
    colors = colors or CATEGORY_COLORS
    items = [(str(label), max(0.0, float(weight)))
             for label, weight in values if weight is not None]
    total = sum(weight for _, weight in items)

    if total <= 0:
        return {
            'gradient': 'conic-gradient(#eceff1 0 100%)',
            'segments': [],
            'center_value': center_value or '0%',
            'center_label': center_label or 'no data',
        }

    segments, running = [], 0.0
    for index, (label, weight) in enumerate(sorted(items, key=lambda kv: -kv[1])):
        share = weight / total
        start, end = running * 100.0, (running + share) * 100.0
        segments.append({
            'label': label,
            'pct': share * 100.0,
            'color': colors[index % len(colors)],
            'start': start,
            'end': end,
        })
        running += share

    stops = ', '.join(
        f"{segment['color']} {segment['start']:.2f}% {segment['end']:.2f}%"
        for segment in segments
    )
    top = segments[0]
    return {
        'gradient': f'conic-gradient({stops})',
        'segments': segments,
        'center_value': center_value or f'{top["pct"]:.1f}%',
        'center_label': center_label or top['label'],
    }


# ---------------------------------------------------------------------------
# semicircular gauge
# ---------------------------------------------------------------------------
def gauge_geometry(value, low=10.0, high=100.0, width=240.0, height=140.0,
                   band_low=38.0, band_high=60.0):
    """Arc path, progress dash and needle tip for the yield gauge."""
    try:
        reading = float(value)
    except (TypeError, ValueError):
        reading = float(low)
    try:
        span = float(high) - float(low)
        fraction = 0.0 if span <= 0 else (reading - float(low)) / span
    except (TypeError, ValueError):
        fraction = 0.0
    fraction = max(0.0, min(1.0, fraction))

    radius = (width / 2.0) - 18.0
    cx, cy = width / 2.0, height - 24.0
    arc_length = math.pi * radius
    angle = math.radians(180.0 * fraction)   # 180 deg = left, 0 deg = right

    if fraction < 0.4:
        color = GUARD_COLORS['low']
    elif fraction < 0.66:
        color = GUARD_COLORS['mid']
    else:
        color = GUARD_COLORS['high']

    return {
        'width': width,
        'height': height,
        'cx': round(cx, 2),
        'cy': round(cy, 2),
        'radius': round(radius, 2),
        'track_path': (f'M {cx - radius:.2f} {cy:.2f} '
                       f'A {radius:.2f} {radius:.2f} 0 0 1 {cx + radius:.2f} {cy:.2f}'),
        'dasharray': f'{fraction * arc_length:.2f} {arc_length:.2f}',
        'tip_x': round(cx + radius * 0.76 * math.cos(angle), 2),
        'tip_y': round(cy - radius * 0.76 * math.sin(angle), 2),
        'fraction': fraction * 100.0,
        'color': color,
        'value': reading,
        'low': low,
        'high': high,
        'band_low': band_low,
        'band_high': band_high,
    }


# ---------------------------------------------------------------------------
# radar chart
# ---------------------------------------------------------------------------
def radar_geometry(values, guide, size=320.0, radius=104.0, rings=4):
    """Polygon points, axis lines and labels for the sensor profile radar."""
    center = size / 2.0
    features = [field['name'] for field in guide]
    count = len(features)
    if count == 0:
        return {'size': size, 'rings': [], 'axes': [], 'points': [],
                'current': '', 'ideal': ''}

    def point(index, fraction):
        angle = math.radians(-90.0 + index * 360.0 / count)
        return (center + radius * fraction * math.cos(angle),
                center + radius * fraction * math.sin(angle))

    ring_polygons = []
    for ring in range(1, rings + 1):
        fraction = ring / rings
        coords = [point(i, fraction) for i in range(count)]
        ring_polygons.append(' '.join(f'{x:.2f},{y:.2f}' for x, y in coords))

    axes, dots, current = [], [], []
    for index, field in enumerate(guide):
        x, y = point(index, 1.0)
        label_x, label_y = point(index, 1.24)
        raw = values.get(field['name'], field.get('default', 0))
        fraction = bar_position(raw, field['low'], field['high']) / 100.0
        px, py = point(index, fraction)
        current.append((px, py))
        dots.append((px, py, fmt(raw, 1)))
        axes.append({
            'label': FEATURE_LABELS.get(field['name'], field['name']),
            'x1': round(center, 2), 'y1': round(center, 2),
            'x2': round(x, 2), 'y2': round(y, 2),
            'label_x': round(label_x, 2), 'label_y': round(label_y, 2),
            'anchor': 'middle' if abs(label_x - center) < 6 else
                      ('start' if label_x > center else 'end'),
        })

    ideal = [point(i, 0.5) for i in range(count)]
    return {
        'size': size,
        'center': center,
        'rings': ring_polygons,
        'axes': axes,
        'dots': dots,
        'current': ' '.join(f'{x:.2f},{y:.2f}' for x, y in current),
        'ideal': ' '.join(f'{x:.2f},{y:.2f}' for x, y in ideal),
        'ring_labels': [f'{int(100 * r / rings)}%' for r in range(1, rings + 1)],
    }


# ---------------------------------------------------------------------------
# time-series sparklines
# ---------------------------------------------------------------------------
def sparkline(values, width=320.0, height=96.0, pad=8.0):
    """Polyline points plus the y-range for one telemetry series."""
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            numbers.append(None)

    present = [n for n in numbers if n is not None]
    if not present:
        return {'points': '', 'dots': [], 'min': 0, 'max': 0, 'count': 0,
                'width': width, 'height': height}

    low, high = min(present), max(present)
    if high - low < 1e-9:
        low, high = low - 1.0, high + 1.0
    span = high - low
    usable_h = height - 2 * pad
    count = len(numbers)
    step = (width - 2 * pad) / max(1, count - 1) if count > 1 else 0.0

    coords, dots = [], []
    for index, value in enumerate(numbers):
        if value is None:
            continue
        x = pad + (index * step if count > 1 else width / 2.0)
        y = pad + usable_h - ((value - low) / span) * usable_h
        coords.append((x, y))
        dots.append((x, y))

    return {
        'points': ' '.join(f'{x:.2f},{y:.2f}' for x, y in coords),
        'dots': [(round(x, 2), round(y, 2)) for x, y in dots],
        'min': low,
        'max': high,
        'count': count,
        'width': width,
        'height': height,
    }


# ---------------------------------------------------------------------------
# weather forecast bars
# ---------------------------------------------------------------------------
def forecast_geometry(rows, height=150.0):
    """Per-day column geometry: min/max bar position on a shared scale."""
    usable = height - 26.0
    lows, highs = [], []
    for row in rows:
        try:
            lows.append(float(row.get('t_min')))
            highs.append(float(row.get('t_max')))
        except (TypeError, ValueError):
            lows.append(None)
            highs.append(None)

    present = [v for v in lows + highs if v is not None]
    if not present:
        return {'days': [], 'scale_min': 0, 'scale_max': 1, 'height': height}

    scale_min = min(present) - 3.0
    scale_max = max(present) + 3.0
    span = scale_max - scale_min or 1.0

    days = []
    for index, row in enumerate(rows):
        low, high = lows[index], highs[index]
        if low is None or high is None:
            days.append({'date': row.get('date'), 'valid': False})
            continue
        days.append({
            'date': row.get('date'),
            'valid': True,
            'bottom': round(((low - scale_min) / span) * usable, 2),
            'height': round(((high - low) / span) * usable, 2),
            't_min': low,
            't_max': high,
            'precip_prob': row.get('precip_prob') or 0,
            'precip_sum': row.get('precip_sum') or 0,
        })
    return {'days': days, 'scale_min': scale_min, 'scale_max': scale_max,
            'height': height, 'usable': usable}


def history_stats(records, guide):
    """Per-feature min / mean / max for a list of logged readings."""
    stats = []
    if not records:
        return stats
    for field in guide:
        values = []
        for row in records:
            value = row.get(field['name'])
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        stats.append({
            'label': field['label'],
            'unit': field['unit'],
            'low': field['low'],
            'high': field['high'],
            'min': min(values),
            'mean': sum(values) / len(values),
            'max': max(values),
            'count': len(values),
        })
    return stats


# Exposed to Jinja as globals.
VIZ = {
    'fmt': fmt,
    'pct': pct,
    'bar_position': bar_position,
    'in_range': in_range,
    'donut_segments': donut_segments,
    'gauge_geometry': gauge_geometry,
    'radar_geometry': radar_geometry,
    'sparkline': sparkline,
    'forecast_geometry': forecast_geometry,
    'history_stats': history_stats,
    'PRIORITY_COLORS': PRIORITY_COLORS,
    'CATEGORY_COLORS': CATEGORY_COLORS,
}
