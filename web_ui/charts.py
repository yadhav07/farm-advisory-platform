"""Chart factory for the web dashboard.

Renders every dashboard visual server-side with matplotlib using the
non-interactive ``Agg`` backend, so the UI needs no JavaScript charting
library and no CDN. Each function returns raw PNG bytes that Flask serves
straight from a ``/chart/<kind>`` route.

Charts provided
----------------
``disease_donut``     donut of disease class probabilities (Random Forest)
``yield_gauge``       semicircular gauge for the Yield_Rate forecast
``sensor_radar``      radar of the normalized sensor profile vs healthy baseline
``sensor_bars``       per-feature reading vs its ideal agronomic band
``telemetry``         small-multiple time series of logged sensor readings
``vision_donut``      donut of image-classifier confidences (leaf / sky)
``forecast``          3-day weather forecast bars + precipitation line
``priority_donut``    donut of advisory actions by priority
"""

import io
import math
import threading

import matplotlib
matplotlib.use('Agg')  # headless rendering - no display required

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Matplotlib is not thread-safe; Flask serves requests on multiple threads.
_LOCK = threading.Lock()

# Palette shared with the stylesheet.
GREEN = '#2e7d32'
GREEN_MID = '#66bb6a'
GREEN_LIGHT = '#c8e6c9'
DARK = '#1b5e20'
AMBER = '#f9a825'
RED = '#c62828'
BLUE = '#1565c0'
TEAL = '#00838f'
PURPLE = '#6a1b9a'
GREY = '#90a4ae'
INK = '#263238'
MUTED = '#607d8b'
GRID = '#e0e7e2'

CATEGORY_COLORS = [GREEN, BLUE, AMBER, RED, TEAL, PURPLE, GREEN_MID, GREY]

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


def render(func, *args, **kwargs):
    """Thread-safe entry point - run a chart function under the global lock."""
    with _LOCK:
        return func(*args, **kwargs)


def _finish(fig):
    """Render a figure to PNG bytes and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=112, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return buffer.getvalue()


def _style_axes(ax, grid_axis='y'):
    ax.set_facecolor('white')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _friendly(name):
    return FEATURE_LABELS.get(name, name.replace('_', ' '))


# ---------------------------------------------------------------------------
# Model output charts
# ---------------------------------------------------------------------------
def disease_donut(probabilities, center_label='crop state'):
    """Donut chart of Random Forest class probabilities."""
    items = [(k, float(v)) for k, v in probabilities.items() if float(v) > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k.replace('_', ' ') for k, _ in items]
    values = [v for _, v in items]
    colors = CATEGORY_COLORS[:len(items)]

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    wedges, _ = ax.pie(
        values, startangle=90, counterclock=False, colors=colors,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
    )
    top = labels[0] if labels else '-'
    top_pct = values[0] * 100 if values else 0
    ax.text(0, 0.10, f'{top_pct:.0f}%', ha='center', va='center',
            fontsize=19, fontweight='bold', color=DARK)
    ax.text(0, -0.20, top.title(), ha='center', va='center',
            fontsize=9, color=MUTED)
    ax.legend(wedges, [f'{l.title()}  {v * 100:.1f}%' for l, v in zip(labels, values)],
              loc='center left', bbox_to_anchor=(0.98, 0.5),
              frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title('Disease class probabilities', color=INK, fontsize=11,
                 fontweight='bold', pad=12)
    return _finish(fig)


def yield_gauge(value, low=10.0, high=100.0):
    """Semicircular gauge for the Yield_Rate regression output."""
    value = float(value if value is not None else low)
    fraction = max(0.0, min(1.0, (value - low) / (high - low)))

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.set_aspect('equal')
    ax.axis('off')

    # Background track with amber / green / red zones.
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.30, facecolor='#eceff1',
                       edgecolor='white', linewidth=1))
    ax.add_patch(Wedge((0, 0), 1.0, 0, 66, width=0.30, facecolor='#ffcdd2',
                       edgecolor='none'))
    ax.add_patch(Wedge((0, 0), 1.0, 66, 138, width=0.30, facecolor='#ffecb3',
                       edgecolor='none'))
    ax.add_patch(Wedge((0, 0), 1.0, 138, 180, width=0.30, facecolor='#c8e6c9',
                       edgecolor='none'))

    # Needle.
    angle = math.radians(180 * fraction)
    ax.plot([0, 0.82 * math.cos(angle)], [0, 0.82 * math.sin(angle)],
            color=INK, linewidth=3, solid_capstyle='round', zorder=5)
    ax.scatter([0], [0], s=70, color=INK, zorder=6)

    ax.text(0, -0.30, f'{value:.1f}', ha='center', va='center',
            fontsize=24, fontweight='bold', color=DARK)
    ax.text(0, -0.52, 'Yield_Rate forecast', ha='center', va='center',
            fontsize=9, color=MUTED)
    ax.text(-1.02, -0.05, f'{low:.0f}', ha='center', fontsize=8, color=MUTED)
    ax.text(1.02, -0.05, f'{high:.0f}', ha='center', fontsize=8, color=MUTED)
    return _finish(fig)


def priority_donut(recommendations):
    """Donut of advisory actions grouped by priority."""
    order = [('High', RED), ('Medium', AMBER), ('Low', '#fdd835')]
    counts = {name: 0 for name, _ in order}
    for rec in recommendations or []:
        if rec.get('priority') in counts:
            counts[rec['priority']] += 1

    if not recommendations:
        fig, ax = plt.subplots(figsize=(4.4, 2.6))
        ax.axis('off')
        ax.text(0.5, 0.55, 'No actions', ha='center', fontsize=15,
                fontweight='bold', color=DARK)
        ax.text(0.5, 0.30, 'Current management plan is fine', ha='center',
                fontsize=9, color=MUTED)
        return _finish(fig)

    values = [counts[name] for name, _ in order if counts[name] > 0]
    colors = [color for name, color in order if counts[name] > 0]
    labels = [f'{name} ({counts[name]})' for name, _ in order if counts[name] > 0]

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, colors=colors,
                       wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2))
    ax.text(0, 0.08, str(len(recommendations)), ha='center', va='center',
            fontsize=20, fontweight='bold', color=DARK)
    ax.text(0, -0.22, 'actions', ha='center', va='center', fontsize=9, color=MUTED)
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(0.98, 0.5),
              frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title('Advisory priority mix', color=INK, fontsize=11,
                 fontweight='bold', pad=10)
    return _finish(fig)


# ---------------------------------------------------------------------------
# Sensor telemetry charts
# ---------------------------------------------------------------------------
def sensor_radar(values, guide):
    """Radar of the current reading normalized against each feature's range."""
    features = [f['name'] for f in guide]
    angles = [n / len(features) * 2 * math.pi for n in range(len(features))]
    angles += angles[:1]

    def norm(name, value):
        field = next(f for f in guide if f['name'] == name)
        span = field['high'] - field['low']
        return max(0.0, min(1.0, (float(value) - field['low']) / span))

    current = [norm(f, values[f]) for f in features]
    current += current[:1]
    baseline = [0.5] * len(features)
    baseline += baseline[:1]

    fig, ax = plt.subplots(figsize=(5.0, 4.4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, current, color=GREEN, linewidth=2.4, label='Current reading', zorder=3)
    ax.fill(angles, current, color=GREEN_MID, alpha=0.28, zorder=2)
    ax.plot(angles, baseline, color=BLUE, linewidth=1.5, linestyle='--',
            label='Ideal midpoint', zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_friendly(f) for f in features], fontsize=9, color=INK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color=MUTED)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.9)
    ax.spines['polar'].set_color(GRID)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2,
              frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title('Sensor profile vs ideal range', color=INK, fontsize=11,
                 fontweight='bold', pad=18)
    return _finish(fig)


def sensor_bars(values, guide):
    """Per-feature reading plotted against its ideal agronomic band."""
    features = list(guide)[::-1]
    labels, readings, in_range, lows, highs = [], [], [], [], []

    for field in features:
        labels.append(f"{_friendly(field['name'])} ({field['unit']})")
        value = float(values.get(field['name'], field['default']))
        span = field['high'] - field['low']
        readings.append(max(0.0, min(1.0, (value - field['low']) / span)) * 100)
        lows.append(0.0)
        highs.append(100.0)
        in_range.append(field['low'] <= value <= field['high'])

    y = range(len(features))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    # Ideal band marker (the middle 60% of the admissible range).
    ax.barh(list(y), highs, color='#f1f8f2', height=0.62, zorder=1)
    ax.barh(list(y), [40] * len(features), left=30, color=GREEN_LIGHT,
            height=0.62, zorder=2, label='ideal band')
    ax.barh(list(y), readings, color=[GREEN if ok else RED for ok in in_range],
            height=0.24, zorder=4, label='reading')

    for index, (field, pct, ok) in enumerate(zip(features, readings, in_range)):
        value = float(values.get(field['name'], field['default']))
        ax.text(min(pct + 2, 92), index, f'{value:.1f}',
                va='center', fontsize=8.5,
                color=INK if ok else RED, fontweight='bold' if not ok else 'normal')

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['min', '', 'mid', '', 'max'], fontsize=8, color=MUTED)
    _style_axes(ax, grid_axis='x')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='lower right')
    ax.set_title('Reading position within each sensor range (red = outside range)',
                 color=INK, fontsize=10.5, fontweight='bold', pad=10)
    return _finish(fig)


def telemetry(records, guide=None):
    """Small-multiple time series of the logged sensor readings.

    Each panel keeps one coherent scale so a low-range signal (pH) is never
    flattened by a high-range one (lux).
    """
    if not records:
        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No telemetry logged yet', ha='center', va='center',
                fontsize=12, color=MUTED)
        return _finish(fig)

    panels = [
        ('Temperature', [('Temperature', '#ef6c00')]),
        ('Humidity', [('Humidity', BLUE)]),
        ('Light intensity', [('Light_Intensity', '#f9a825')]),
        ('Soil moisture', [('Moisture', TEAL)]),
        ('Soil pH', [('PH', '#5d4037')]),
        ('Nutrients N·P·K', [('Nitrogen', GREEN), ('Phosphorus', AMBER),
                             ('Potassium', PURPLE)]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 4.9))
    x = list(range(len(records)))

    for ax, (title, series) in zip(axes.ravel(), panels):
        for feature, color in series:
            pairs = [(i, float(r[feature])) for i, r in enumerate(records)
                     if r.get(feature) is not None]
            if not pairs:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], color=color,
                    linewidth=1.9, marker='o', markersize=3.2, label=_friendly(feature))
        ax.set_title(title, fontsize=10, color=INK, fontweight='bold', pad=6)
        _style_axes(ax)
        if len(x) == 1:
            ax.set_xlim(-0.5, 0.5)
        else:
            ax.set_xlim(min(x), max(x))
        ax.margins(y=0.22)
        ax.legend(frameon=True, facecolor='white', edgecolor='none',
                  framealpha=0.85, fontsize=7.8, labelcolor=INK,
                  loc='upper left', handlelength=1.4, borderpad=0.3)

    fig.suptitle('Logged telemetry over time', color=INK, fontsize=11.5,
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig)


# ---------------------------------------------------------------------------
# Vision + weather charts
# ---------------------------------------------------------------------------
def vision_donut(items, title='Classification confidence'):
    """Donut of the top image-classifier confidences."""
    items = [(k, float(v)) for k, v in items if float(v) > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k.replace('_', ' ').title() for k, _ in items]
    values = [v for _, v in items]
    colors = CATEGORY_COLORS[:len(items)]

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, colors=colors,
                       wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2))
    if values:
        ax.text(0, 0.10, f'{values[0] * 100:.1f}%', ha='center', va='center',
                fontsize=18, fontweight='bold', color=DARK)
        # Wrap long class names onto two centred lines instead of truncating.
        words = labels[0].split()
        if len(words) > 2:
            split = (len(words) + 1) // 2
            ax.text(0, -0.16, ' '.join(words[:split]), ha='center', va='center',
                    fontsize=8.5, color=MUTED)
            ax.text(0, -0.31, ' '.join(words[split:]), ha='center', va='center',
                    fontsize=8.5, color=MUTED)
        else:
            ax.text(0, -0.22, labels[0], ha='center', va='center',
                    fontsize=8.5, color=MUTED)
    ax.legend(wedges, [f'{l}  {v * 100:.1f}%' for l, v in zip(labels, values)],
              loc='center left', bbox_to_anchor=(0.98, 0.5),
              frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(title, color=INK, fontsize=11, fontweight='bold', pad=10)
    return _finish(fig)


def forecast(rows):
    """3-day forecast: temperature range bars + precipitation probability line."""
    if not rows:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.axis('off')
        ax.text(0.5, 0.5, 'Forecast unavailable', ha='center', va='center',
                fontsize=12, color=MUTED)
        return _finish(fig)

    days = [r['date'] for r in rows]
    t_max = [float(r['t_max'] or 0) for r in rows]
    t_min = [float(r['t_min'] or 0) for r in rows]
    precip = [float(r['precip_prob'] or 0) for r in rows]
    index = list(range(len(days)))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (lo, hi) in enumerate(zip(t_min, t_max)):
        ax.bar(i, hi - lo, bottom=lo, width=0.46, color=GREEN_LIGHT,
               edgecolor=GREEN, linewidth=1.4, zorder=3)
        ax.text(i, hi + 0.8, f'{hi:.0f}°', ha='center', fontsize=9,
                color=DARK, fontweight='bold')
        ax.text(i, lo - 2.2, f'{lo:.0f}°', ha='center', fontsize=8.5, color=MUTED)

    ax.set_xticks(index)
    ax.set_xticklabels([d[5:] for d in days], fontsize=9)
    _style_axes(ax)
    ax.set_ylabel('°C', color=MUTED, fontsize=9)
    ax.set_ylim(min(t_min) - 5, max(t_max) + 5)

    ax2 = ax.twinx()
    ax2.plot(index, precip, color=BLUE, linewidth=2.2, marker='o',
             markersize=6, zorder=5)
    for i, value in enumerate(precip):
        ax2.annotate(f'{value:.0f}%', (i, value), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=8.5, color=BLUE)
    ax2.set_ylim(0, 130)
    ax2.set_ylabel('precipitation probability', color=BLUE, fontsize=9)
    ax2.tick_params(colors=BLUE, labelsize=8, length=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_color(GRID)

    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=GREEN_LIGHT, edgecolor=GREEN,
                      label='temperature range'),
        plt.Line2D([0], [0], color=BLUE, marker='o', label='precipitation probability'),
    ], loc='lower center', ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _finish(fig)
