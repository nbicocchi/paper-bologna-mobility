"""Definitions for the car-sharing study. Never export user IDs or number plates."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cars'
CACHE = ROOT / 'code/dati_esterni/cars'
TEXT = ROOT.parent / 'paper-vehicle-sharing-text'
SEED = 20261024
LAT0, LON0 = 44.4949, 11.3426
VERSION = 'cars-v1'
DST_DATES = ['2023-03-26', '2023-10-29', '2024-03-31', '2024-10-27']
HOLIDAYS = {f'{year}-{day}' for year in (2023, 2024) for day in
            ('01-01', '01-06', '04-25', '05-01', '06-02', '08-15', '10-04', '11-01', '12-08', '12-25', '12-26')}
HOLIDAYS |= {'2023-04-09', '2023-04-10', '2024-03-31', '2024-04-01'}
for path in (OUT, CACHE, TEXT / 'figures', TEXT / 'tables'):
    path.mkdir(parents=True, exist_ok=True)


def xy(lat, lon):
    return ((np.asarray(lon) - LON0) * 111.195 * np.cos(np.deg2rad(LAT0)),
            (np.asarray(lat) - LAT0) * 111.195)


def period(time):
    """Calendar-date day 06--23; night 00--06 plus 23--24. Not a service day."""
    return np.where(time.dt.hour.between(6, 22), 'day', 'night')


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, default=lambda x:
        x.item() if hasattr(x, 'item') else str(x)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    total = y.sum()
    return {'wape': float(abs(y - prediction).sum() / total) if total > 0 else np.nan,
            'bias': float((prediction - y).sum() / total) if total > 0 else np.nan,
            'mae': float(abs(y - prediction).mean()), 'observed': float(total), 'n': len(y)}


def chord_distance(ox, oy, dx, dy, sx, sy):
    vx, vy = dx - ox, dy - oy
    den = vx * vx + vy * vy
    u = np.divide((sx - ox) * vx + (sy - oy) * vy, den,
                  out=np.zeros_like(den, dtype=float), where=den > 0)
    delta = np.hypot(ox + np.clip(u, 0, 1) * vx - sx,
                     oy + np.clip(u, 0, 1) * vy - sy)
    return u, delta
