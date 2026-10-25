"""Paths, fixed protocol and arithmetic for the standalone sharing study."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / 'dataset'
CACHE = ROOT / 'cache'
OUT = ROOT / 'results'
TEXT = REPO / 'sharing-text'
SEED = 20261025
REPLICATES = 2000
LAT0, LON0 = 44.4949, 11.3426
FEATURES = ['Context', 'Citywide', 'Endpoints', 'Geometry']
EXCLUDED_BIKE = {'Laura_Bassi', 'Massarenti', 'Mazzini', 'Murri', 'Orti', 'Saragozza'}
DST_DATES = ['2023-03-26', '2023-10-29', '2024-03-31', '2024-10-27']
HOLIDAYS = {f'{year}-{day}' for year in (2023, 2024) for day in
            ('01-01', '01-06', '04-25', '05-01', '06-02', '08-15', '10-04', '11-01', '12-08', '12-25', '12-26')}
HOLIDAYS |= {'2023-04-09', '2023-04-10', '2024-03-31', '2024-04-01'}


def directories():
    for path in [CACHE, OUT, TEXT / 'figures', TEXT / 'tables', CACHE / 'sources']:
        path.mkdir(parents=True, exist_ok=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x:
        x.item() if hasattr(x, 'item') else str(x)) + '\n')


def xy(lat, lon):
    return ((np.asarray(lon) - LON0) * 111.195 * np.cos(np.deg2rad(LAT0)),
            (np.asarray(lat) - LAT0) * 111.195)


def period(time, mode):
    """Bike service date starts at 06; vehicle periods use calendar dates."""
    if time.dt.tz is not None:
        time = time.dt.tz_localize(None)
    date = time - pd.Timedelta(hours=6) if mode == 'bike' else time
    return date.dt.strftime('%Y-%m-%d'), np.where(time.dt.hour.between(6, 22), 'day', 'night')


def calendar(mode):
    if mode == 'bike':
        time = pd.Series(pd.date_range('2023-01-01 06:00', '2025-01-01 06:00',
                         freq='h', tz='Europe/Rome', inclusive='left')).dt.tz_localize(None)
    else:
        time = pd.Series(pd.date_range('2023-01-01', '2025-01-01', freq='h', inclusive='left'))
    date, regime = period(time, mode)
    return pd.DataFrame({'date': date, 'regime': regime}).groupby(['date', 'regime']).size().rename('exposure').reset_index()


def chord_distance(ox, oy, dx, dy, sx, sy):
    vx, vy = dx - ox, dy - oy
    den = vx * vx + vy * vy
    u = np.divide((sx - ox) * vx + (sy - oy) * vy, den,
                  out=np.zeros_like(den, dtype=float), where=den > 0)
    return u, np.hypot(ox + np.clip(u, 0, 1) * vx - sx, oy + np.clip(u, 0, 1) * vy - sy)


def metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    total = observed.sum()
    ae = np.abs(predicted - observed).sum()
    signed = (predicted - observed).sum()
    return dict(observed=float(total), ae=float(ae), signed=float(signed), n=len(observed),
                wape=float(ae / total) if total > 0 else np.nan,
                bias=float(signed / total) if total > 0 else np.nan)
