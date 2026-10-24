"""Shared definitions. Local intermediates must never contain user/bicycle IDs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
CACHE = ROOT / 'code' / 'dati_esterni'
TEXT = ROOT.parent / 'paper-vehicle-sharing-text'
for p in [OUT, CACHE, TEXT / 'figures', TEXT / 'tables']:
    p.mkdir(parents=True, exist_ok=True)
SEED = 20261016
LAT0, LON0 = 44.4949, 11.3426
EXCLUDED_SITES = frozenset({'Laura_Bassi', 'Massarenti', 'Mazzini', 'Murri', 'Orti', 'Saragozza'})
MIN_RIDE_SECONDS = 30

def ride_duration_mask(seconds):
    """Keep finite recorded durations >=30 s; equality at the cutoff is included."""
    seconds=np.asarray(seconds,dtype=float)
    return np.isfinite(seconds)&(seconds>=MIN_RIDE_SECONDS)

def xy(lat, lon):
    """Local equirectangular kilometres, adequate for descriptive urban grids."""
    return ((np.asarray(lon)-LON0)*111.195*np.cos(np.deg2rad(LAT0)),
            (np.asarray(lat)-LAT0)*111.195)

def clock(dt):
    """06:00 service-day boundary; night 23:00--05:59, tagged to preceding day."""
    return ((dt-pd.Timedelta(hours=6)).dt.strftime('%Y-%m-%d'),
            np.where((dt.dt.hour>=6)&(dt.dt.hour<23), 'day', 'night'))

def dump(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)))

def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    return {'mae':float(np.mean(abs(y-p))), 'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'wape':float(np.sum(abs(y-p))/max(np.sum(y),1)),
            'bias':float(np.sum(p-y)/max(np.sum(y),1)), 'n':len(y)}
