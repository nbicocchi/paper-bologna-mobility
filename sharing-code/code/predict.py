"""Two distinct missing-sensor experiments; every estimator is fitted anew on 2023."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import CACHE, OUT, FEATURES, SEED, dump

BASE = ['x', 'y', 'radius', 'dow', 'sin_year', 'cos_year', 'night', 'temperature_mean_c', 'precipitation_mm']
HISTORY_KEYS = ['site', 'direction', 'regime']
HISTORY = ['log_site_rate', 'log_weekday_rate']


def feature_sets(mode, scenario):
    base = BASE + (['direction'] if mode == 'bike' else ['holiday'])
    if scenario == 'known':
        base += HISTORY
    global_features = ['global_trips'] + (['electric_share'] if mode == 'bike' else [])
    return dict(zip(FEATURES, [base, base + global_features,
        base + global_features + ['starts_500m', 'ends_500m'],
        base + global_features + ['starts_500m', 'ends_500m', 'chord_100', 'chord_250']]))


def history_tables(train):
    """Frozen, exposure-weighted local profiles from historical counts only."""
    if not train.year.eq(2023).all():
        raise ValueError('History must contain 2023 observations only')
    tables = {}
    for name, keys in [('site', ['site', 'direction']), ('regime', HISTORY_KEYS), ('weekday', HISTORY_KEYS + ['dow'])]:
        sums = train.groupby(keys)[['count', 'exposure']].sum()
        tables[name] = (sums['count'] / sums.exposure).rename('historical_rate')
    return tables


def add_history(frame, tables):
    result = frame.copy()
    rates = {}
    for name, table in tables.items():
        keys = table.index.names
        values = table.reindex(pd.MultiIndex.from_frame(result[list(keys)])).to_numpy()
        rates[name] = values
    if np.isnan(rates['site']).any():
        raise ValueError('Known-site test includes an unseen location/direction')
    result['history_regime_fallback'] = np.isnan(rates['regime'])
    rates['regime'] = np.where(np.isnan(rates['regime']), rates['site'], rates['regime'])
    result['history_weekday_fallback'] = np.isnan(rates['weekday'])
    rates['weekday'] = np.where(np.isnan(rates['weekday']), rates['regime'], rates['weekday'])
    for column, name in zip(HISTORY, ['regime', 'weekday']):
        result[column] = np.log(np.maximum(rates[name], .01))
    return result


def estimator(family, columns):
    if family == 'Boosting':
        return HistGradientBoostingRegressor(loss='poisson', max_iter=120, max_leaf_nodes=7,
            learning_rate=.06, min_samples_leaf=40, l2_regularization=10,
            early_stopping=False, random_state=SEED)
    categories = [c for c in ['dow', 'night', 'direction'] if c in columns]
    numeric = [c for c in columns if c not in categories]
    transform = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories),
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric)])
    return make_pipeline(transform, PoissonRegressor(alpha=1., max_iter=1000, tol=1e-7))


def split_masks(d, sites, mode, scenario):
    if scenario == 'known':
        yield 'all_2023', d.year.eq(2023), d.year.eq(2024), []
    elif mode == 'bike':
        for site in sites.site:
            yield str(site), d.year.eq(2023) & d.site.ne(site), d.site.eq(site), []
    else:
        for fold in range(5):
            test_sites = sites.loc[sites.fold.eq(fold)]
            candidates = sites.loc[sites.fold.ne(fold)]
            distance = cKDTree(test_sites[['x', 'y']]).query(candidates[['x', 'y']])[0]
            train_sites = candidates.loc[distance > .25, 'site']
            excluded = candidates.loc[distance <= .25, 'site'].tolist()
            yield str(fold), d.year.eq(2023) & d.site.isin(train_sites), d.fold.eq(fold), excluded


def run(mode):
    d = pd.read_parquet(CACHE / f'{mode}-model-input.parquet')
    sites = pd.read_csv(OUT / f'{mode}-sites.csv')
    predictions, design = [], []
    for scenario in ['new', 'known']:
        specs = feature_sets(mode, scenario)
        for split, train_mask, test_mask, excluded in split_masks(d, sites, mode, scenario):
            train, test = d.loc[train_mask].copy(), d.loc[test_mask].copy()
            assert train.year.eq(2023).all()
            assert not (train_mask & test_mask).any()
            if scenario == 'new':
                assert set(train.site).isdisjoint(test.site)
            else:
                assert set(test.site).issubset(train.site) and test.year.eq(2024).all()
                tables = history_tables(train)
                for name, table in tables.items():
                    table.to_csv(OUT / f'{mode}-history-{name}.csv')
                train, test = add_history(train, tables), add_history(test, tables)
            record = dict(mode=mode, scenario=scenario, split=split, train_years=sorted(train.year.unique().tolist()),
                train_sites=sorted(train.site.unique().tolist()), test_sites=sorted(test.site.unique().tolist()),
                train_rows=len(train), test_rows=len(test), buffer_excluded=excluded, features=specs,
                train_date_min=train.date.min(), train_date_max=train.date.max(),
                test_date_min=test.date.min(), test_date_max=test.date.max())
            if scenario == 'known':
                record['test_history_regime_fallback_rows'] = int(test.history_regime_fallback.sum())
                record['test_history_weekday_fallback_rows'] = int(test.history_weekday_fallback.sum())
            if mode == 'car' and scenario == 'new':
                a = sites.loc[sites.site.isin(train.site), ['x', 'y']]
                b = sites.loc[sites.site.isin(test.site), ['x', 'y']]
                record['minimum_distance_km'] = float(cKDTree(a).query(b)[0].min())
                assert record['minimum_distance_km'] > .25
            design.append(record)
            print(mode, scenario, split, 'train', len(train), 'test', len(test), flush=True)
            for family in ['Boosting', 'Poisson', 'Scaling']:
                for feature in (FEATURES if family != 'Scaling' else ['Geometry']):
                    if family == 'Scaling':
                        denominator = np.dot(train.chord_250, train.exposure)
                        if denominator <= 0:
                            raise ValueError('No sharing exposure for scaling')
                        rate = test.chord_250.to_numpy() * train['count'].sum() / denominator
                    else:
                        columns = specs[feature]
                        model = estimator(family, columns)
                        weights = {'sample_weight': train.exposure} if family == 'Boosting' else {'poissonregressor__sample_weight': train.exposure}
                        model.fit(train[columns], train.rate, **weights)
                        rate = model.predict(test[columns])
                    z = test[['site', 'date', 'year', 'regime', 'direction', 'exposure', 'count', 'block']].copy()
                    z['prediction'] = rate * z.exposure
                    if not np.isfinite(z.prediction).all() or (z.prediction < 0).any():
                        raise ValueError('Invalid prediction')
                    z['family'], z['feature'], z['scenario'] = family, feature, scenario
                    predictions.append(z)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(OUT / f'{mode}-predictions.parquet', index=False)
    dump(OUT / f'{mode}-prediction-design.json', dict(seed=SEED, splits=design,
        target='Complete-period count; learn hourly rate weighted by exposure',
        history='Known only: log 2023 hourly rate by site/direction/regime and by those keys plus weekday; floor .01; missing weekday falls back to regime, missing regime to site/direction',
        history_used_by='Both Context and every sharing model; none in the new-location experiment',
        scaling='Single training-total expansion factor in both scenarios; no local historical features',
        boosting=dict(iterations=120, leaves=7, learning_rate=.06, min_leaf=40, l2=10),
        poisson=dict(alpha=1., max_iter=1000, tol=1e-7),
        temporal_inputs='Contemporaneous sharing, calendar and weather; no 2024 counts or refitting'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['bike', 'car'])
    args = parser.parse_args()
    for mode in ([args.mode] if args.mode else ['bike', 'car']):
        run(mode)
