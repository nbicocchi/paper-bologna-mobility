"""Fixed model/feature comparisons under buffered, grouped spatial validation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cars_common import ROOT, OUT, CACHE, TEXT, SEED, HOLIDAYS, metrics, dump
from cars_analyse import plt, save

BASE = ['x', 'y', 'radius', 'dow', 'sin_year', 'cos_year', 'night', 'holiday',
        'temperature_mean_c', 'precipitation_mm']
SHARING = ['global_trips', 'starts_500m', 'ends_500m', 'chord_100', 'chord_250']
SETS = {'base': BASE, 'global': BASE + SHARING[:1],
        'local': BASE + SHARING[:3], 'geometry': BASE + SHARING}


class UniformScale:
    def fit(self, X, y, sample_weight):
        numerator = np.dot(y, sample_weight)
        denominator = np.dot(np.asarray(X)[:, 0], sample_weight)
        if denominator <= 0:
            raise ValueError('Uniform expansion has no training sharing exposure')
        self.factor = numerator / denominator
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] * self.factor


class Poisson:
    def fit(self, X, y, sample_weight):
        cats = ['dow', 'night']
        numeric = [c for c in X if c not in cats]
        self.transform = ColumnTransformer([
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
            ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric)])
        transformed = self.transform.fit_transform(X)
        self.model = PoissonRegressor(alpha=1., max_iter=1000, tol=1e-7)
        self.model.fit(transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(self.transform.transform(X))


def estimator(name):
    if name == 'scaling': return UniformScale()
    if name.startswith('poisson'): return Poisson()
    return HistGradientBoostingRegressor(loss='poisson', max_iter=120, max_leaf_nodes=7,
        learning_rate=.06, min_samples_leaf=40, l2_regularization=10,
        early_stopping=False, random_state=SEED)


def prediction():
    sites = pd.read_csv(OUT / 'sites.csv')
    d = pd.read_parquet(CACHE / 'counts.parquet')
    d = d.merge(pd.read_parquet(CACHE / 'features.parquet'), on=['site', 'date', 'regime'], validate='one_to_one')
    short = pd.read_parquet(CACHE / 'features-short.parquet').rename(columns={c: c + '_short' for c in SHARING})
    d = d.merge(short, on=['site', 'date', 'regime'], validate='one_to_one')
    d = d.merge(sites, on='site', validate='many_to_one')
    weather = pd.read_csv(ROOT / 'dataset/meteo-bologna-ilmeteo-2019-2025.csv', sep=';')
    weather = weather[['date', 'temperature_mean_c', 'precipitation_mm']].drop_duplicates('date')
    d = d.merge(weather, on='date', how='left', validate='many_to_one')
    dates = pd.to_datetime(d.date)
    d['year'] = dates.dt.year
    d['dow'] = dates.dt.weekday
    d['sin_year'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    d['cos_year'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    d['night'] = d.regime.eq('night').astype(int)
    d['holiday'] = d.date.isin(HOLIDAYS).astype(int)
    d['rate'] = d['count'] / d.exposure
    for c in SHARING + [c + '_short' for c in SHARING]:
        d[c] /= d.exposure
    d.to_parquet(CACHE / 'prediction-input.parquet', index=False)
    specs = {f'{family}_{feature}': columns for family in ('boosting', 'poisson') for feature, columns in SETS.items()}
    specs['scaling'] = ['chord_250']
    for family in ('boosting', 'poisson'):
        specs[f'{family}_geometry_short'] = BASE + [c + '_short' for c in SHARING]
    frames, folds = [], []
    for fold in range(5):
        test_sites = sites.loc[sites.fold.eq(fold)]
        candidates = sites.loc[sites.fold.ne(fold)]
        distance = cKDTree(test_sites[['x', 'y']]).query(candidates[['x', 'y']])[0]
        train_sites = candidates.loc[distance > .25, 'site']
        train = d.year.eq(2023) & d.site.isin(train_sites)
        test = d.fold.eq(fold)
        record = {'fold': fold, 'train_sites': train_sites.tolist(), 'test_sites': test_sites.site.tolist(),
                  'buffer_excluded_sites': candidates.loc[distance <= .25, 'site'].tolist(),
                  'minimum_train_test_distance_km': float(distance[distance > .25].min()),
                  'train_periods': int(train.sum()), 'test_periods': int(test.sum())}
        folds.append(record)
        print('Fold', fold, 'training sites', len(train_sites), 'test sites', len(test_sites), flush=True)
        for name, cols in specs.items():
            model = estimator(name)
            model.fit(d.loc[train, cols], d.loc[train, 'rate'], sample_weight=d.loc[train, 'exposure'])
            z = d.loc[test, ['site', 'date', 'year', 'regime', 'exposure', 'count', 'cluster', 'block', 'fold']].copy()
            z['prediction'] = model.predict(d.loc[test, cols]) * z.exposure
            if not np.isfinite(z.prediction).all():
                raise ValueError(f'Nonfinite predictions in {name}, fold {fold}')
            z['model'] = name
            frames.append(z)
    pred = pd.concat(frames, ignore_index=True)
    pred.to_parquet(OUT / 'predictions.parquet', index=False)
    dump('prediction-design.json', {'seed': SEED, 'train_year': 2023, 'folds': folds,
        'features': specs, 'buffer_km': .25, 'cluster_linkage_km': .1,
        'block_km': 1, 'fold_formula': '(floor(cluster_mean_x) + 2*floor(cluster_mean_y)) mod 5',
        'boosting': {'loss': 'poisson', 'iterations': 120, 'leaves': 7, 'learning_rate': .06,
                     'min_leaf': 40, 'l2': 10, 'early_stopping': False},
        'poisson': {'alpha': 1, 'max_iter': 1000, 'tol': 1e-7},
        'weather_missing_rows': int(d[['temperature_mean_c', 'precipitation_mm']].isna().any(axis=1).sum())})
    return pred


def scores(pred):
    site_rows, summaries = [], []
    for (year, model, site), g in pred.groupby(['year', 'model', 'site']):
        site_rows.append({'year': year, 'model': model, 'site': site, 'block': g.block.iloc[0],
                          'cluster': g.cluster.iloc[0], 'fold': g.fold.iloc[0], **metrics(g['count'], g.prediction)})
    site_scores = pd.DataFrame(site_rows)
    site_scores.to_csv(OUT / 'site-scores.csv', index=False)
    for (year, model), g in pred.groupby(['year', 'model']):
        s = site_scores.loc[site_scores.year.eq(year) & site_scores.model.eq(model)]
        cluster_wape = [metrics(v['count'], v.prediction)['wape'] for _, v in g.groupby('cluster')]
        summaries.append({'year': year, 'model': model, 'sites': g.site.nunique(),
                          'sites_positive_total': int(s.observed.gt(0).sum()),
                          'median_wape': s.wape.median(), 'macro_wape': s.wape.mean(),
                          'cluster_macro_wape': np.nanmean(cluster_wape),
                          **{'pooled_' + k: v for k, v in metrics(g['count'], g.prediction).items()}})
    pd.DataFrame(summaries).to_csv(OUT / 'prediction-summary.csv', index=False)
    rng = np.random.default_rng(SEED)
    bootstrap = []
    for year in (2023, 2024):
        s = site_scores.loc[site_scores.year.eq(year)]
        pivot = s.pivot(index=['site', 'block'], columns='model', values='wape').dropna()
        block_count = pivot.index.get_level_values('block').nunique()
        # Identical predictions must have identical intervals, not Monte Carlo differences.
        indices = rng.integers(0, block_count, size=(2000, block_count))
        for model in pivot:
            if model.endswith('_base'): continue
            baseline = 'poisson_base' if model.startswith('poisson') else 'boosting_base'
            delta = (pivot[model] - pivot[baseline]).rename('delta').reset_index()
            by_block = delta.groupby('block').delta.agg(['sum', 'size'])
            draws = by_block['sum'].to_numpy()[indices].sum(axis=1) / by_block['size'].to_numpy()[indices].sum(axis=1)
            bootstrap.append({'year': year, 'model': model, 'baseline': baseline,
                              'delta_macro_wape': delta.delta.mean(), 'lo': np.quantile(draws, .025),
                              'hi': np.quantile(draws, .975), 'blocks': len(by_block),
                              'fraction_sites_improved': float(delta.delta.lt(0).mean())})
    pd.DataFrame(bootstrap).to_csv(OUT / 'paired-bootstrap.csv', index=False)
    period_rows = []
    for (year, model, regime), g in pred.groupby(['year', 'model', 'regime']):
        period_rows.append({'year': year, 'model': model, 'regime': regime, **metrics(g['count'], g.prediction)})
    pd.DataFrame(period_rows).to_csv(OUT / 'period-scores.csv', index=False)
    reliability = []
    for year, g in site_scores.loc[site_scores.model.eq('boosting_base')].groupby('year'):
        reliability.append({'year': int(year), 'devices': len(g), 'zero_total_devices': int(g.observed.eq(0).sum()),
                            'under_100_annual_detections': int(g.observed.lt(100).sum()),
                            'minimum_positive_total': float(g.loc[g.observed.gt(0), 'observed'].min()),
                            'zero_total_site_ids': g.loc[g.observed.eq(0), 'site'].tolist(),
                            'under_100_site_ids': g.loc[g.observed.lt(100), 'site'].tolist()})
    dump('counter-reliability.json', reliability)
    return site_scores


def figures(pred, site_scores):
    models = [f'{f}_{s}' for f in ('boosting', 'poisson') for s in SETS]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    labels = ['Boost · no sharing', 'Boost · citywide', 'Boost · endpoints', 'Boost · geometry',
              'Poisson · no sharing', 'Poisson · citywide', 'Poisson · endpoints', 'Poisson · geometry', 'Scaling']
    for ax, year in zip(axes, (2023, 2024)):
        data = [site_scores.loc[site_scores.year.eq(year) & site_scores.model.eq(m), 'wape'].dropna() * 100 for m in models + ['scaling']]
        ax.boxplot(data, orientation='horizontal', tick_labels=labels, showfliers=False, widths=.6)
        ax.set(title=str(year), xlabel='Detector WAPE (%)', xscale='symlog', xlim=(0, 1000))
    axes[0].invert_yaxis()
    fig.tight_layout(); save(fig, 'prediction')
    # Diagnostic examples selected by 2023 observed mean-flow quartiles, not by test error.
    g = pred.loc[pred.year.eq(2023) & pred.model.eq('boosting_base')].copy()
    observed = g.groupby('site').apply(lambda z: z['count'].sum() / z.exposure.sum(), include_groups=False).sort_values()
    selected = [int(observed.index[round(q * (len(observed) - 1))]) for q in (.125, .375, .625, .875)]
    dump('diagnostic-sites.json', {'selection': '2023 observed mean-rate quantiles 0.125, 0.375, 0.625, 0.875', 'sites': selected})
    z = pred.loc[pred.year.eq(2024) & pred.site.isin(selected)].copy()
    z['week'] = pd.to_datetime(z.date).dt.to_period('W').dt.start_time
    weekly = z.groupby(['site', 'model', 'week'])[['count', 'prediction', 'exposure']].sum().reset_index()
    weekly['observed_rate'] = weekly['count'] / weekly.exposure
    weekly['predicted_rate'] = weekly.prediction / weekly.exposure
    weekly.to_csv(OUT / 'weekly-2024.csv', index=False)
    meta = pd.read_csv(OUT / 'sites.csv', index_col='site')
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    for ax, site in zip(axes.flat, selected):
        b = weekly.loc[weekly.site.eq(site)]
        for model, label in [('boosting_base', 'Boost no sharing'), ('boosting_geometry', 'Boost geometry'),
                             ('poisson_base', 'Poisson no sharing'), ('poisson_geometry', 'Poisson geometry')]:
            a = b.loc[b.model.eq(model)].set_index('week').asfreq('W-MON')
            if model == 'boosting_base': ax.plot(a.index, a.observed_rate, color='black', label='Observed', lw=1.4)
            ax.plot(a.index, a.predicted_rate, label=label, lw=.7)
        ax.set(title=f'{site} · {meta.loc[site, "street"].strip().title()}', ylabel='Vehicles / observed hour')
        ax.tick_params(axis='x', rotation=30, labelsize=6)
        ax.title.set_fontsize(7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1)); save(fig, 'observed-predicted')
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E6AB02']
    for fold, color in enumerate(colors):
        s = meta.loc[meta.fold.eq(fold)]
        ax.scatter(s.x, s.y, c=color, s=6, label=f'Fold {fold + 1}', alpha=.6)
    ax.set(xlabel='East of reference point (km)', ylabel='North of reference point (km)', aspect='equal')
    ax.legend(frameon=False, fontsize=6, markerscale=1.5)
    fig.tight_layout(); save(fig, 'folds')


def main():
    pred = prediction()
    site_scores = scores(pred)
    figures(pred, site_scores)


if __name__ == '__main__':
    main()
