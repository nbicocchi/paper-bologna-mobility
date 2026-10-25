"""Matched within-scenario sharing contrasts and spatial-unit sensitivity summaries."""
import numpy as np
import pandas as pd
from common import OUT, FEATURES, SEED, REPLICATES, metrics

KEYS = ['mode', 'scenario', 'year', 'family', 'feature']


def aggregate(sites):
    rows = []
    for keys, g in sites.groupby(KEYS, sort=True):
        total = g.observed.sum()
        rows.append(dict(zip(KEYS, keys), sites=len(g), positive_sites=int(g.observed.gt(0).sum()),
            units=g.unit.nunique(), observed=total, n=int(g.n.sum()), pooled_wape=g.ae.sum() / total,
            pooled_bias=g.signed.sum() / total, macro_wape=g.wape.mean(), median_wape=g.wape.median(),
            mae=g.ae.sum() / g.n.sum()))
    return pd.DataFrame(rows)


def paired_contrast(frame, feature, comparator, rng):
    left = frame.loc[frame.feature.eq(feature)].set_index('site')
    right = frame.loc[frame.feature.eq(comparator)].set_index('site').reindex(left.index)
    if right.ae.isna().any() or not np.array_equal(left.observed, right.observed) or not np.array_equal(left.n, right.n):
        raise ValueError('Unmatched test observations')
    units = pd.DataFrame({'unit': left.unit, 'difference': left.ae - right.ae,
                          'observed': left.observed}).groupby('unit').sum()
    indices = rng.integers(0, len(units), size=(REPLICATES, len(units)))
    denominator = units.observed.to_numpy()[indices].sum(axis=1)
    boot = np.divide(units.difference.to_numpy()[indices].sum(axis=1), denominator,
                     out=np.full(REPLICATES, np.nan), where=denominator > 0)
    positive = left.observed > 0
    return dict(delta_pp=100 * units.difference.sum() / units.observed.sum(),
        lo_pp=100 * np.nanquantile(boot, .025), hi_pp=100 * np.nanquantile(boot, .975), units=len(units),
        improved_sites=int((left.ae[positive] < right.ae[positive]).sum()), positive_sites=int(positive.sum()),
        improved_fraction=float((left.ae[positive] < right.ae[positive]).mean()),
        valid_bootstrap_replicates=int(np.isfinite(boot).sum()))


def contrasts(sites):
    rows = []
    for keys, frame in sites.loc[sites.family.ne('Scaling')].groupby(KEYS[:4]):
        for i, feature in enumerate(FEATURES[1:], 1):
            for kind, comparator in [('vs_context', 'Context'), ('increment', FEATURES[i - 1])]:
                # Same units drawn for all matched representations and both scenarios.
                rng = np.random.default_rng(SEED + keys[2] + (100 if keys[0] == 'car' else 0))
                rows.append(dict(zip(KEYS[:4], keys), feature=feature, comparator=comparator, kind=kind,
                                 **paired_contrast(frame, feature, comparator, rng)))
    return pd.DataFrame(rows)


def persistence(sites):
    rows = []
    new = sites.loc[sites.scenario.eq('new') & sites.family.ne('Scaling') & sites.observed.gt(0)]
    for (mode, family), frame in new.groupby(['mode', 'family']):
        for feature in FEATURES[1:]:
            gains = {}
            for year in [2023, 2024]:
                wide = frame.loc[frame.year.eq(year)].pivot(index='site', columns='feature', values='ae')
                gains[year] = wide.Context - wide[feature]
            common = pd.concat(gains, axis=1, join='inner')
            early, late = common[2023] > 0, common[2024] > 0
            rows.append(dict(mode=mode, family=family, feature=feature, common_positive_sites=len(common),
                improve_both=int((early & late).sum()), improve_2023_only=int((early & ~late).sum()),
                improve_2024_only=int((~early & late).sum()), improve_neither=int((~early & ~late).sum())))
    return pd.DataFrame(rows)


def main():
    rows, reconciliation, period_scores = [], [], []
    for mode in ['bike', 'car']:
        pred = pd.read_parquet(OUT / f'{mode}-predictions.parquet')
        for keys, g in pred.groupby(['scenario', 'year', 'family', 'feature', 'site']):
            scenario, year, family, feature, site = keys
            rows.append(dict(mode=mode, scenario=scenario, year=year, family=family, feature=feature,
                site=str(site), unit=str(site) if mode == 'bike' else g.block.iloc[0], **metrics(g['count'], g.prediction)))
        for keys, g in pred.groupby(['scenario', 'year', 'family', 'feature']):
            reconciliation.append(dict(mode=mode, **dict(zip(KEYS[1:], keys)), **metrics(g['count'], g.prediction)))
        for keys, g in pred.groupby(['scenario', 'year', 'family', 'feature', 'regime']):
            period_scores.append(dict(mode=mode, **dict(zip(KEYS[1:] + ['regime'], keys)),
                                      **metrics(g['count'], g.prediction)))
        weekly = pred.loc[pred.year.eq(2024)].copy()
        weekly['week'] = pd.to_datetime(weekly.date).dt.to_period('W').dt.start_time
        weekly = weekly.groupby(['scenario', 'family', 'feature', 'site', 'direction', 'week'])[
            ['count', 'prediction', 'exposure']].sum().reset_index()
        weekly.to_csv(OUT / f'{mode}-weekly-2024.csv', index=False)
    sites = pd.DataFrame(rows)
    summary = aggregate(sites)
    direct = pd.DataFrame(reconciliation).merge(summary, on=KEYS, validate='one_to_one')
    for metric, pooled in [('wape', 'pooled_wape'), ('bias', 'pooled_bias')]:
        if not np.allclose(direct[metric], direct[pooled], rtol=1e-10, atol=1e-10):
            raise ValueError('Period-to-site reconciliation failed')
    direct.to_csv(OUT / 'reconciliation.csv', index=False)
    pd.DataFrame(period_scores).to_csv(OUT / 'period-scores.csv', index=False)
    products = {'sites': sites, 'summary': summary, 'contrasts': contrasts(sites), 'persistence': persistence(sites)}
    for name, product in products.items():
        product.to_csv(OUT / f'{name}.csv', index=False)
    # Same observations in both 2024 scenarios, not just the same locations.
    a = sites.loc[sites.year.eq(2024)]
    matched = a.pivot(index=['mode', 'site', 'family', 'feature'], columns='scenario', values=['observed', 'n'])
    if matched.isna().any().any() or not np.array_equal(matched[('observed', 'new')], matched[('observed', 'known')]) or not np.array_equal(matched[('n', 'new')], matched[('n', 'known')]):
        raise ValueError('The two scenarios have different 2024 targets')
    print('Reconciled', len(summary), 'configurations from fresh period predictions.', flush=True)


if __name__ == '__main__':
    main()
