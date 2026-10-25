"""Reconcile existing experiments and calculate new within-mode contrasts."""
import argparse
import json
import platform
import shutil
from importlib.metadata import version

import numpy as np
import pandas as pd

from common import (ROOT, REPO, OUT, INPUT, TEXT, SOURCES, MAPPING, FEATURES,
                    SEED, REPLICATES, sha, year_from_split)
from distributions import freeze_distributions


def snapshot(frozen=False):
    INPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, relative in SOURCES.items():
        source, target = REPO / relative, INPUT / name
        if not frozen:
            shutil.copyfile(source, target)
        records.append({'snapshot': name, 'source': relative, 'sha256': sha(target)})
    return records


def load_sites():
    bike = pd.read_csv(INPUT / 'bike-sites.csv')
    rows = []
    for row in bike.to_dict('records'):
        for model, (family, feature) in MAPPING['bike'].items():
            observed = row['observed_count']
            rows.append(dict(mode='bike', year=year_from_split(row['split']),
                             site=str(row['site']), unit=str(row['site']),
                             family=family, feature=feature, source_model=model,
                             observed=observed, n=row['directional_periods'],
                             ae=row[f'{model}_wape'] * observed,
                             signed=row[f'{model}_bias'] * observed))
    car = pd.read_csv(INPUT / 'car-sites.csv')
    for row in car.to_dict('records'):
        if row['model'] not in MAPPING['car']:
            continue
        family, feature = MAPPING['car'][row['model']]
        # MAE*n preserves absolute errors at all-zero devices too.
        rows.append(dict(mode='car', year=int(row['year']), site=str(row['site']),
                         unit=str(row['block']), family=family, feature=feature,
                         source_model=row['model'], observed=row['observed'], n=row['n'],
                         ae=row['mae'] * row['n'],
                         signed=row['bias'] * row['observed'] if row['observed'] > 0
                         else row['mae'] * row['n']))
    sites = pd.DataFrame(rows)
    sites['wape'] = sites['ae'] / sites['observed'].replace(0, np.nan)
    sites['bias'] = sites['signed'] / sites['observed'].replace(0, np.nan)
    return sites


def aggregate(sites):
    rows = []
    for keys, frame in sites.groupby(['mode', 'year', 'family', 'feature'], sort=True):
        total = frame.observed.sum()
        rows.append(dict(zip(['mode', 'year', 'family', 'feature'], keys),
                         sites=len(frame), positive_sites=int((frame.observed > 0).sum()),
                         units=frame.unit.nunique(), observed=total, n=int(frame.n.sum()),
                         pooled_wape=frame.ae.sum() / total,
                         pooled_bias=frame.signed.sum() / total,
                         macro_wape=frame.wape.mean(), median_wape=frame.wape.median(),
                         mae=frame.ae.sum() / frame.n.sum()))
    return pd.DataFrame(rows)


def reconcile(summary):
    checks = []
    for mode in MAPPING:
        source = pd.read_csv(INPUT / f'{mode}-summary.csv')
        if mode == 'bike':
            source['year'] = source['split'].map(year_from_split)
        for row in summary[summary['mode'] == mode].itertuples():
            model = next(m for m, pair in MAPPING[mode].items()
                         if pair == (row.family, row.feature))
            original = source[(source.year == row.year) & (source.model == model)].iloc[0]
            for metric in ['pooled_wape', 'pooled_bias', 'macro_wape']:
                delta = abs(getattr(row, metric) - original[metric])
                if not np.isclose(getattr(row, metric), original[metric], rtol=1e-10, atol=1e-10):
                    raise ValueError(f'Upstream reconciliation failed: {mode} {model} {metric}')
                checks.append(dict(mode=mode, year=row.year, model=model,
                                   metric=metric, absolute_difference=delta))
    return pd.DataFrame(checks)


def paired_contrast(frame, feature, comparator, rng):
    left = frame[frame.feature == feature].set_index('site')
    right = frame[frame.feature == comparator].set_index('site').reindex(left.index)
    if right.ae.isna().any() or not np.array_equal(left.observed, right.observed):
        raise ValueError('Unmatched observations in feature comparison')
    units = pd.DataFrame({'unit': left.unit, 'difference': left.ae - right.ae,
                          'observed': left.observed}).groupby('unit').sum()
    sample = rng.integers(0, len(units), size=(REPLICATES, len(units)))
    den = units.observed.to_numpy()[sample].sum(axis=1)
    boot = np.divide(units.difference.to_numpy()[sample].sum(axis=1), den,
                     out=np.full(REPLICATES, np.nan), where=den > 0)
    positive = left.observed > 0
    return dict(delta_pp=100 * units.difference.sum() / units.observed.sum(),
                lo_pp=100 * np.nanquantile(boot, .025),
                hi_pp=100 * np.nanquantile(boot, .975), units=len(units),
                improved_fraction=float((left.ae[positive] < right.ae[positive]).mean()),
                positive_sites=int(positive.sum()),
                valid_bootstrap_replicates=int(np.isfinite(boot).sum()))


def contrasts(sites):
    rows = []
    for keys, frame in sites[sites.family != 'Scaling'].groupby(['mode', 'year', 'family']):
        for i, feature in enumerate(FEATURES[1:], 1):
            for kind, comparator in [('vs_context', 'Context'), ('increment', FEATURES[i - 1])]:
                # Identical resampled units across learners/features in a mode/year.
                rng = np.random.default_rng(SEED + keys[1] + (100 if keys[0] == 'car' else 0))
                result = paired_contrast(frame, feature, comparator, rng)
                rows.append(dict(zip(['mode', 'year', 'family'], keys), feature=feature,
                                 comparator=comparator, kind=kind, **result))
    return pd.DataFrame(rows)


def persistence(sites, contrast):
    rows = []
    for (mode, family), frame in sites[sites.family != 'Scaling'].groupby(['mode', 'family']):
        for feature in FEATURES[1:]:
            gains = {}
            for year in [2023, 2024]:
                f = frame[(frame.year == year) & (frame.observed > 0)]
                wide = f.pivot(index='site', columns='feature', values='ae')
                gains[year] = wide['Context'] - wide[feature]
            common = pd.concat(gains, axis=1, join='inner')
            early, late = common[2023] > 0, common[2024] > 0
            c = contrast[(contrast['mode'] == mode) & (contrast.family == family)
                         & (contrast.feature == feature) & (contrast.kind == 'vs_context')]
            change = c.set_index('year').delta_pp
            rows.append(dict(mode=mode, family=family, feature=feature,
                             common_positive_sites=len(common),
                             improve_both=int((early & late).sum()),
                             improve_2023_only=int((early & ~late).sum()),
                             improve_2024_only=int((~early & late).sum()),
                             improve_neither=int((~early & ~late).sum()),
                             delta_2023_pp=change.loc[2023], delta_2024_pp=change.loc[2024],
                             advantage_change_pp=change.loc[2024] - change.loc[2023]))
    return pd.DataFrame(rows)


def denominator_diagnostics(sites):
    rows = []
    for (mode, year), frame in sites[(sites.family == 'Boosting') &
                                    (sites.feature == 'Context')].groupby(['mode', 'year']):
        positive = frame[frame.observed > 0]
        max_site = positive.loc[positive.wape.idxmax()]
        rows.append(dict(mode=mode, year=year, sites=len(frame),
                         zero_sites=int((frame.observed == 0).sum()),
                         minimum_positive_total=positive.observed.min(),
                         maximum_wape_site=max_site.site,
                         maximum_wape=positive.wape.max(),
                         largest_share_of_wape_sum=positive.wape.max() / positive.wape.sum()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = snapshot(args.frozen)
    inputs.extend(freeze_distributions(args.frozen))
    sites = load_sites()
    summary = aggregate(sites)
    checks = reconcile(summary)
    contrast = contrasts(sites)
    products = {'sites.csv': sites, 'summary.csv': summary, 'reconciliation.csv': checks,
                'contrasts.csv': contrast, 'persistence.csv': persistence(sites, contrast),
                'denominators.csv': denominator_diagnostics(sites)}
    for name, frame in products.items():
        frame.to_csv(OUT / name, index=False)
    manifest = dict(analysis='sharing-comparison-v2-rhythms', seed=SEED, replicates=REPLICATES,
                    frozen=args.frozen, inputs=inputs, python=platform.python_version(),
                    packages={name: version(name) for name in ['numpy', 'pandas', 'matplotlib', 'pyarrow']},
                    scope='Urban-rhythm and trip-distribution analysis plus secondary held-out comparison; no refitting',
                    uncertainty='Paired pooled-WAPE spatial-unit resampling within mode/year')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Reconciled {len(summary)} configurations; exported {len(contrast)} paired contrasts.')


if __name__ == '__main__':
    main()
