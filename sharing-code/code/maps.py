"""Comparable endpoint maps from frozen grids and cached OSM tiles, offline."""
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from common import INPUT, OUT, TEXT, sha

LAT0, LON0 = 44.4949, 11.3426
CELL = .5
EXTENT = (-7., 8., -5., 6.)
WINDOWS = ['06:00–10:00', '10:00–15:00', '15:00–20:00', '20:00–06:00']
MINIMUM = {'bike': 200, 'car': 50}


def project(x, y):
    """Local equirectangular kilometres to EPSG:3857, without a GIS dependency."""
    lon = LON0 + np.asarray(x) / (111.195 * np.cos(np.deg2rad(LAT0)))
    lat = LAT0 + np.asarray(y) / 111.195
    return (6378137. * np.deg2rad(lon),
            6378137. * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2)))


def basemap():
    """Rebuild the original mosaic, checking every frozen tile against its manifest."""
    design = json.loads((INPUT / 'basemap.json').read_text())
    positions = []
    for tile in design['tiles']:
        path = INPUT / tile['file']
        if sha(path) != tile['sha256']:
            raise ValueError(f'Basemap hash mismatch: {path.name}')
        _, zoom, x, y = path.stem.split('-')
        if int(zoom) != design['zoom']:
            raise ValueError('Mixed basemap zoom levels')
        positions.append((int(x), int(y), path))
    x0, x1 = min(p[0] for p in positions), max(p[0] for p in positions)
    y0, y1 = min(p[1] for p in positions), max(p[1] for p in positions)
    image = np.ones(((y1-y0+1)*256, (x1-x0+1)*256, 3))
    if len(positions) != (x1-x0+1)*(y1-y0+1):
        raise ValueError('Incomplete cached basemap')
    for x, y, path in positions:
        tile = plt.imread(path)[..., :3]
        if tile.shape != (256, 256, 3):
            raise ValueError('Unexpected tile dimensions')
        image[(y-y0)*256:(y-y0+1)*256, (x-x0)*256:(x-x0+1)*256] = tile
    # Low-contrast geographic context keeps the statistical colours identifiable.
    gray = image @ np.array([.299, .587, .114])
    image = np.repeat((.65*gray+.35)[..., None], 3, axis=2)
    half = math.pi*6378137.
    size = 2*half/(2**design['zoom'])
    extent = (x0*size-half, (x1+1)*size-half, half-(y1+1)*size, half-y0*size)
    return image, extent


def load_grid(mode):
    raw = pd.read_csv(INPUT / f'{mode}-map-grid.csv')
    if mode == 'bike':
        grid = raw.rename(columns={'block': 'window', 'activity': 'events',
                                    'hours': 'exposure'})
    else:
        grid = raw.copy()
    grid['mode'] = mode
    grid['events'] = grid.arrivals + grid.departures
    grid['net'] = grid.arrivals - grid.departures
    grid['events_per_hour'] = grid.events / grid.exposure
    grid['net_per_hour'] = grid.net / grid.exposure
    grid['normalised_balance'] = grid.net / grid.events
    grid['displayed'] = grid.events.ge(MINIMUM[mode])
    grid['in_viewport'] = (grid.gx.between(EXTENT[0]/CELL, EXTENT[1]/CELL-1)
                           & grid.gy.between(EXTENT[2]/CELL, EXTENT[3]/CELL-1))
    if grid.duplicated(['window', 'gx', 'gy']).any():
        raise ValueError('Duplicate spatial cells')
    if (grid.exposure <= 0).any() or (grid.events <= 0).any():
        raise ValueError('Nonpositive map exposure/events')
    return grid


def rhythm_summaries(grids=None):
    """Exploratory geographic summaries; no inferred land use or linked people."""
    grids = grids if grids is not None else {mode: load_grid(mode) for mode in MINIMUM}
    regions, reversals, concordance = [], [], []
    for mode, grid in grids.items():
        viewport = grid[grid.in_viewport].copy()
        x = (viewport.gx + .5) * CELL
        y = (viewport.gy + .5) * CELL
        # Disjoint geometric summaries, not administrative or land-use districts.
        viewport['region'] = np.select([np.hypot(x, y) < 1.5, y >= 1.5],
                                       ['Centre', 'Northern band'], default='Remaining area')
        grouped = viewport.groupby(['region', 'window'])[['arrivals', 'departures']].sum().reset_index()
        grouped['mode'] = mode
        grouped['net'] = grouped.arrivals - grouped.departures
        grouped['normalised_balance'] = grouped.net / (grouped.arrivals + grouped.departures)
        regions.append(grouped)
        common = viewport[viewport.displayed].pivot(index=['gx', 'gy'], columns='window',
                                                   values='normalised_balance')[[0, 2]].dropna()
        opposite = common[0] * common[2] < 0
        reversals.append(dict(mode=mode, morning_window=0, late_afternoon_window=2,
                              common_displayed_cells=len(common), reversing_cells=int(opposite.sum()),
                              reversal_fraction=float(opposite.mean()),
                              morning_afternoon_correlation=float(common[0].corr(common[2]))))
    for window in range(4):
        paired = []
        for mode, grid in grids.items():
            frame = grid[grid.in_viewport & grid.displayed & grid.window.eq(window)]
            paired.append(frame.set_index(['gx', 'gy'])[['events_per_hour', 'normalised_balance']]
                          .add_suffix(f'_{mode}'))
        common = paired[0].join(paired[1], how='inner')
        concordance.append(dict(window=window, common_displayed_cells=len(common),
            activity_rank_correlation=float(common.events_per_hour_bike.rank().corr(
                                             common.events_per_hour_car.rank())),
            balance_correlation=float(common.normalised_balance_bike.corr(common.normalised_balance_car))))
    return pd.concat(regions, ignore_index=True), pd.DataFrame(reversals), pd.DataFrame(concordance)


def draw_counter_map():
    """Map only the observation networks actually used in prediction tests."""
    bike = pd.read_csv(INPUT / 'bike-counter-locations.csv')
    selected = pd.read_csv(INPUT / 'bike-sites.csv')['site'].unique()
    bike = bike[bike.site.isin(selected)].groupby('site')[['x', 'y']].mean().reset_index()
    car = pd.read_csv(INPUT / 'car-counter-locations.csv')
    expected = set(pd.read_csv(INPUT / 'car-sites.csv').query('year == 2023').site)
    if set(car.site) != expected or len(bike) != 4 or len(car) != 752:
        raise ValueError('Counter map does not match the experimental cohorts')
    context, extent = basemap()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), layout='constrained')
    cx, cy = project(car.x, car.y)
    bx, by = project(bike.x, bike.y)
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    clusters = car.groupby('cluster').agg(x=('x', 'mean'), y=('y', 'mean'),
                                         block=('block', 'first'), fold=('fold', 'first'))
    blocks = sorted(car.block.unique())
    if len(clusters) != 246 or len(blocks) != 63:
        raise ValueError('Unexpected vehicle spatial grouping')
    corners_x, corners_y = [], []
    for block in blocks:
        gx, gy = map(int, block.split(':'))
        px, py = project(np.array([gx, gx+1, gx+1, gx]),
                         np.array([gy, gy, gy+1, gy+1]))
        axes[1].add_patch(Polygon(np.column_stack([px, py]), facecolor='#0072b2',
                                 alpha=.10, edgecolor='none', zorder=1))
        axes[1].add_patch(Polygon(np.column_stack([px, py]), fill=False,
                                 edgecolor='#0072b2', lw=.65, alpha=.8, zorder=1))
        corners_x.extend(px)
        corners_y.extend(py)
    axes[1].scatter(cx, cy, s=18, color='#d55e00', alpha=.8, linewidths=.25,
                    edgecolors='white', zorder=2)
    kx, ky = project(clusters.x, clusters.y)
    axes[1].scatter(kx, ky, s=22, marker='+', color='#163e63', linewidths=.8, zorder=3)
    axes[1].legend(handles=[
        Line2D([], [], marker='o', ls='', color='#d55e00', markersize=4, label='752 devices'),
        Line2D([], [], marker='+', ls='', color='#163e63', markersize=5, label='246 group centres'),
        Polygon([[0, 0], [1, 0], [1, 1]], facecolor='#0072b2', alpha=.25,
                edgecolor='#0072b2', label='63 blocks · 1 km')],
        loc='lower left', fontsize=8, framealpha=.95)
    axes[0].scatter(bx, by, s=85, marker='^', color='#0072b2', edgecolors='white',
                    linewidths=.8, zorder=3)
    offsets = {'Ercolani': (8, -18), 'Parri': (-42, 15),
               'Sabotino': (-60, -18), 'San Donato': (10, 7)}
    for i, row in bike.iterrows():
        axes[0].annotate(row.site, (bx[i], by[i]), xytext=offsets[row.site],
                    textcoords='offset points', fontsize=9, zorder=4,
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2),
                    arrowprops=dict(arrowstyle='-', color='#0072b2'))
    # Include every eligible detector; cached tiles may end beyond the urban core.
    allx, ally = np.r_[cx, bx, corners_x], np.r_[cy, by, corners_y]
    sx, sy = project(np.array([car.x.min()+.5, car.x.min()+2.5]),
                     np.repeat(car.y.max()-.5, 2))
    for ax, title in zip(axes, ['(a) Bicycle counters · 4 locations',
                                 '(b) Motor-vehicle detectors · 752 devices']):
        ax.imshow(context, extent=extent, origin='upper', zorder=0)
        ax.set(xlim=(allx.min()-650, allx.max()+650),
               ylim=(ally.min()-650, ally.max()+650), aspect='equal', title=title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(sx, sy, color='black', lw=2)
        ax.text(sx.mean(), sy[0]+180, '2 km', ha='center', fontsize=9)
        ax.text(.98, .98, 'N ↑', transform=ax.transAxes, ha='right', va='top')
    fig.supxlabel('Map: © OpenStreetMap contributors · openstreetmap.org/copyright\n'
                  'Eligible 2023 network; colocated devices can overlap', fontsize=9)
    fig.savefig(TEXT / 'figures' / 'counter-network.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    clusters.to_csv(OUT / 'counter-map-groups.csv')
    pd.concat([bike.assign(mode='bike'), car[['site', 'x', 'y']].assign(mode='car')],
              ignore_index=True).to_csv(OUT / 'counter-map-locations.csv', index=False)


def draw_maps():
    draw_counter_map()
    context, extent = basemap()
    grids = {mode: load_grid(mode) for mode in MINIMUM}
    regions, reversals, concordance = rhythm_summaries(grids)
    regions.to_csv(OUT / 'rhythm-regions.csv', index=False)
    reversals.to_csv(OUT / 'rhythm-reversals.csv', index=False)
    concordance.to_csv(OUT / 'rhythm-concordance.csv', index=False)
    columns = ['mode', 'window', 'gx', 'gy', 'departures', 'arrivals', 'events',
               'net', 'exposure', 'events_per_hour', 'net_per_hour',
               'normalised_balance', 'displayed', 'in_viewport']
    pd.concat([g[columns] for g in grids.values()]).to_csv(OUT / 'spatial-grid.csv', index=False)
    xedges = np.arange(EXTENT[0], EXTENT[1] + CELL, CELL)
    yedges = np.arange(EXTENT[2], EXTENT[3] + CELL, CELL)
    mx, my = project(*np.meshgrid(xedges, yedges))
    nx, ny = len(xedges)-1, len(yedges)-1
    index = pd.MultiIndex.from_product([range(int(EXTENT[2]/CELL), int(EXTENT[3]/CELL)),
                                        range(int(EXTENT[0]/CELL), int(EXTENT[1]/CELL))],
                                       names=['gy', 'gx'])
    metadata = {'cell_km': CELL, 'extent_km': EXTENT, 'minimum_events': MINIMUM,
                'windows': WINDOWS, 'years': [2023, 2024],
                'basemap': 'Cached OpenStreetMap tiles; context, not historical routes',
                'activity': '(arrivals + departures) / source-specific valid hours',
                'balance': '(arrivals - departures) / (arrivals + departures)',
                'date_matching': 'Source-specific coverage; not matched across modes',
                'night': {'bike': 'Continuous overnight, assigned to preceding evening',
                          'car': '00–06 plus 20–24 on the same calendar date'},
                'rhythm_regions': {'purpose': 'Exploratory geometric summaries, not validated land use',
                    'centre': 'Cell centre less than 1.5 km from reference point',
                    'northern_band': 'Noncentral cell centre at least 1.5 km north of reference',
                    'remaining_area': 'Other viewport cells',
                    'aggregation': 'Sum arrivals and departures before normalisation; no display threshold'},
                'rhythm_reversals': 'Opposite nonzero signs at cells displayed in both 06–10 and 15–20',
                'rhythm_concordance': 'Common displayed cells per window; source dates not matched',
                'panels': []}
    for name, measure in [('sharing-activity', 'events_per_hour'),
                          ('sharing-balance', 'normalised_balance')]:
        fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), layout='constrained')
        for row, (mode, grid) in enumerate(grids.items()):
            shown = grid[grid.displayed & grid.in_viewport]
            maximum = shown.events_per_hour.max()
            norm = Normalize(0, maximum) if name == 'sharing-activity' else TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            for window, ax in enumerate(axes[row]):
                data = grid[grid.window == window].set_index(['gy', 'gx']).reindex(index)
                values = data[measure].to_numpy().reshape(ny, nx)
                mask = ~(data.events.ge(MINIMUM[mode]).to_numpy().reshape(ny, nx))
                values = np.ma.masked_where(mask | ~np.isfinite(values), values)
                ax.imshow(context, extent=extent, origin='upper', zorder=0)
                mesh = ax.pcolormesh(mx, my, values, norm=norm,
                                     cmap='YlOrRd' if name == 'sharing-activity' else 'RdBu_r',
                                     alpha=.82, rasterized=True, zorder=1)
                ax.set(xlim=(mx.min(), mx.max()), ylim=(my.min(), my.max()), aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'({chr(97+row*4+window)}) {WINDOWS[window]}', fontsize=11)
                if window == 0:
                    ax.set_ylabel('Bike-sharing' if mode == 'bike' else 'Car-sharing', fontsize=12)
                # Ground-distance scale expressed in the original local projection.
                bx, by = project(np.array([-6.3, -4.3]), np.array([-4.4, -4.4]))
                ax.plot(bx, by, color='black', lw=2)
                ax.text(bx.mean(), by[0]+200, '2 km', ha='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=1))
                if name == 'sharing-activity':
                    subset = grid[grid.window == window]
                    displayed = subset[subset.displayed & subset.in_viewport]
                    metadata['panels'].append(dict(mode=mode, window=window,
                        exposure_hours=float(subset.exposure.iloc[0]),
                        displayed_cells_in_viewport=len(displayed),
                        activity_scale_max=float(maximum),
                        maximum_events_per_hour=float(displayed.events_per_hour.max()),
                        median_absolute_balance=float(displayed.normalised_balance.abs().median())))
            label = 'Endpoint events / hour' if name == 'sharing-activity' else '(Arrivals − departures) / events'
            colorbar = fig.colorbar(mesh, ax=axes[row].tolist(), shrink=.86, pad=.015, label=label)
            if name == 'sharing-balance':
                colorbar.set_ticks([-1, -.5, 0, .5, 1])
        fig.supxlabel('Map: © OpenStreetMap contributors · openstreetmap.org/copyright\n'
                      '2023–2024 · 500 m cells · source-specific valid dates; blank cells are suppressed, not zero', fontsize=9)
        fig.savefig(TEXT / 'figures' / f'{name}.pdf', bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
    (OUT / 'spatial-design.json').write_text(json.dumps(metadata, indent=2) + '\n')
    pd.DataFrame(metadata['panels']).to_csv(OUT / 'spatial-summary.csv', index=False)
