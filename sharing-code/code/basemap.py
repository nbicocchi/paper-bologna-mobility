"""Cached OSM context only: no rental data are sent and no routing is inferred."""
import datetime
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import requests
from common import CACHE, OUT, LAT0, LON0, sha, dump

ATTRIBUTION = '© OpenStreetMap contributors · openstreetmap.org/copyright'
HALF_WORLD = math.pi * 6378137.


def project(x, y):
    lon = LON0 + np.asarray(x) / (111.195 * np.cos(np.deg2rad(LAT0)))
    lat = LAT0 + np.asarray(y) / 111.195
    return (6378137. * np.deg2rad(lon),
            6378137. * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2)))


def load(bounds, zoom=12):
    """Fetch only the fixed viewport's tiles, sequentially; reuse verified images."""
    west, east, south, north = bounds
    mx, my = project(np.array([west, east]), np.array([south, north]))
    size = 2 * HALF_WORLD / 2**zoom
    x0, x1 = np.floor((mx + HALF_WORLD) / size).astype(int)
    y0, y1 = np.floor((HALF_WORLD - my[::-1]) / size).astype(int)
    folder = CACHE / 'basemap'
    folder.mkdir(parents=True, exist_ok=True)
    image = np.ones(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3))
    tiles = []
    with requests.Session() as session:
        session.headers['User-Agent'] = 'BolognaSharingResearch/1.0 (scientific manuscript; cached static viewport)'
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                name = f'osm-{zoom}-{x}-{y}.png'
                path = folder / name
                url = f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png'
                if not path.exists():
                    response = session.get(url, timeout=60)
                    response.raise_for_status()
                    tile = plt.imread(io.BytesIO(response.content), format='png')
                    if tile.shape[:2] != (256, 256):
                        raise ValueError('Unexpected OSM tile geometry')
                    path.write_bytes(response.content)
                tile = plt.imread(path)[..., :3]
                if tile.shape != (256, 256, 3):
                    raise ValueError(f'Invalid cached tile: {name}')
                image[(y-y0)*256:(y-y0+1)*256, (x-x0)*256:(x-x0+1)*256] = tile
                tiles.append(dict(file=str(path.relative_to(CACHE.parent.parent)), url=url, sha256=sha(path)))
    gray = image @ np.array([.299, .587, .114])
    image = np.repeat((.65 * gray + .35)[..., None], 3, axis=2)
    extent = (x0 * size - HALF_WORLD, (x1 + 1) * size - HALF_WORLD,
              HALF_WORLD - (y1 + 1) * size, HALF_WORLD - y0 * size)
    dump(OUT / 'basemap-sources.json', dict(provider='OpenStreetMap standard raster tiles', zoom=zoom,
        recorded_utc=datetime.datetime.now(datetime.UTC).isoformat(), attribution=ATTRIBUTION,
        license_url='https://www.openstreetmap.org/copyright', tiles=tiles,
        purpose='Present-day cartographic context, not a historical network or model input'))
    return image, extent


def backdrop(ax, image, extent, bounds):
    ax.imshow(image, extent=extent, origin='upper', zorder=0)
    west, east, south, north = bounds
    x, y = project(np.array([west, east]), np.array([south, north]))
    ax.set(xlim=x, ylim=y, aspect='equal')
    xticks = np.arange(math.ceil(west / 5) * 5, east + .001, 5.)
    yticks = np.arange(math.ceil(south / 5) * 5, north + .001, 5.)
    ax.set_xticks(project(xticks, np.zeros(len(xticks)))[0], [f'{v:g}' for v in xticks])
    ax.set_yticks(project(np.zeros(len(yticks)), yticks)[1], [f'{v:g}' for v in yticks])
