"""Freeze official hourly recording-completeness exports for 2019--2025."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'code/dati_esterni/recovery/source'
BASE = 'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/'


def fetch(year):
    records = []
    for kind in ['accuratezza-spire', 'rilevazione-flusso-veicoli-tramite-spire']:
        dataset = f'{kind}-anno-{year}'
        url = BASE + dataset
        meta = DEST / f'{dataset}.json'
        if not meta.exists():
            response = requests.get(url, timeout=180)
            if response.status_code == 404 and kind.startswith('rilevazione'):
                dataset = f'rilevazione-autoveicoli-tramite-spire-anno-{year}'
                url = BASE + dataset
                response = requests.get(url, timeout=180)
            response.raise_for_status()
            meta.write_text(json.dumps(response.json(), indent=2))
        # Recover the actual historical catalogue ID also when reusing a snapshot.
        url = BASE + json.loads(meta.read_text())['dataset_id']
        records.append({'year': year, 'url': url, 'file': meta.name,
                        'sha256': hashlib.sha256(meta.read_bytes()).hexdigest()})
        if kind.startswith('accuratezza'):
            target = DEST / f'{dataset}.csv'
            old = ROOT / 'code/dati_esterni/cars/source' / target.name
            if not target.exists() and old.exists():
                shutil.copy2(old, target)
            if not target.exists():
                print('Downloading', year, flush=True)
                with requests.get(url + '/exports/csv', timeout=900, stream=True) as response:
                    response.raise_for_status()
                    part = target.with_suffix('.part')
                    with part.open('wb') as out:
                        for block in response.iter_content(1024 * 1024):
                            out.write(block)
                    part.replace(target)
            records.append({'year': year, 'url': url + '/exports/csv', 'file': target.name,
                            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    return records


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [record for result in pool.map(fetch, range(2019, 2026)) for record in result]
    (DEST / 'sources.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
