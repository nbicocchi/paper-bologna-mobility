"""Cache official loop-accuracy exports and their documentation, without altering inputs."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'code/dati_esterni/cars/source'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for year in (2023, 2024):
        base = 'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/'
        for dataset in (f'accuratezza-spire-anno-{year}',
                        f'rilevazione-flusso-veicoli-tramite-spire-anno-{year}'):
            url = base + dataset
            metadata = DEST / f'{dataset}.json'
            if not metadata.exists():
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                metadata.write_text(json.dumps(response.json(), indent=2))
            if dataset.startswith('accuratezza'):
                path = DEST / f'{dataset}.csv'
                export = url + '/exports/csv'
                if not path.exists():
                    print('Downloading', export, flush=True)
                    response = requests.get(export, timeout=600)
                    response.raise_for_status()
                    temporary = path.with_suffix('.part')
                    temporary.write_bytes(response.content)
                    temporary.replace(path)
                records.append({'year': year, 'url': export, 'path': str(path.relative_to(ROOT)),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (DEST / 'sources.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
