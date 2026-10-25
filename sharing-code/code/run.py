"""Rebuild both experiments from dataset/, test and export; no frozen/upstream results."""
import datetime
import importlib.metadata
import platform
import subprocess
import sys
from common import DATA, CACHE, OUT, TEXT, ROOT, REPO, directories, sha, dump


def main():
    directories()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    names = [f'{mode}-sharing-{year}.csv' for mode in ['bike', 'car'] for year in [2023, 2024]]
    names += [f'rilevazione-autoveicoli-tramite-spire-anno-{year}.csv' for year in [2023, 2024]]
    names += ['colonnine-conta-bici.csv', 'meteo-bologna-ilmeteo-2019-2025.csv']
    inputs = {str((DATA / name).relative_to(REPO)): sha(DATA / name) for name in names}
    dump(OUT / 'run-status.json', dict(status='running', started=started))
    here = ROOT / 'code'
    scripts = {str(p.relative_to(REPO)): sha(p) for p in sorted(here.glob('*.py'))}
    for script in ['prepare.py', 'features.py', 'describe.py', 'predict.py', 'analyse.py', 'export.py']:
        print('Running', script, flush=True)
        subprocess.run([sys.executable, str(here / script)], check=True, cwd=REPO)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(here), '-p', 'test_*.py', '-v'], check=True, cwd=REPO)
    for relative, expected in inputs.items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f'Input changed during run: {relative}')
    for relative, expected in scripts.items():
        if sha(REPO / relative) != expected:
            raise RuntimeError(f'Code changed during run: {relative}; rerun the full pipeline')
    outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json', 'run-status.json'])
    outputs += sorted((TEXT / 'figures').glob('*.pdf')) + sorted((TEXT / 'tables').glob('*.tex'))
    dump(OUT / 'manifest.json', dict(protocol='sharing-standalone-two-scenarios-v1', started=started,
        finished=datetime.datetime.now(datetime.UTC).isoformat(), inputs=inputs,
        external={str(p.relative_to(REPO)): sha(p) for p in sorted((CACHE / 'sources').glob('*'))},
        scripts=scripts,
        outputs={str(p.relative_to(REPO)): sha(p) for p in outputs},
        python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'pyarrow', 'requests']},
        tests='passed', source_policy='No imports, results, or intermediate data from other study directories'))
    dump(OUT / 'run-status.json', dict(status='passed', started=started, finished=datetime.datetime.now(datetime.UTC).isoformat()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        dump(OUT / 'run-status.json', dict(status='failed', error=str(error),
            finished=datetime.datetime.now(datetime.UTC).isoformat()))
        raise
