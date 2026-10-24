"""Reproduce the recovery study; --cached skips downloading and raw-data preparation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached', action='store_true', help='Reuse audited daily parquet and frozen source exports')
    args = parser.parse_args()
    scripts = ([] if args.cached else ['recovery_fetch.py', 'recovery_prepare.py']) + ['recovery_analyse.py', 'recovery_test.py']
    for script in scripts:
        subprocess.run([sys.executable, str(ROOT / 'code' / script)], cwd=ROOT, check=True)
    inputs = sorted((ROOT / 'dataset').glob('*autoveicoli*spire*csv'))
    inputs += sorted((ROOT / 'code/dati_esterni/recovery/source').glob('*'))
    code = sorted((ROOT / 'code').glob('recovery_*.py')) + [Path(__file__), ROOT / 'requirements.txt']
    text = ROOT.parent / 'paper-car-sharing-text'
    output = sorted((ROOT / 'results/recovery').glob('*.csv')) + sorted(text.glob('tables/recovery-*.tex')) + sorted(text.glob('figures/recovery-*.pdf'))
    manuscript = [text / 'main.tex', text / 'references.bib',
                  ROOT / 'results/recovery/literature/README.md']
    provenance = {'python': platform.python_version(), 'command': sys.argv,
                  'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels', 'pyarrow', 'requests', 'python-dateutil']},
                  'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  'code': {str(p.relative_to(ROOT)): sha(p) for p in code},
                  'manuscript': {str(p.relative_to(ROOT.parent)): sha(p) for p in manuscript},
                  'outputs': {str(p.relative_to(ROOT.parent)): sha(p) for p in output}}
    (ROOT / 'results/recovery/reproducibility.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
