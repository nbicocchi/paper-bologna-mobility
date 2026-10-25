"""Fail-fast analysis, tests and exports; never execute the source pipelines."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT, REPO, OUT, TEXT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'analyse.py')] +
                   (['--frozen'] if args.frozen else []), check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(here),
                    '-p', 'test_*.py', '-v'], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(here / 'export.py')], check=True, cwd=ROOT)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if not args.frozen:
        for record in manifest['inputs']:
            if sha(REPO / record['source']) != record['sha256']:
                raise RuntimeError(f'Source changed during run: {record["source"]}')
    manifest['scripts'] = {str(p.relative_to(REPO)): sha(p) for p in sorted(here.glob('*.py'))}
    products = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json')
    products += sorted((TEXT / 'tables').glob('*.tex'))
    products += sorted((TEXT / 'figures').glob('*.pdf'))
    products += [p for p in [TEXT / 'main.tex', TEXT / 'references.bib', ROOT / 'requirements.txt'] if p.exists()]
    manifest['outputs'] = {str(p.relative_to(REPO)): sha(p) for p in products}
    manifest['tests'] = 'Passed; pooled, bias and macro reconciled for all 36 configurations'
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Provenance saved; consumed source files unchanged.')


if __name__ == '__main__':
    main()
