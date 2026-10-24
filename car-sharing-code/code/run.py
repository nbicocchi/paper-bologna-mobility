"""Run the entire analysis with the current interpreter, sequentially and fail-fast."""
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
for name in ['fetch_accuracy.py', 'cars_prepare.py', 'cars_analyse.py', 'cars_predict.py',
             'cars_export.py', 'cars_test.py']:
    print('\nRunning',name,flush=True)
    subprocess.run([sys.executable,str(HERE/name)],check=True,cwd=HERE.parent)
