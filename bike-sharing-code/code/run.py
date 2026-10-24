"""Run the entire analysis with the current interpreter, sequentially and fail-fast."""
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
for name in ['prepare.py','analyse.py','diagnostics.py','predict.py','synthetic.py','export_tables.py','test_pipeline.py']:
    print('\nRunning',name,flush=True)
    subprocess.run([sys.executable,str(HERE/name)],check=True,cwd=HERE.parent)
