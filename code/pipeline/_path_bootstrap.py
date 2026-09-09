"""Make sibling code directories importable.

The pipeline scripts import each other and cyto7_config by bare
name, which works when code/ and this directory are both on
sys.path. Importing this module arranges that.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
