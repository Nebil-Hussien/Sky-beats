"""Shared pytest setup.

Workflow components live in per-topic folders under src/ (common/, data/, rq1..rq9).
We add each of those folders to sys.path once here, so test modules can simply
`import <component>` by name (e.g. `import genre_shift`, `import train_classifier`)
without any per-file path/importlib boilerplate. Script basenames are unique across
folders, so there are no import collisions.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
for sub in sorted(SRC.iterdir()):
    if sub.is_dir() and not sub.name.startswith((".", "__")):
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
