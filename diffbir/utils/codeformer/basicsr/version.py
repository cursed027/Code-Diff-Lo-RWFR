# basicsr/version.py

import os

__gitsha__ = "unknown"

def _read_version():
    root = os.path.dirname(__file__)
    version_file = os.path.join(root, "VERSION")
    try:
        with open(version_file, "r") as f:
            return f.read().strip()
    except Exception:
        return "0.0.0"

__version__ = _read_version()
