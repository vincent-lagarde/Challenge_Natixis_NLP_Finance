"""NLP Challenge utils."""

import os
from pathlib import Path


def set_working_dir():
    """Set working dir for Jupyter."""
    cwd = Path(os.getcwd())
    print(f"Current working dir: {cwd.as_posix()}")
    if Path(cwd).name == "notebooks":
        new_cwd = Path(cwd).parent
        os.chdir(new_cwd)
        print(f"New working dir: {new_cwd.as_posix()}")
