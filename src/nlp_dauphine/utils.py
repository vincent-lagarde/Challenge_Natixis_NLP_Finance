"""NLP Challenge utils."""

import os
import numpy as np
from pathlib import Path


def set_working_dir():
    """Set working dir for Jupyter."""
    cwd = Path(os.getcwd())
    print(f"Current working dir: {cwd.as_posix()}")
    if Path(cwd).name == "notebooks":
        new_cwd = Path(cwd).parent
        os.chdir(new_cwd)
        print(f"New working dir: {new_cwd.as_posix()}")


def euclidean(u, v):
    return np.linalg.norm(u - v)


def length_norm(u):
    return u / np.sqrt(u.dot(u))


def cosine(u, v):
    return 1.0 - length_norm(u).dot(length_norm(v))
