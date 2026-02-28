"""Utility routines for reading and writing tokenized data in multiple formats.

Currently two formats are supported by extension:

* ``.pkl`` – Python pickle, used by the existing scripts.  Objects are
  loaded with :mod:`pickle` and thus can contain any Python object (usually a
  list of integers).
* ``.bin`` – raw int32 sequence stored with NumPy.  This is handy for
  very large corpora where pickle overhead becomes noticeable, and also lets
  you interoperate with other tooling that expects a simple binary array
  (e.g. ``np.fromfile``).

A simple autodetect is performed based on the filename suffix; using the
explicit helper functions keeps callers short and avoids boilerplate.

The file doesn't depend on :mod:`numpy` at import time; the module is
imported lazily only when needed, so ``requirements.txt`` only needs an extra
entry if you actually use the binary mode.
"""

import pickle
import os


def save_tokens(path: str, tokens):
    """Save a sequence of integer tokens to ``path``.

    The format is chosen by looking at ``os.path.splitext(path)[1]``:

    * ``.bin`` → NumPy ``int32`` array written with ``tofile``
    * otherwise  → pickle dump (the legacy behaviour).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - fallback path
            raise RuntimeError("numpy is required to write .bin files")
        arr = np.array(tokens, dtype=np.int32)
        arr.tofile(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(tokens, f)


def load_tokens(path: str):
    """Load a sequence of integer tokens from ``path``.

    The file extension is used to select the loader:
    ``.bin`` files are interpreted as ``int32`` arrays via NumPy; all
    other names are unpickled.  An error is raised if the file does not
    exist or the contents are not compatible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError("numpy is required to read .bin files")
        arr = np.fromfile(path, dtype=np.int32)
        return arr.tolist()
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
