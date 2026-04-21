"""
pytest configuration for the SparseCore test suite.

The main purpose of this file is to ensure pytest uses the correct
import-resolution strategy. Without it, pytest's default "prepend"
import mode can add the workspace root (/SparseCore/) to sys.path[0],
which makes it prefer the raw folder `sparsecore/` over the properly
installed editable package — and the raw folder has no way to tell
Python that it's a package with an __init__.py that needs executing,
because pytest skips that step.

By putting this conftest.py inside tests/, pytest sets the rootdir
at the tests/ level and won't prepend the workspace root. The
installed sparsecore package (via pip install -e .) is then resolved
through the normal Python editable-install finder, which correctly
runs __init__.py and exposes PaddedCSR.
"""

# Nothing to import here — conftest.py presence alone is enough to
# signal pytest about the test directory boundary.
