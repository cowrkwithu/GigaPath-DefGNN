"""Pipeline CLI scripts (Phase 10).

Each script is a thin argparse wrapper around a ``src/`` module.
Scripts are invokable directly (``python scripts/01_preprocess.py``);
the leading ``sys.path`` insertion at the top of each script makes the
repo root importable.
"""
