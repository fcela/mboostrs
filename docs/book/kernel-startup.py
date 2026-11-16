"""Configure chart output before ipykernel executes book cells."""

import importlib
import os
import runpy

previous_startup = os.environ.get("MBOOSTRS_DOCS_PREVIOUS_PYTHONSTARTUP")
if previous_startup:
    globals().update(runpy.run_path(os.path.expanduser(previous_startup), run_name="__main__"))

importlib.import_module("altair").renderers.enable("svg")
