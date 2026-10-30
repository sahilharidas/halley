"""A local GUI for Halley: build a model, run it, read its runs, explain a value and compare two
runs, in the browser.

``python -m halley.gui`` (or the ``halley-gui`` command) starts a web server on this computer's
loopback interface and opens the GUI in the default browser (:mod:`halley.gui.__main__`). The
server (:mod:`halley.gui.server`) serves the pages under ``static/`` and a JSON API that only a
page holding the session's token can call. The API's routes (:mod:`halley.gui.app`) call
Halley's public functions, ``build``, ``run``, ``load_run``, ``explain`` and ``diff``, on the
shipped models (:mod:`halley.gui.catalog`, with their reference inputs in
:mod:`halley.gui.examples`) and on model files the user loads. Nothing here computes a value the
engine does not: a run made in the GUI is the run the same call makes in Python, with the same
run id. The GUI needs nothing beyond Halley's own dependencies.
"""
