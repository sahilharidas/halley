"""Sphinx configuration for the site at https://halley.kndra.co.za.

Build it with:

    uv run --group docs sphinx-build -W -b html docs docs/_build/html
"""

import halley

project = "Halley"
author = "Sahil Haridas"
copyright = "2026, Sahil Haridas"
release = halley.__version__
version = release

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
# Links to headings, as GitHub renders them, down to the specs' numbered subsections.
myst_heading_anchors = 3

html_theme = "furo"
html_title = f"Halley {release}"
html_baseurl = "https://halley.kndra.co.za/"
# The specs are read as pages; their Markdown is in the repo and the sdist, not on the site.
html_copy_source = False
html_show_sourcelink = False
