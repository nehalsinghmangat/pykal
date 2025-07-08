import os
import sys


# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))  # Adjust as needed

# -- Project information ------------------------------------------------------
project = "pykal"
author = "Your Name"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    #    "autoapi.extension",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",  # HTML math rendering
    "sphinx.ext.imgmath",  # LaTeX/PDF math rendering
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    #   "sphinx.ext.autosummary",
    "sphinx_togglebutton",
]


# AutoAPI config
# autoapi_type = "python"
# autoapi_dirs = ["../../pykal"]  # adjust based on conf.py's location (in docs/source/)
# autoapi_keep_files = True  # Keep the generated .rst files (optional but useful)
# autoapi_root = "autoapi"  # Where generated .rst files are output inside source/


autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": False,
}
# autosummary_generate = True  # Automatically generate stub `.rst` files
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []
nbsphinx_allow_errors = True
autodoc_member_order = "bysource"

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Math configuration -------------------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
imgmath_image_format = "svg"
imgmath_use_preview = True

# -- LaTeX output -------------------------------------------------------------
latex_engine = "pdflatex"  # Or 'xelatex' or 'lualatex'

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    "figure_align": "H",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{mathtools}
\usepackage{physics}
\usepackage{graphicx}
\usepackage{float}
\usepackage{etoolbox}
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily,
  breaklines=true,
  columns=fullflexible
}
""",
}

# -- Options for EPUB output --------------------------------------------------
epub_show_urls = "footnote"
