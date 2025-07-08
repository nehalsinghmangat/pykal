# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "pykal"
copyright = "2025, Nehal Singh Mangat"
author = "Nehal Singh Mangat"
release = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

# Automatically generate stub pages for documented modules
autosummary_generate = True

# -- Options for doctest -----------------------------------------------------
# Code executed before each doctest block
doctest_global_setup = """
import pykal
"""

# Enable testing of doctest blocks in .rst and .py files
doctest_test_doctest_blocks = "True"

# Import Python's doctest module and Sphinx flags
import doctest

# Configure doctest option flags:
# - IGNORE_EXCEPTION_DETAIL: omit exception details
# - NORMALIZE_WHITESPACE: ignore insignificant whitespace
# - ELLIPSIS: allow "..." in expected output

doctest_default_flags = (
    doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
)

# -- intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
