# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add source directories to sys.path so autodoc can find modules
sys.path.insert(0, os.path.abspath('../../pykal'))
sys.path.insert(0, os.path.abspath('../../pykal_ros'))

# -- Project information -----------------------------------------------------

project = 'pykal'
copyright = '2025, Nehal Singh Mangat'
author = 'Nehal Singh Mangat'
release = '0.1.0'  # Version of the project

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",                      # Jupyter notebook support
    'sphinx.ext.autodoc',           # Automatically document docstrings
    'sphinx.ext.napoleon',          # Support for Google/NumPy-style docstrings
    'sphinx.ext.viewcode',          # Add [source] links to Python code
    'sphinx.ext.mathjax',           # Math rendering in HTML (LaTeX-style)
    'sphinx.ext.autosummary',       # Generate API stubs automatically
    'sphinx_autodoc_typehints',     # Show type hints in API docs
    'myst_parser',                  # Markdown support via MyST
    'sphinx.ext.intersphinx',       # Link to external Sphinx docs (like NumPy)
    'sphinx_toolbox.collapse',      # Support for collapsible content blocks
    "sphinx.ext.graphviz",          # Render Graphviz diagrams in docs
]

# Autodoc default behavior for documenting modules
autodoc_default_options = {
    'members': True,               # Include class/method/function members
    'undoc-members': True,         # Include members without docstrings
    'show-inheritance': True,      # Show class inheritance info
    'inherited-members': True,     # Show members inherited from base classes
}

autodoc_typehints = 'description'  # Move type hints from signature into docstring
autosummary_generate = True        # Auto-generate .rst stubs for documented APIs

# Napoleon (Google/NumPy docstring style) configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Allow both .rst and .md files in the docs
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Intersphinx mappings to other project documentation for :ref: cross-links
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Template and pattern configuration
templates_path = ['_templates']                # Custom Jinja templates
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']  # Files to ignore

# -- HTML output configuration -----------------------------------------------

html_theme = 'sphinx_rtd_theme'      # Theme used (ReadTheDocs theme)
html_static_path = ['_static']       # Static assets like CSS/JS/images

# -- Math configuration ------------------------------------------------------

# Load MathJax from CDN for faster builds and modern features
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
imgmath_image_format = "svg"         # Preferred image format for math output
imgmath_use_preview = True           # Display math preview images if possible

# -- LaTeX output configuration ----------------------------------------------

latex_engine = "pdflatex"            # LaTeX engine; alternatives: 'xelatex', 'lualatex'

latex_elements = {
    "papersize": "letterpaper",      # U.S. paper size
    "pointsize": "11pt",             # Font size
    "figure_align": "H",             # Force figure placement using float
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
""",  # Extra LaTeX packages for math, code formatting, and layout
}

# -- EPUB output configuration -----------------------------------------------

epub_show_urls = "footnote"  # Where to show URLs in EPUB output (if built)
