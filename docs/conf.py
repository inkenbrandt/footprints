# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../src"))  # adjust path as needed
# sys.path.append("../..")  # Adjust this path as needed
import fluxfootprints  # Import the package to be documented

project = "fluxfootprints"
copyright = "2025, Paul Inkenbrandt, Diane Menuz, Natascha Kljun, John Volk"
author = "Paul Inkenbrandt, Diane Menuz, Natascha Kljun, John Volk"
release = fluxfootprints.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
]
napoleon_numpy_docstring = True  # Set this to True for NumPy-style
autosummary_generate = True  # Automatically generate .rst files for modules
autosummary_imported_members = True
bibtex_bibfiles = ["refs.bib"]  # Your BibTeX file(s)
bibtex_reference_style = "author_year"  # Use author-year style for citations
bibtex_default_style = "plain"
templates_path = ["_templates"]
# Paths here are relative to this file (docs/).
exclude_patterns = [
    "_build/*",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "notebooks/input_data/*",
    # These notebooks demonstrate the separate `micromet` package, not
    # fluxfootprints, so they are not part of this documentation.
    "notebooks/multiply_rasters.ipynb",
    "notebooks/openet_comparison.ipynb",
]

nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
