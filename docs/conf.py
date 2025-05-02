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

project = "fluxfootprint"
copyright = "2025, Paul Inkenbrandt, Natascha Kljun, John Volk"
author = "Paul Inkenbrandt, Natascha Kljun, John Volk"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
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

templates_path = ["_templates"]
exclude_patterns = ["_build/*", "Thumbs.db", ".DS_Store", "docs/_build/*", "tests/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
