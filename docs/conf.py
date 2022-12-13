# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "StableGNN"
copyright = "2022, NCCR Team (ITMO University)"
author = "NCCR Team (ITMO University)"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_rtd_theme", "sphinx.ext.autodoc", "sphinxcontrib.katex", "sphinx_mdinclude"]

add_module_names = False
add_package_names = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tests", "tutorials"]
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "bamt",
    "optuna",
    "pgmpy",
    "sklearn",
    "pandas",
    "scipy",
    "torch_sparse",
    "numpy",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False, "titles_only": True}
