# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
from pathlib import Path

import toml

sys.path.append(Path(__file__).parent)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

toml_config = toml.load("../pyproject.toml")

project = toml_config["project"]["name"]
release = toml_config["project"]["version"]

copyright = f"{datetime.datetime.now().year}, clinlp"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_new_tab_link",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__call__",
    "ignore-module-all": True,
}

myst_heading_anchors = 3

napoleon_include_init_with_doc = True
napoleon_use_rtype = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
