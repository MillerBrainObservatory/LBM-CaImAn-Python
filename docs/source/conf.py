# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../img/'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'rbo-lbm'
copyright = '2024, Elizabeth R. Miller Brain Observatory.'
author = 'Flynn OConnell, Vaziri Lab Members'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx-prompt',
    'sphinxcontrib.apidoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'myst_parser'
]

html_theme = "sphinx_book_theme"
pygments_style = 'sphinx'
templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

html_title = "CaImAn (Python) MBO User Guide"
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

html_context = {"default_mode": "dark"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

# MyST configuration reference: https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_heading_anchors = 3
linkcheck_ignore = [
    r".*github\.com.*#",
    r"http://127\.0\.0\.1:.*",
]

htmlhelp_basename = 'mbo'
html_theme_options = {
    "logo": {
        "image_light": "_static/numpylogo.svg",
        "image_dark": "_static/numpylogo_dark.svg",
    },
    "github_url": "https://github.com/ru-rbo/rbo-lbm/",
    "collapse_navigation": True,
    "external_links": [
        {"name": "MBO", "url": "https://mbo.rockefeller.edu"},
    ],
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
}

html_css_files = [
    'rbo.css'
]
