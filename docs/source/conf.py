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
copyright = '2024, Flynn OConnell'
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
]

pygments_style = 'sphinx'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
source_suffix = ['.rst', '.md']

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index' 

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


html_theme_options = {
  "github_url": "https://github.com/ru-rbu",
  "external_links": [
      {"name": "RBO", "url": "https://rbo.rockefeller.edu"},
      {"name": "Github", "url": "https://www.github.com/ru-rbo"}
      ],
}

html_theme = 'pydata_sphinx_theme'
html_title ="Rockefeller Brain Observatory Hub"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'rbo.css'
]
