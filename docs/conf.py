# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

os.path.abspath(os.path.join(".."))
os.path.abspath(os.path.join("..", "demos"))
os.path.abspath(os.path.join("..", "demos", "notebooks"))

project = 'LBM-CaImAn-Python'
copyright = '2024, Flynn OConnell'
author = 'Flynn OConnell'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = ['Thumbs.db', '.DS_Store']

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.images",
    "sphinxcontrib.video" ,
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "myst_nb",
    'sphinx.ext.mathjax',
    'sphinx_design',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}


templates_path = ["_templates"]
html_static_path = ['_static']
# html_use_modindex = False
# html_use_index = False

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "LBM-CaImAn-Python"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/caiman-python-logo.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/lbm.ico"

html_theme = 'sphinx_book_theme'
html_title = "LBM-CaImAn-Python"
html_short_title = "MBO.py"
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_use_modindex = True
html_copy_source = False
html_file_suffix = '.html'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.5', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
}

intersphinx_disabled_reftypes = ["*"]

# html_sidebars = {
#     "index": [],
#     "guide/**": ["search-field.html", "sidebar-nav-bs.html"],
# }

