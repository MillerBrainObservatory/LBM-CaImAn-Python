# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

os.path.abspath(os.path.join("..", ".."))
os.path.abspath(os.path.join("..", "demos"))
sys.path.insert(0, os.path.abspath(os.path.join("..", "core")))
matlab_src_dir = os.path.abspath("../core/")

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
    "nbsphinx",
    'sphinx.ext.mathjax',
    'sphinx_design'
]


# List of documents that shouldn't be included in the build.
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
# html_use_modindex = False
# html_use_index = False

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "LBM-CaImAn-Python"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_images/MillerBrainObservatory_logo.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/mbo_icon.ico"

html_theme = 'sphinx_book_theme'
html_title = "CaImAn-Python"
html_short_title = "MBO"
html_static_path = ['_static']
html_css_files = ["mbo.css"]
html_use_modindex = True
html_copy_source = False
html_file_suffix = '.html'

# This is a dictionary where the value is a tuple.
# The first link is a link to our "deployed" documentation URL
# The second is a path relative to the local build so sphinx can instead
# map to that location.

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.5', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
}

intersphinx_disabled_reftypes = ["*"]

# html_theme_options = {
#     "github_url": "https://github.com/MillerBrainObservatory/millerbrainobservatory.github.io",
#     "collapse_navigation": True,
#     "external_links": [
#         {"name": "MBO.edu", "url": "https://mbo.rockefeller.edu/"},
#         {"name": "LBM.Mat", "url": "https://mbo.rockefeller.edu/"},
#         {"name": "LBM.Py", "url": "https://mbo.rockefeller.edu/"},
#         {"name": "LBM.scanreader", "url": "https://mbo.rockefeller.edu/"},
#     ],
#     "navbar_start": [ "navbar_logo","search-button", ],
#     "header_links_before_dropdown": 6,
#     "navbar_end": [ "navbar-icon-links" ],
#     "navbar_persistent": [],
#     "navbar_align": "content",
# }


# html_theme_options = {
#     "external_links": [
#         {"name": "MBO",  "url": "https://mbo.rockefeller.edu"},
#     ],
#     "github_url": "https://github.com/MillerBrainObservatory",
#     "navbar_align": "left",
#     "navbar_end": ["navbar-icon-links"],
#     "navbar_start": ["navbar-logo"],
#     "show_nav_level": 2,
#     "show_toc_level": 1,
#     "use_edit_page_button": False,
#     "header_links_before_dropdown": 8,
# }

# html_sidebars = {
#     "index": [],
#     "pipelines/**": ["search-field.html", "sidebar-nav-bs.html"],
#     "user_guide/**": ["search-field.html", "sidebar-nav-bs.html"],
# }


templates_path = ["_templates"]
