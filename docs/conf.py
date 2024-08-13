# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from pathlib import Path

os.path.abspath(os.path.join(".."))
os.path.abspath(os.path.join("..", "demos"))
os.path.abspath(os.path.join("..", "demos", "notebooks"))

project = "LBM-CaImAn-Python"
copyright = "2024, Elizabeth R. Miller Brain Observatory | The Rockefeller University. All Rights Reserved"
author = "Flynn OConnell"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = ["Thumbs.db", ".DS_Store"]
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.images",
    "sphinxcontrib.video",
    "sphinxcontrib.matlab",
    # "myst_parser",
    "myst_nb",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinx_tippy",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
html_static_path = ["_static"]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "LBM-CaImAn-Python"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/caiman-python-logo.svg"

html_favicon = "./_static/icon_caiman_python.png"

html_theme = "sphinx_book_theme"
html_title = "LBM-CaImAn-Python"
html_short_title = "MBO.py"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_use_modindex = True
html_copy_source = False
html_file_suffix = ".html"

current_filepath = Path()
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.5", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "mbo": (
        "https://millerbrainobservatory.github.io/",
        None,
    ),
}

templates_path = ["_templates"]

html_theme = "sphinx_book_theme"

html_logo = "_static/CaImAn-MATLAB_logo.svg"
html_short_title = "LBM CaImAn Pipeline"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# html_js_files = ["subtoc.js"]
html_favicon = "./_static/lbm_caiman_mat.svg"
html_copy_source = True

intersphinx_disabled_reftypes = ["*"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/executablebooks/sphinx-book-theme",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 3,
    "icon_links": [
        {
            "name": "MBO User Hub",
            "url": "https://millerbrainobservatory.github.io/",
            "icon": "./_static/icon_mbo_home.png",
            "type": "local",
        },
        {
            "name": "MBO Github",
            "url": "https://github.com/MillerBrainObservatory/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Connect with MBO",
            "url": "https://mbo.rockefeller.edu/contact/",
            "icon": "fa-regular fa-address-card",
            "type": "fontawesome",
        },
    ],
}
