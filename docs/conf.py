# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join("..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "lbm_caiman_python")))

project = "LBM-CaImAn-Python"
copyright = "2024, Elizabeth R. Miller Brain Observatory | The Rockefeller University. All Rights Reserved"
author = "Flynn OConnell"
release = "0.8.0"

# def fetch_notebooks():
    # import requests
    # from bs4 import BeautifulSoup
#     BASE_URL = "https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/blob/master/demos/render/"
#     RAW_BASE_URL = "https://raw.githubusercontent.com/MillerBrainObservatory/LBM-CaImAn-Python/master/demos/render/"
#     OUTPUT_DIR = "./notebooks"  # Destination for notebooks

#     # Ensure the output directory exists
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Get the HTML content of the GitHub directory
#     response = requests.get(BASE_URL)
#     if response.status_code != 200:
#         raise RuntimeError(f"Failed to fetch directory listing: {response.status_code}")

#     # Parse the HTML to find links to .ipynb files
#     soup = BeautifulSoup(response.text, 'html.parser')
#     notebook_links = [
#         a['href'] for a in soup.find_all('a', href=True)
#         if a['href'].endswith(".ipynb")
#     ]

#     if not notebook_links:
#         raise RuntimeError("No .ipynb files found in the specified directory.")

#     # Download each notebook
#     for link in notebook_links:
#         notebook_name = os.path.basename(link)
#         raw_url = RAW_BASE_URL + notebook_name

#         notebook_response = requests.get(raw_url)
#         if notebook_response.status_code == 200:
#             output_path = os.path.join(OUTPUT_DIR, notebook_name)
#             with open(output_path, "wb") as f:
#                 f.write(notebook_response.content)
#             print(f"Downloaded {notebook_name} to {output_path}")
#         else:
#             print(f"Failed to download {notebook_name}: {notebook_response.status_code}")

#     print(f"All notebooks downloaded to {OUTPUT_DIR}")

# fetch_notebooks()

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
    # "sphinxcontrib.images",
    "sphinxcontrib.video",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_tippy",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

myst_admonition_enable = True
myst_amsmath_enable = True
myst_html_img_enable = True
myst_url_schemes = ("http", "https", "mailto")

images_config = {"cache_path": "./_images/"}

templates_path = ["_templates"]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "LBM-CaImAn-Python"

html_logo = "./_static/logo_py.png"
html_favicon = "_static/icon_caiman_python.svg"
html_theme = "sphinx_book_theme"

html_short_title = "LBM CaImAn Pipeline"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = True
html_file_suffix = ".html"
# html_use_modindex = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "mbo": (
        "https://millerbrainobservatory.github.io/",
        None,
    ),
    "caiman": ("https://caiman.readthedocs.io/en/latest/", None),
    "mesmerize": ("https://mesmerize-core.readthedocs.io/en/latest", None),
    "suite2p": ("https://suite2p.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
intersphinx_disabled_reftypes = ["*"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/",
    "repository_branch": "master",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 3,
    "navbar_align": "content",
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
