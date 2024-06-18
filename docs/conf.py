import os

os.path.abspath(os.path.join(".."))
os.path.abspath(os.path.join("..", ".."))
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
    "nbsphinx",
    'sphinx.ext.mathjax',
    'sphinx_design'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

images_config = dict(
    backend='LightBox2',
    default_image_width='100%',
    default_show_title='True',
    default_group='default'
)

templates_path = ["_templates"]
html_css_files = ["mbo.css"]
html_theme = "sphinx_book_theme"
html_title = "LBM-CaImAn-Python"
html_short_title = "MBO"
html_static_path = ["_static"]


html_theme_options = {
  "external_links": [
      {"name": "MBO", "url": "https://mbo.rockefeller.edu"},
      {"name": "Hub", "url": "https://millerbrainobservatory.github.io/"},
  ]
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.5', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None)
}

intersphinx_disabled_reftypes = ["*"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

#html_sidebars = {
#    # "index": [],
#    #"guide/**": ["search-field.html", "sidebar-nav-bs.html"],
#}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_url_schemes = ("http", "https", "mailto")
