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
# sys.path.insert(0, os.path.abspath('../..'))

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

# -- Project information -----------------------------------------------------

project = 'scbean'
copyright = '2020, Jialu Hu'
author = 'Jialu Hu'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'IPython.sphinxext.ipython_console_highlighting',
'nbsphinx',
'recommonmark',
'sphinx.ext.todo', 
'sphinx.ext.viewcode',  
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.napoleon',
'sphinx.ext.coverage',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# import mock

# MOCK_MODULES = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'matplotlib.pyplot', 
# 'scipy.interpolate', # 'scipy.special', 'math', '__future__', 'toolboxutilities']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()

intersphinx_mapping = dict(
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    bbknn=('https://bbknn.readthedocs.io/en/latest/', None),
    cycler=('https://matplotlib.org/cycler/', None),
    h5py=('http://docs.h5py.org/en/stable/', None),
    ipython=('https://ipython.readthedocs.io/en/stable/', None),
    leidenalg=('https://leidenalg.readthedocs.io/en/latest/', None),
    louvain=('https://louvain-igraph.readthedocs.io/en/latest/', None),
    matplotlib=('https://matplotlib.org/', None),
    networkx=('https://networkx.github.io/documentation/networkx-1.10/', None),
    numpy=('https://docs.scipy.org/doc/numpy/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    pytest=('https://docs.pytest.org/en/latest/', None),
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy/reference/', None),
    scvelo=('https://scvelo.readthedocs.io/', None),
    seaborn=('https://seaborn.pydata.org/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
)


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True