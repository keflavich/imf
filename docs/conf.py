# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

#import os
#import sys
#sys.path.insert(0, os.path.abspath('../imf/'))


project = 'imf'
copyright = '2026, Adam Ginsburg, Sergey Koposov, Theo Richardson, Tiffany Christian'
author = 'Adam Ginsburg, Sergey Koposov, Theo Richardson, Tiffany Christian'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'nbsphinx']

nbsphinx_execute = 'always'
nbsphinx_timeout = 300
nbsphinx_allow_errors = False
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
    "--ExecutePreprocessor.skip_cells_with_tag=skip-execution",
]

autodoc_default_options = {'member-order': 'bysource'}
autodoc_inherit_docstrings = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
