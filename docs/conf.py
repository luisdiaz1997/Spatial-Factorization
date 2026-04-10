# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -------------------------------------------------------

project = 'Spatial-Factorization'
copyright = '2025, Luis Chumpitaz Diaz'
author = 'Luis Chumpitaz Diaz'
release = '0.1.0'

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output ---------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
}

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- MathJax -------------------------------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            'E': ['\\mathbb{E}', 0],
            'KL': ['\\text{KL}', 0],
            'calL': ['\\mathcal{L}', 0],
            'R': ['\\mathbb{R}', 0],
        }
    }
}
