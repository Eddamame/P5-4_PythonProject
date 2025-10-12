# app/modules/__init__.py
"""
Stock Market Analysis Modules Package
This package contains all analysis, visualization, and data handling modules.
"""

# Make submodules available for import
from . import data_handler
from . import data_fetcher
from . import metrics
from . import prediction
from . import visualization
from . import validation

__all__ = [
    'data_handler',
    'data_fetcher',
    'metrics',
    'prediction',
    'visualization',
    'validation'
]