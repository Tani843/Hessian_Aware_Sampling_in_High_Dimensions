"""
Results processing and report generation module.

This module provides automated tools for processing benchmark results
and generating publication-ready reports and tables.
"""

from .result_generator import *

__all__ = [
    'ResultsProcessor',
    'generate_experiment_report',
    'create_latex_table',
    'compute_statistical_significance'
]