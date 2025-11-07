"""
Package pipelines pour le preprocessing et l'évaluation des modèles.
"""

from pipelines.feature_selection import feature_selection
from pipelines.feature_engineering import feature_engineering
from pipelines.pipeline import pipeline
from pipelines.model_evaluation import evaluate_model, plot_confusion_matrix

__all__ = [
    'feature_selection',
    'feature_engineering',
    'pipeline',
    'evaluate_model',
    'plot_confusion_matrix'
]