"""Evaluación: particiones, métricas, modelos comparables y corredor de experimentos."""
from .metrics import confusion, evaluate
from .models import RandomForestModel, make_model
from .runner import (environment_info, paired_comparison, register_evaluation, run_experiment, save_results,
                     summarize)
from .splits import Split, make_splits, max_stratified_splits

__all__ = ["Split", "RandomForestModel", "confusion", "environment_info", "evaluate", "make_model",
           "make_splits", "max_stratified_splits", "paired_comparison", "register_evaluation", "run_experiment",
           "save_results", "summarize"]
