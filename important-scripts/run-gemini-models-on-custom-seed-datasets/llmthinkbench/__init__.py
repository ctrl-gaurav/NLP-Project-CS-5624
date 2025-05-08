"""
LLMThinkBench: A framework for evaluating reasoning capabilities of Large Language Models.

This package provides tools to assess LLM performance on various reasoning tasks,
measuring accuracy, instruction following, and other metrics.
"""

__version__ = "0.1.4"
__author__ = "Gaurav Srivastava"

from .tasks.base_task import BaseTask
from .tasks.sorting_task import SortingTask
from .tasks.comparison_task import ComparisonTask
from .tasks.sum_task import SumTask
from .tasks.multiplication_task import MultiplicationTask
from .tasks.odd_count_task import OddCountTask
from .tasks.even_count_task import EvenCountTask
from .tasks.absolute_difference_task import AbsoluteDifferenceTask
from .tasks.division_task import DivisionTask
from .tasks.find_maximum_task import FindMaximumTask
from .tasks.find_minimum_task import FindMinimumTask
from .tasks.mean_task import MeanTask
from .tasks.median_task import MedianTask
from .tasks.mode_task import ModeTask
from .tasks.subtraction_task import SubtractionTask
from .models.model_handler import ModelHandler
from .utils.reporting import generate_final_report

# Convenience function to run evaluations
def evaluate(model_id, tasks=None, **kwargs):
    """
    Evaluate an LLM on reasoning tasks.
    
    Args:
        model_id (str): Hugging Face model ID
        tasks (list): List of tasks to evaluate (default: all available tasks)
        **kwargs: Additional arguments for evaluation
        
    Returns:
        dict: Evaluation results
    """
    from .cli import run_evaluation
    if tasks is None:
        tasks = ["sorting", "comparison", "sum", "multiplication", "odd_count", 
                "even_count", "absolute_difference", "division", "find_maximum", "find_minimum", "mean", "median", "mode", "subtraction"]
    return run_evaluation(model_id, tasks, **kwargs)