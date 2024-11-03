"""Module provides wrapper for benchmarking."""

from periomod.wrapper._basewrapper import BaseEvaluatorWrapper
from periomod.wrapper._wrapper import (
    BenchmarkWrapper,
    EvaluatorWrapper,
    load_benchmark,
    load_learners,
)

__all__ = [
    "BaseEvaluatorWrapper",
    "BenchmarkWrapper",
    "EvaluatorWrapper",
    "load_learners",
    "load_benchmark",
]
