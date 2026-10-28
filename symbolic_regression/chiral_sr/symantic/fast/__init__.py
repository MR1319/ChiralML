"""Memory-bounded, streaming implementation of SyMANTIC (regression and classification)."""
from .classification import SymanticClassifier
from .model import FastSymanticModel

__all__ = ["FastSymanticModel", "SymanticClassifier"]
