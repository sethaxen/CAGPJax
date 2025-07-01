"""Gaussian process models."""

from .base import AbstractComputationAwareGP
from .cagp import ComputationAwareGP

__all__ = ["AbstractComputationAwareGP", "ComputationAwareGP"]
