"""Gaussian process models."""

from .base import AbstractComputationallyAwareGP
from .cagp import ComputationallyAwareGP

__all__ = ["AbstractComputationallyAwareGP", "ComputationallyAwareGP"]
