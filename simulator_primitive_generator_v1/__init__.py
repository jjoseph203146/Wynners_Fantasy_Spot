"""
Primitive Game-State Generator V1.

SHADOW / VALIDATION ONLY.

This package does not influence production forecasts, FanDuel projections,
player selection, lineup generation, solver behavior, or live services.
"""

from .core import (
    CONTRACT_VERSION,
    PrimitiveContractError,
    build_primitive_result,
    canonical_json,
    content_hash,
    validate_primitive_result,
)

__all__ = [
    "CONTRACT_VERSION",
    "PrimitiveContractError",
    "build_primitive_result",
    "canonical_json",
    "content_hash",
    "validate_primitive_result",
]
