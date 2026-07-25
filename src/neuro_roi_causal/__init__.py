"""Functional ROI causal analysis utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interventions import InterventionAudit, apply_parcel_intervention

__all__ = ["InterventionAudit", "apply_parcel_intervention"]


def __getattr__(name: str):
    if name in __all__:
        from .interventions import InterventionAudit, apply_parcel_intervention

        return {
            "InterventionAudit": InterventionAudit,
            "apply_parcel_intervention": apply_parcel_intervention,
        }[name]
    raise AttributeError(name)
