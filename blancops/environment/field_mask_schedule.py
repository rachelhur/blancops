"""Time-windowed field-mask schedule for offline rollouts.

Holds the masking spec (a baseline rule plus time windows) and selects the
active rule for a given timestamp. Pure spec and selection: no ephemeris or
environment coupling, so it is independently testable. Resolution of a rule to a
positional boolean mask over a field-id array lives in `resolve_positional_mask`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID_MODES = ("mask", "keep_only")


@dataclass(frozen=True)
class MaskRule:
    """A single masking rule.

    mode "mask" masks the listed propids; mode "keep_only" masks the complement
    (every field whose propid is not listed).
    """
    propids: frozenset
    mode: str

    def __post_init__(self):
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"MaskRule.mode must be one of {_VALID_MODES}, got {self.mode!r}."
            )


@dataclass(frozen=True)
class MaskWindow:
    """A masking rule active over the half-open interval [start, end)."""
    start: float
    end: float
    rule: MaskRule

    def __post_init__(self):
        if not self.start < self.end:
            raise ValueError(
                f"MaskWindow requires start < end, got start={self.start}, "
                f"end={self.end}."
            )

    def contains(self, ts: float) -> bool:
        return self.start <= ts < self.end


@dataclass(frozen=True)
class FieldMaskSchedule:
    """Baseline mask plus time-windowed overrides.

    `active_rule(ts)` returns the first window whose [start, end) contains ts,
    else the baseline rule.
    """
    baseline: MaskRule
    windows: tuple = ()

    def rules(self) -> list:
        """All rules in the schedule (baseline first, then each window)."""
        return [self.baseline] + [w.rule for w in self.windows]

    def active_rule(self, ts: float) -> MaskRule:
        for window in self.windows:
            if window.contains(ts):
                return window.rule
        return self.baseline

    @classmethod
    def build(cls, baseline_propids, baseline_mode="mask",
              window_start=None, window_end=None,
              window_propids=None, window_mode="keep_only"):
        """Build a schedule from flat CLI-style args.

        Returns None when no baseline propids are given (no masking). A single
        window is added when start, end, and propids are all provided.
        """
        if not baseline_propids:
            return None
        baseline = MaskRule(propids=frozenset(baseline_propids), mode=baseline_mode)

        windows = ()
        if window_start is not None and window_end is not None and window_propids:
            windows = (
                MaskWindow(
                    start=float(window_start),
                    end=float(window_end),
                    rule=MaskRule(
                        propids=frozenset(window_propids), mode=window_mode
                    ),
                ),
            )
        return cls(baseline=baseline, windows=windows)


def resolve_positional_mask(rule, fids, field_ids_for_propids) -> np.ndarray:
    """Resolve a MaskRule to a positional boolean mask over `fids`.

    field_ids_for_propids : callable propids -> set[int]
        Typically `lookups.field_ids_for_propids`. The returned mask is True at
        the positions of fields to drop: the rule's propid fields for "mask",
        the complement for "keep_only".
    """
    masked_ids = field_ids_for_propids(rule.propids)
    positional = np.isin(fids, sorted(masked_ids))
    if rule.mode == "keep_only":
        positional = ~positional
    return positional
