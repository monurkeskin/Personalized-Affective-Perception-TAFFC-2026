"""Explicit transport contracts; no model, policy or normalization implementation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from types import MappingProxyType
from typing import Mapping


def identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 200 or "\0" in value:
        raise ValueError(f"{name} must be a nonempty identifier of at most 200 characters")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _channel(value, name):
    if not isinstance(value, Mapping) or set(value) != {"Arousal", "Valance"}:
        raise ValueError(f"{name} requires separate Arousal and Valance fields")
    # Preserve supplied finite values: no inferred scaling, clipping or renormalization.
    return MappingProxyType({key: finite(value[key], f"{name}.{key}") for key in ("Arousal", "Valance")})


@dataclass(frozen=True)
class AffectSnapshot:
    session_id: str
    offer_id: str
    window_id: str
    model_id: str
    normalization_id: str
    observed_at: str
    raw: Mapping[str, float]
    normalized: Mapping[str, float]
    condition: str | None = None

    def __post_init__(self):
        for name in ("session_id", "offer_id", "window_id", "model_id", "normalization_id"):
            identifier(getattr(self, name), name)
        try:
            instant = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError("timezone required")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("observed_at requires an ISO timestamp with timezone") from exc
        object.__setattr__(self, "raw", _channel(self.raw, "raw"))
        object.__setattr__(self, "normalized", _channel(self.normalized, "normalized"))
        if self.condition is not None:
            identifier(self.condition, "condition")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError("Affect snapshot must be an object with raw and normalized channels")
        raw = data.get("raw", data.get("raw_predictions"))
        normalized = data.get("normalized", data.get("normalized_predictions"))
        if normalized is None:
            raise ValueError("Missing normalized affect channel; normalization is never inferred")
        if raw is None:
            raise ValueError("Missing raw affect channel")
        try:
            return cls(**{name: data[name] for name in ("session_id", "offer_id", "window_id", "model_id", "normalization_id", "observed_at")}, raw=raw, normalized=normalized, condition=data.get("condition"))
        except KeyError as exc:
            raise ValueError(f"Affect snapshot is missing {exc.args[0]}") from exc

    def to_dict(self):
        result = {name: getattr(self, name) for name in ("session_id", "offer_id", "window_id", "model_id", "normalization_id", "observed_at")}
        result.update(raw=dict(self.raw), normalized=dict(self.normalized))
        if self.condition is not None:
            result["condition"] = self.condition
        return result


class PolicyClock:
    """Both legacy method names expose the explicitly selected policy scalar."""
    def __init__(self, time_semantics, duration_seconds=600):
        if time_semantics != "elapsed_fraction":
            raise ValueError("Explicit time_semantics='elapsed_fraction' is required; remaining seconds are not policy time")
        self.duration_seconds = finite(duration_seconds, "duration_seconds")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self._fraction = 0.0

    def set_elapsed_fraction(self, value):
        value = finite(value, "elapsed_fraction")
        if not 0 <= value <= 1:
            raise ValueError("elapsed_fraction must lie in [0,1]")
        self._fraction = value

    def get_current_time(self):
        return self._fraction

    def get_remaining_time(self):
        return self._fraction

    def get_deadline(self):
        return self.duration_seconds

    def time_is_up(self):
        return self._fraction >= 1


class LegacyBid(list):
    """List-compatible bid with explicit original issue order and perspective."""
    def __init__(self, values, issue_order, domain_totals=None, *, perspective="Agent", bidder=None):
        self.issue_order = tuple(issue_order)
        if not self.issue_order or len(set(self.issue_order)) != len(self.issue_order) or not all(isinstance(key, str) and key for key in self.issue_order):
            raise ValueError("issue_order must contain distinct named issues")
        if isinstance(values, Mapping):
            if set(values) != set(self.issue_order):
                raise ValueError("Bid must describe exactly the configured named issues")
            values = [values[key] for key in self.issue_order]
        else:
            values = list(values)
        if len(values) != len(self.issue_order):
            raise ValueError("Bid length differs from issue_order")
        if perspective not in {"Agent", "Human"}:
            raise ValueError("Unknown bid perspective")
        self.domain_totals = dict(domain_totals) if domain_totals is not None else None
        if self.domain_totals is not None:
            if set(self.domain_totals) != set(self.issue_order):
                raise ValueError("domain_totals must describe exactly the configured issues")
            for key, value in zip(self.issue_order, values):
                total = self.domain_totals[key]
                if type(total) is not int or total < 0 or type(value) is not int or not 0 <= value <= total:
                    raise ValueError("Allocation values and totals must be legal integer counts")
        elif any(type(value) not in (str, int) for value in values):
            raise ValueError("Bid values must be categorical strings or integer counts")
        self.perspective = perspective
        self.bidder = bidder or perspective
        super().__init__(values)

    def get_bid(self, perspective=None):
        perspective = perspective or self.perspective
        if perspective not in {"Agent", "Human"}:
            raise ValueError("Unknown bid perspective")
        if perspective == self.perspective or self.domain_totals is None:
            return list(self)
        return [self.domain_totals[key] - value for key, value in zip(self.issue_order, self)]

    def get_bidder(self):
        return self.bidder

    def get_arguments(self):
        return []

    def to_named(self, perspective):
        return dict(zip(self.issue_order, self.get_bid(perspective)))

    def __deepcopy__(self, memo):
        return type(self)(self, self.issue_order, self.domain_totals, perspective=self.perspective, bidder=self.bidder)
