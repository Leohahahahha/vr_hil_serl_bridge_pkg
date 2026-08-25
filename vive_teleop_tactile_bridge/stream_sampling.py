"""Pure helpers for fixed-rate sampling from timestamped ROS buffers."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, Optional, TypeVar


T = TypeVar("T")


def nearest_time_ordered(
    items: Iterable[T],
    target_time: float,
    *,
    accept: Optional[Callable[[T], bool]] = None,
) -> Optional[T]:
    """Return the nearest accepted item from an ascending-time collection.

    The search starts at the newest item and stops after crossing the target;
    every still older accepted item must then be farther away.  Ties prefer the
    newer item, which is desirable for a delayed fixed-rate sampling grid.
    """

    best: Optional[T] = None
    best_distance = float("inf")
    for item in reversed(items):  # type: ignore[arg-type]
        if accept is not None and not accept(item):
            continue
        item_time = float(getattr(item, "t"))
        distance = abs(item_time - float(target_time))
        # Reverse iteration visits newer samples first.  Keep that newer item
        # when decimal timestamps are mathematically tied but differ by a few
        # floating-point ulps (for example, 0.9 and 1.1 around 1.0).
        if distance < best_distance - 1e-12:
            best = item
            best_distance = distance
        if item_time <= target_time:
            break
    return best


def ros_message_stamp_ns(message: Any) -> int:
    """Return a ROS message header stamp as nanoseconds, or zero if absent."""

    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(
        getattr(stamp, "nanosec", 0)
    )


def has_usable_action_label(item: Any, target_time: float, max_dt_sec: float) -> bool:
    """Whether a buffered command is a valid action label for this target."""

    if item is None or abs(float(item.t) - float(target_time)) > float(max_dt_sec):
        return False
    payload = getattr(item, "payload", None)
    if not isinstance(payload, dict):
        return False
    action8 = payload.get("action8")
    try:
        return action8 is not None and len(action8) == 8
    except TypeError:
        return False


_STALE_DT_RE = re.compile(r"_dt_[+-]?\d+(?:\.\d+)?$")


def skip_reason_key(reason: str) -> str:
    """Collapse numeric sync offsets into stable per-source reason counters."""

    reason = str(reason).strip()
    components: list[str] = []
    for token in reason.split():
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if value == "ok":
            continue
        value = _STALE_DT_RE.sub("", value)
        components.append(f"{name}:{value}")
    if components:
        return "+".join(components)
    return _STALE_DT_RE.sub("", reason).replace(" ", "_") or "unknown"
