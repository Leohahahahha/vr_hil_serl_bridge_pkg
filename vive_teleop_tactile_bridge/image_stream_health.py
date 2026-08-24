"""Clock-domain mapping and health diagnostics for remote image streams."""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np


def normalize_qos_reliability(value: Any, *, field_name: str) -> str:
    reliability = str(value).strip().lower().replace("-", "_")
    if reliability not in {"reliable", "best_effort"}:
        raise ValueError(
            f"{field_name} must be 'reliable' or 'best_effort', got: {value!r}"
        )
    return reliability


def normalize_image_message_type(value: Any, *, field_name: str) -> str:
    message_type = str(value).strip().lower().replace("-", "_")
    aliases = {
        "image": "raw",
        "raw": "raw",
        "compressed": "compressed",
        "compressedimage": "compressed",
        "compressed_image": "compressed",
    }
    try:
        return aliases[message_type]
    except KeyError as exc:
        raise ValueError(
            f"{field_name} must be 'raw' or 'compressed', got: {value!r}"
        ) from exc


class RemoteImageStampMonitor:
    """Validate a remote camera stamp and map it into local monotonic time.

    ``receive_wall - header_stamp`` contains both one-way transport/processing
    delay and the remaining wall-clock offset between the publisher and
    recorder. It is therefore a health signal, not network-only latency.
    """

    def __init__(
        self,
        *,
        logger: Any,
        stream_name: str,
        expected_hz: float,
        max_apparent_age_sec: float,
        max_future_sec: float,
        reject_invalid: bool,
        diagnostics_interval_sec: float,
    ) -> None:
        self._logger = logger
        self._stream_name = stream_name
        self._expected_hz = max(0.0, float(expected_hz))
        self._max_apparent_age_sec = max(0.0, float(max_apparent_age_sec))
        self._max_future_sec = max(0.0, float(max_future_sec))
        self._reject_invalid = bool(reject_invalid)
        self._diagnostics_interval_sec = max(1.0, float(diagnostics_interval_sec))

        self._last_header_ns: Optional[int] = None
        self._last_log_monotonic_ns: Optional[int] = None
        self._last_log_received = 0
        self._last_log_rejected = 0
        self._last_log_fallback = 0
        self._received = 0
        self._accepted = 0
        self._rejected = 0
        self._fallback = 0
        self._duplicates = 0
        self._backwards = 0
        self._age_window: deque[float] = deque(maxlen=600)
        self._header_interval_window: deque[float] = deque(maxlen=600)

    def map_stamp(
        self,
        stamp: Any,
        *,
        receive_wall_ns: int,
        receive_monotonic_ns: int,
    ) -> Optional[float]:
        self._received += 1
        receive_monotonic = receive_monotonic_ns * 1e-9
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        reason: Optional[str] = None
        apparent_age_sec: Optional[float] = None

        if stamp_ns <= 0:
            reason = "zero_or_negative_header_stamp"
        else:
            apparent_age_sec = (receive_wall_ns - stamp_ns) * 1e-9
            self._age_window.append(apparent_age_sec)
            if self._last_header_ns is not None:
                delta_ns = stamp_ns - self._last_header_ns
                if delta_ns == 0:
                    self._duplicates += 1
                    reason = "duplicate_header_stamp"
                elif delta_ns < 0:
                    self._backwards += 1
                    reason = "backwards_header_stamp"
                else:
                    self._header_interval_window.append(delta_ns * 1e-9)
                    self._last_header_ns = stamp_ns
            else:
                self._last_header_ns = stamp_ns
            if reason is None and apparent_age_sec > self._max_apparent_age_sec:
                reason = "header_too_old_or_clocks_unsynchronised"
            if reason is None and apparent_age_sec < -self._max_future_sec:
                reason = "header_from_future_or_clocks_unsynchronised"

        if reason is None and apparent_age_sec is not None:
            self._accepted += 1
            sample_t: Optional[float] = float(receive_monotonic - apparent_age_sec)
        elif self._reject_invalid:
            self._rejected += 1
            sample_t = None
        else:
            self._fallback += 1
            sample_t = receive_monotonic

        self._maybe_log(receive_monotonic_ns, last_reason=reason)
        return sample_t

    def _maybe_log(self, now_monotonic_ns: int, *, last_reason: Optional[str]) -> None:
        if self._last_log_monotonic_ns is None:
            self._last_log_monotonic_ns = now_monotonic_ns
            self._last_log_received = self._received
            return
        elapsed_sec = (now_monotonic_ns - self._last_log_monotonic_ns) * 1e-9
        if elapsed_sec < self._diagnostics_interval_sec:
            return

        received_delta = self._received - self._last_log_received
        receive_hz = received_delta / elapsed_sec if elapsed_sec > 0.0 else 0.0
        if self._header_interval_window:
            intervals = np.asarray(self._header_interval_window, dtype=np.float64)
            median_interval = float(np.median(intervals))
            header_hz = 1.0 / median_interval if median_interval > 0.0 else 0.0
        else:
            header_hz = 0.0
        if self._age_window:
            ages_ms = np.asarray(self._age_window, dtype=np.float64) * 1000.0
            age_text = (
                f"age_ms[p50={np.percentile(ages_ms, 50):.1f},"
                f"p95={np.percentile(ages_ms, 95):.1f},"
                f"max={np.max(ages_ms):.1f}]"
            )
        else:
            age_text = "age_ms[N/A]"

        message = (
            f"[IMAGE_STAMP] stream={self._stream_name} receive_hz={receive_hz:.2f} "
            f"header_hz={header_hz:.2f} {age_text} received={self._received} "
            f"accepted={self._accepted} rejected={self._rejected} fallback={self._fallback} "
            f"duplicate={self._duplicates} backwards={self._backwards}"
        )
        unhealthy_rate = self._expected_hz > 0.0 and receive_hz < self._expected_hz * 0.80
        rejected_delta = self._rejected - self._last_log_rejected
        fallback_delta = self._fallback - self._last_log_fallback
        if last_reason is not None or rejected_delta or fallback_delta or unhealthy_rate:
            if last_reason is not None:
                message += f" last_reason={last_reason}"
            self._logger.warning(message)
        else:
            self._logger.info(message)

        self._last_log_monotonic_ns = now_monotonic_ns
        self._last_log_received = self._received
        self._last_log_rejected = self._rejected
        self._last_log_fallback = self._fallback
