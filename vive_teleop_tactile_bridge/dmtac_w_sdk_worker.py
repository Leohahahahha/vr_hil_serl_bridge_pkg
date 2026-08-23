#!/usr/bin/env python3
"""Python 3.11 worker for two legacy DM-Tac W SDK 0.1.4 sensors.

This file intentionally imports neither rclpy nor any ROS message package.  It
is executed with the SDK virtual-environment interpreter and writes the newest
left/right samples to fixed mmap files in /dev/shm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import platform
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# ``python -I`` excludes the script directory from sys.path.  Add only this
# installed package-share scripts directory so the adjacent protocol module is
# importable without inheriting ROS PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dmtac_w_ipc import (  # noqa: E402
    OUTPUT_MODE_FULL,
    OUTPUT_MODE_SHEAR_DEPTH,
    OUTPUT_MODES,
    MMapFrameWriter,
    get_payload_bytes,
    normalize_output_mode,
    pack_modalities,
)


STATUS_OK = 0
STATUS_RESETTING = 1
STATUS_DISCONNECTED = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-serial", required=True)
    parser.add_argument("--right-serial", required=True)
    parser.add_argument("--left-frame-path", default="/dev/shm/dmtac_w_left.frame")
    parser.add_argument("--right-frame-path", default="/dev/shm/dmtac_w_right.frame")
    parser.add_argument("--session-id", required=True, type=int)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=60.0)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--show-sdk-fps", action="store_true")
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default=OUTPUT_MODE_FULL,
        help="SDK modalities to request and publish",
    )
    return parser.parse_args()


def _read_modalities(sensor: Any, output_mode: str) -> dict[str, np.ndarray]:
    if output_mode == OUTPUT_MODE_SHEAR_DEPTH:
        # Deliberately do not call the other three SDK getters in compact mode.
        return {
            "shear": np.asarray(sensor.getShear()),
            "depth": np.asarray(sensor.getDepth()),
        }
    if output_mode == OUTPUT_MODE_FULL:
        return {
            "raw_image": np.asarray(sensor.getRawImage()),
            "deformation2d": np.asarray(sensor.getDeformation2D()),
            "normal": np.asarray(sensor.getNormal()),
            "shear": np.asarray(sensor.getShear()),
            "depth": np.asarray(sensor.getDepth()),
        }
    raise ValueError(f"unsupported DM-Tac output_mode={output_mode!r}")


def _sample_sensor(sensor: Any, output_mode: str) -> tuple[int, int, bytes]:
    capture_start_ns = time.time_ns()
    arrays = _read_modalities(sensor, output_mode)
    capture_end_ns = time.time_ns()
    payload = pack_modalities(arrays, output_mode)
    return capture_start_ns, capture_end_ns, payload


def _sample_sensors_parallel(
    sensors: dict[str, Any], executor: ThreadPoolExecutor, output_mode: str
) -> dict[str, tuple[int, int, bytes]]:
    """Read left/right sensors concurrently; getters within one sensor stay serial."""
    futures = {
        side: executor.submit(_sample_sensor, sensor, output_mode)
        for side, sensor in sensors.items()
    }
    samples: dict[str, tuple[int, int, bytes]] = {}
    errors: list[tuple[str, BaseException]] = []
    for side, future in futures.items():
        try:
            samples[side] = future.result()
        except BaseException as exc:
            errors.append((side, exc))
    if errors:
        side, exc = errors[0]
        raise RuntimeError(f"{side} DM-Tac sampling failed: {exc!r}") from exc
    return samples


def _statuses(sensors: dict[str, Any]) -> dict[str, int]:
    return {side: int(sensor.getStatus()) for side, sensor in sensors.items()}


def _wait_until_ready(
    sensors: dict[str, Any],
    *,
    stop_event: threading.Event,
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    last_report: dict[str, int] | None = None
    while not stop_event.is_set():
        states = _statuses(sensors)
        if states != last_report:
            print(f"[SDK] sensor status: {states}", flush=True)
            last_report = states
        if all(state == STATUS_OK for state in states.values()):
            print("[SDK] both sensors are OK", flush=True)
            return
        if any(state == STATUS_DISCONNECTED for state in states.values()):
            raise RuntimeError(f"DM-Tac W disconnected while starting: {states}")
        invalid = {side: state for side, state in states.items() if state not in (0, 1, 2)}
        if invalid:
            raise RuntimeError(f"unknown DM-Tac W status values: {invalid}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"DM-Tac W did not finish automatic reset within {timeout_sec}s: {states}"
            )
        stop_event.wait(0.5)
    raise InterruptedError("worker stopped while waiting for sensors")


def _warm_up(
    sensors: dict[str, Any],
    *,
    cycles: int,
    stop_event: threading.Event,
    executor: ThreadPoolExecutor,
    output_mode: str,
) -> None:
    print(
        f"[SDK] warming up {cycles} cycle(s); these samples are discarded",
        flush=True,
    )
    for cycle in range(cycles):
        if stop_event.is_set():
            raise InterruptedError("worker stopped during warmup")
        started = time.monotonic()
        samples = _sample_sensors_parallel(sensors, executor, output_mode)
        pair_elapsed_ms = (time.monotonic() - started) * 1000.0
        for side, (capture_start_ns, capture_end_ns, _payload) in samples.items():
            elapsed_ms = (capture_end_ns - capture_start_ns) / 1_000_000.0
            print(
                f"[SDK] warmup cycle={cycle} side={side} elapsed_ms={elapsed_ms:.2f} "
                f"pair_ms={pair_elapsed_ms:.2f}",
                flush=True,
            )


def run(args: argparse.Namespace) -> int:
    output_mode = normalize_output_mode(args.output_mode)
    payload_bytes = get_payload_bytes(output_mode)
    if args.left_serial == args.right_serial:
        raise ValueError("left and right DM-Tac W serial numbers must differ")
    if args.session_id <= 0:
        raise ValueError("session-id must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.startup_timeout_sec <= 0:
        raise ValueError("startup-timeout-sec must be positive")
    if args.warmup_cycles < 0:
        raise ValueError("warmup-cycles must be non-negative")
    if Path(args.left_frame_path) == Path(args.right_frame_path):
        raise ValueError("left and right frame paths must differ")

    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"DM-Tac W SDK 0.1.4 requires x86_64, got {machine}")
    if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
        raise RuntimeError(
            "DM-Tac W SDK 0.1.4 requires Python 3.8 through 3.11; "
            f"worker is running {sys.version.split()[0]}"
        )
    sdk_version = importlib.metadata.version("dmrobotics")
    if sdk_version != "0.1.4":
        raise RuntimeError(
            f"this bridge layout requires dmrobotics 0.1.4, got {sdk_version}"
        )

    from dmrobotics import Sensor

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"[SDK] received signal {signum}; stopping", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    writers = {
        "left": MMapFrameWriter(
            args.left_frame_path,
            session_id=args.session_id,
            output_mode=output_mode,
        ),
        "right": MMapFrameWriter(
            args.right_frame_path,
            session_id=args.session_id,
            output_mode=output_mode,
        ),
    }
    sensors: dict[str, Any] = {}
    sampler_pool: ThreadPoolExecutor | None = None
    try:
        print(
            "[SDK] opening sensors; keep both tactile surfaces completely unloaded "
            "during automatic reset",
            flush=True,
        )
        sensors["left"] = Sensor(args.left_serial, KEEP_FPS_Print=args.show_sdk_fps)
        sensors["right"] = Sensor(args.right_serial, KEEP_FPS_Print=args.show_sdk_fps)

        reported = {
            side: str(sensor.getCameraID()) for side, sensor in sensors.items()
        }
        expected = {"left": args.left_serial, "right": args.right_serial}
        if reported != expected:
            raise RuntimeError(
                f"DM-Tac W serial mismatch: expected={expected}, reported={reported}"
            )

        _wait_until_ready(
            sensors,
            stop_event=stop_event,
            timeout_sec=args.startup_timeout_sec,
        )
        sampler_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dmtac-sensor")
        _warm_up(
            sensors,
            cycles=args.warmup_cycles,
            stop_event=stop_event,
            executor=sampler_pool,
            output_mode=output_mode,
        )

        print(
            f"[SDK] streaming at {args.fps:.3f} Hz; output_mode={output_mode} "
            f"payload={payload_bytes} bytes/side",
            flush=True,
        )
        period_sec = 1.0 / args.fps
        next_deadline = time.monotonic()
        frame_idx = 0
        report_started = time.monotonic()
        report_frames = 0
        status_check_period = max(1, int(round(args.fps * 2.0)))

        while not stop_event.is_set():
            pair_started = time.monotonic()
            try:
                samples = _sample_sensors_parallel(
                    sensors, sampler_pool, output_mode
                )
            except Exception:
                # A getter can fail if the device starts an internal reset. If
                # the status confirms that case, discard the whole pair and
                # resume only after both sensors are stable again. Other SDK
                # errors remain fatal so corrupted data is never hidden.
                states = _statuses(sensors)
                if any(state == STATUS_RESETTING for state in states.values()):
                    print(
                        f"[SDK] reset detected during sampling; discarding pair: {states}",
                        flush=True,
                    )
                    _wait_until_ready(
                        sensors,
                        stop_event=stop_event,
                        timeout_sec=args.startup_timeout_sec,
                    )
                    _warm_up(
                        sensors,
                        cycles=args.warmup_cycles,
                        stop_event=stop_event,
                        executor=sampler_pool,
                        output_mode=output_mode,
                    )
                    next_deadline = time.monotonic()
                    continue
                raise

            # Check after acquiring both sides and before committing either
            # mmap file. This prevents reset frames from entering the ROS
            # stream and keeps the left/right software frame index paired.
            states = _statuses(sensors)
            if any(state == STATUS_DISCONNECTED for state in states.values()):
                raise RuntimeError(f"DM-Tac W disconnected: {states}")
            invalid = {
                side: state
                for side, state in states.items()
                if state not in (STATUS_OK, STATUS_RESETTING, STATUS_DISCONNECTED)
            }
            if invalid:
                raise RuntimeError(f"unknown DM-Tac W status values: {invalid}")
            if any(state == STATUS_RESETTING for state in states.values()):
                print(
                    f"[SDK] reset detected after sampling; discarding pair: {states}",
                    flush=True,
                )
                _wait_until_ready(
                    sensors,
                    stop_event=stop_event,
                    timeout_sec=args.startup_timeout_sec,
                )
                _warm_up(
                    sensors,
                    cycles=args.warmup_cycles,
                    stop_event=stop_event,
                    executor=sampler_pool,
                    output_mode=output_mode,
                )
                next_deadline = time.monotonic()
                continue

            frame_idx += 1
            for side, (capture_start_ns, capture_end_ns, payload) in samples.items():
                writers[side].write(
                    frame_idx=frame_idx,
                    capture_start_ns=capture_start_ns,
                    capture_end_ns=capture_end_ns,
                    payload=payload,
                )

            report_frames += 1
            if frame_idx % status_check_period == 0:
                print(f"[SDK] sensor status: {states}", flush=True)

            now = time.monotonic()
            if now - report_started >= 5.0:
                actual_fps = report_frames / (now - report_started)
                pair_ms = (now - pair_started) * 1000.0
                capture_ms = {
                    side: (end_ns - start_ns) / 1_000_000.0
                    for side, (start_ns, end_ns, _payload) in samples.items()
                }
                capture_mid_ns = {
                    side: (start_ns + end_ns) // 2
                    for side, (start_ns, end_ns, _payload) in samples.items()
                }
                lr_skew_ms = abs(capture_mid_ns["left"] - capture_mid_ns["right"]) / 1_000_000.0
                print(
                    f"[SDK] frame={frame_idx} actual_fps={actual_fps:.2f} "
                    f"last_pair_ms={pair_ms:.2f} left_ms={capture_ms['left']:.2f} "
                    f"right_ms={capture_ms['right']:.2f} lr_skew_ms={lr_skew_ms:.2f}",
                    flush=True,
                )
                report_started = now
                report_frames = 0

            next_deadline += period_sec
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                stop_event.wait(remaining)
            elif remaining < -period_sec:
                print(
                    f"[SDK] sampling fell behind by {-remaining * 1000.0:.2f} ms; "
                    "resetting schedule",
                    flush=True,
                )
                next_deadline = time.monotonic()
        return 0
    finally:
        if sampler_pool is not None:
            sampler_pool.shutdown(wait=True, cancel_futures=True)
        for side, sensor in reversed(tuple(sensors.items())):
            try:
                sensor.disconnect()
                print(f"[SDK] disconnected {side}", flush=True)
            except Exception as exc:
                print(f"[SDK] disconnect failed for {side}: {exc!r}", flush=True)
        for writer in writers.values():
            writer.close()


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except InterruptedError as exc:
        print(f"[SDK] stopped: {exc}", flush=True)
        raise SystemExit(0)
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
