from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_META_COLUMNS = [
    "side",
    "zarr_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "wall_time",
    "sensor_id",
    "sensor_index",
    "frame_idx",
    "capture_time",
    "repeated",
    "ros_stamp_float",
    "recv_time",
    "sync_dt",
    "data_len",
    "data_dtype",
    "source_dtype",
    "msg_package",
    "layout_json",
]

LEROBOT_META_COLUMNS = [
    "side",
    "global_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "dtype",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {e}") from e
    return rows


def sides_from_arg(value: str) -> list[str]:
    if value == "both":
        return ["left", "right"]
    if value in {"left", "right"}:
        return [value]
    raise ValueError(f"invalid side {value!r}; expected left, right, or both")


def layout_json(layout: Any) -> str:
    if not isinstance(layout, dict):
        return "{}"
    return json.dumps(layout, ensure_ascii=False, sort_keys=True)


def tactile_value_columns(
    *,
    dim: int,
    msg_package: str | None = None,
    tactile_profile: str | None = None,
    layout: dict[str, Any] | None = None,
) -> list[str]:
    layout = layout or {}
    package = str(msg_package or "")
    profile = str(tactile_profile or "")

    if _looks_like_tashan(package, profile, layout, dim):
        return _tashan_columns(dim, layout)
    if _looks_like_paxini(package, profile, layout, dim):
        return _paxini_columns(dim, layout)
    return _generic_columns(dim)


def vector_value_dict(values: Any, columns: list[str]) -> dict[str, float | int]:
    import numpy as np

    arr = np.asarray(values).reshape(-1)
    if arr.size != len(columns):
        raise ValueError(f"tactile vector dim {arr.size} != column count {len(columns)}")
    return {column: _python_scalar(value) for column, value in zip(columns, arr)}


def _looks_like_tashan(package: str, profile: str, layout: dict[str, Any], dim: int) -> bool:
    return (
        package == "tashan_tactile"
        or profile == "tashan"
        or "cap_len" in layout
        or dim == 25
    )


def _looks_like_paxini(package: str, profile: str, layout: dict[str, Any], dim: int) -> bool:
    return (
        package == "paxini_tactile"
        or profile == "paxini"
        or "force_raw_start" in layout
        or dim == 234
    )


def _tashan_columns(dim: int, layout: dict[str, Any]) -> list[str]:
    lengths = [
        ("cap", _int_layout(layout, "cap_len", 16 if dim == 25 else 0)),
        ("nf", _int_layout(layout, "nf_len", 2 if dim == 25 else 0)),
        ("tf", _int_layout(layout, "tf_len", 2 if dim == 25 else 0)),
        ("tf_dir", _int_layout(layout, "tf_dir_len", 2 if dim == 25 else 0)),
        ("s_prox", _int_layout(layout, "s_prox_len", 2 if dim == 25 else 0)),
        ("m_prox", _int_layout(layout, "m_prox_len", 1 if dim == 25 else 0)),
    ]
    columns = []
    for prefix, length in lengths:
        columns.extend(f"{prefix}_{idx:03d}" for idx in range(max(0, int(length))))
    return columns if len(columns) == dim else _generic_columns(dim)


def _paxini_columns(dim: int, layout: dict[str, Any]) -> list[str]:
    columns = _generic_columns(dim)
    force_start = _int_layout(layout, "force_raw_start", 0)
    force_len = _int_layout(layout, "force_raw_len", 3 if dim == 234 else 0)
    dist_start = _int_layout(layout, "distribution_raw_start", 3)
    dist_len = _int_layout(layout, "distribution_raw_len", max(0, dim - 3) if dim == 234 else 0)
    _write_range_names(columns, "force_raw", force_start, force_len)
    _write_range_names(columns, "distribution_raw", dist_start, dist_len)
    return columns


def _write_range_names(columns: list[str], prefix: str, start: int, length: int) -> None:
    if start < 0 or length < 0:
        return
    end = min(len(columns), start + length)
    for col_idx, value_idx in enumerate(range(start, end)):
        columns[value_idx] = f"{prefix}_{col_idx:03d}"


def _generic_columns(dim: int) -> list[str]:
    return [f"value_{idx:03d}" for idx in range(int(dim))]


def _int_layout(layout: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(layout.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _python_scalar(value: Any) -> float | int:
    try:
        item = value.item()
    except AttributeError:
        item = value
    if isinstance(item, bool):
        return int(item)
    if isinstance(item, int):
        return int(item)
    return float(item)
