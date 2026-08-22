from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tactile_csv_utils import (
    RAW_META_COLUMNS,
    layout_json,
    read_json,
    read_jsonl,
    sides_from_arg,
    tactile_value_columns,
    vector_value_dict,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw tactile zarr stores to wide CSV files.")
    parser.add_argument("--root", required=True, help="Raw dataset root")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <root>/exports/tactile_csv")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="Tactile side to export")
    parser.add_argument("--start-index", type=int, default=0, help="First zarr row index to export")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of zarr rows to export")
    args = parser.parse_args()

    result = export_raw_tactile_csv(
        root=Path(args.root),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        side=args.side,
        start_index=int(args.start_index),
        limit=None if args.limit is None else int(args.limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export_raw_tactile_csv(
    *,
    root: Path,
    out_dir: Path | None,
    side: str = "both",
    start_index: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    out_dir = (out_dir or (root / "exports" / "tactile_csv")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_json(root / "meta" / "info.json")
    msg_package = str(info.get("tactile_msg_package", ""))

    results = {}
    for side_name in sides_from_arg(side):
        results[side_name] = _export_side(
            root=root,
            out_dir=out_dir,
            side=side_name,
            msg_package=msg_package,
            start_index=start_index,
            limit=limit,
        )

    summary = {
        "root": str(root),
        "out_dir": str(out_dir),
        "tactile_msg_package": msg_package,
        "sides": results,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _export_side(
    *,
    root: Path,
    out_dir: Path,
    side: str,
    msg_package: str,
    start_index: int,
    limit: int | None,
) -> dict[str, Any]:
    import zarr

    zarr_path = root / "tactile" / f"tactile_{side}" / "data.zarr"
    meta_path = root / "tactile" / f"tactile_{side}" / "meta.jsonl"
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    arr = zarr.open(str(zarr_path), mode="r")
    meta_rows = read_jsonl(meta_path)
    meta_by_index = {int(row["zarr_index"]): row for row in meta_rows if row.get("zarr_index") is not None}

    total_rows = int(arr.shape[0])
    if total_rows <= 0:
        raise ValueError(f"{zarr_path} is empty")
    if start_index < 0 or start_index >= total_rows:
        raise IndexError(f"start-index {start_index} out of range [0, {total_rows})")
    end_index = total_rows if limit is None else min(total_rows, start_index + max(0, int(limit)))
    indices = list(range(start_index, end_index))

    first_meta = _first_meta(meta_by_index, indices)
    dim = int(np.asarray(arr[indices[0]]).reshape(-1).size)
    columns = tactile_value_columns(
        dim=dim,
        msg_package=msg_package,
        layout=first_meta.get("layout") if isinstance(first_meta.get("layout"), dict) else None,
    )

    rows = []
    for zarr_index in indices:
        meta = meta_by_index.get(zarr_index, {})
        values = np.asarray(arr[zarr_index]).reshape(-1)
        row = _raw_meta_row(side, zarr_index, meta)
        row.update(vector_value_dict(values, columns))
        rows.append(row)

    df = pd.DataFrame(rows, columns=RAW_META_COLUMNS + columns)
    csv_path = out_dir / f"raw_tactile_{side}.csv"
    df.to_csv(csv_path, index=False)
    return {
        "csv": str(csv_path),
        "zarr": str(zarr_path),
        "meta": str(meta_path),
        "rows": int(len(df)),
        "total_zarr_rows": total_rows,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "dim": dim,
        "zarr_dtype": str(arr.dtype),
        "value_columns": columns,
    }


def _first_meta(meta_by_index: dict[int, dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    for idx in indices:
        meta = meta_by_index.get(idx)
        if meta:
            return meta
    return {}


def _raw_meta_row(side: str, zarr_index: int, meta: dict[str, Any]) -> dict[str, Any]:
    row = {column: meta.get(column) for column in RAW_META_COLUMNS}
    row["side"] = side
    row["zarr_index"] = int(zarr_index)
    row["layout_json"] = layout_json(meta.get("layout"))
    return row


if __name__ == "__main__":
    main()
