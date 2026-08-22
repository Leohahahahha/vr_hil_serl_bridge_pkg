from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TACTILE_LEFT_REL = Path("tactile") / "tactile_left" / "data.zarr"
TACTILE_RIGHT_REL = Path("tactile") / "tactile_right" / "data.zarr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible raw tactile datasets into one raw dataset.")
    parser.add_argument(
        "--raw-root",
        action="append",
        required=True,
        help="Input raw dataset root. Pass once per source dataset, in the desired episode order.",
    )
    parser.add_argument("--out-root", required=True, help="Merged output raw dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Replace out-root if it already exists")
    args = parser.parse_args()

    result = merge_raw_tactile_datasets(
        raw_roots=[Path(p) for p in args.raw_root],
        out_root=Path(args.out_root),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def merge_raw_tactile_datasets(
    *,
    raw_roots: list[Path],
    out_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not raw_roots:
        raise ValueError("at least one --raw-root is required")

    raw_roots = [p.expanduser().resolve() for p in raw_roots]
    out_root = out_root.expanduser().resolve()
    tmp_root = out_root.with_name(f".{out_root.name}.tmp_merge")

    source_infos = [_load_json(root / "meta" / "info.json") for root in raw_roots]
    _check_compatible(raw_roots, source_infos)

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    if tmp_root.exists():
        if not overwrite:
            raise FileExistsError(f"{tmp_root} exists; pass --overwrite to remove the stale temp directory")
        shutil.rmtree(tmp_root)

    _make_output_dirs(tmp_root)

    first_info = dict(source_infos[0])
    storage = first_info.get("storage", {}) if isinstance(first_info.get("storage"), dict) else {}
    front_image_dir = Path(str(storage.get("image_dir", "image/front")))
    wrist_image_dir = Path(str(storage.get("wrist_image_dir", "image/wrist")))

    all_rows: list[dict[str, Any]] = []
    episode_infos: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    global_index = 0
    out_episode_index = 0

    try:
        for source_index, raw_root in enumerate(raw_roots):
            source_summary = {
                "source_index": int(source_index),
                "raw_root": str(raw_root),
                "episode_start": int(out_episode_index),
                "frame_start": int(global_index),
                "episodes": 0,
                "frames": 0,
            }
            image_index = _load_jsonl_map(raw_root / front_image_dir / "index.jsonl", "path")
            wrist_index = _load_jsonl_map(raw_root / wrist_image_dir / "index.jsonl", "path")
            tactile_left_meta = _load_tactile_meta(raw_root / "tactile" / "tactile_left" / "meta.jsonl")
            tactile_right_meta = _load_tactile_meta(raw_root / "tactile" / "tactile_right" / "meta.jsonl")

            left_zarr = _open_zarr(raw_root / TACTILE_LEFT_REL)
            right_zarr = _open_zarr(raw_root / TACTILE_RIGHT_REL)

            for raw_episode_path in _find_episode_parquets(raw_root):
                df = pd.read_parquet(raw_episode_path)
                if df.empty:
                    continue
                _require_columns(
                    df,
                    [
                        "episode_index",
                        "frame_index",
                        "index",
                        "image.path",
                        "wrist_image.path",
                        "tactile.left.zarr_index",
                        "tactile.right.zarr_index",
                    ],
                    raw_episode_path,
                )

                out_front_dir = tmp_root / front_image_dir / f"episode_{out_episode_index:06d}"
                out_wrist_dir = tmp_root / wrist_image_dir / f"episode_{out_episode_index:06d}"
                out_front_dir.mkdir(parents=True, exist_ok=True)
                out_wrist_dir.mkdir(parents=True, exist_ok=True)

                out_episode_rows: list[dict[str, Any]] = []
                left_indices = [int(v) for v in df["tactile.left.zarr_index"].to_list()]
                right_indices = [int(v) for v in df["tactile.right.zarr_index"].to_list()]
                left_batch = _zarr_batch(left_zarr, left_indices, TACTILE_LEFT_REL)
                right_batch = _zarr_batch(right_zarr, right_indices, TACTILE_RIGHT_REL)
                left_start = _append_zarr_batch(tmp_root / TACTILE_LEFT_REL, left_batch)
                right_start = _append_zarr_batch(tmp_root / TACTILE_RIGHT_REL, right_batch)

                with open(tmp_root / "tactile" / "tactile_left" / "meta.jsonl", "a", encoding="utf-8") as left_meta_out, open(
                    tmp_root / "tactile" / "tactile_right" / "meta.jsonl", "a", encoding="utf-8"
                ) as right_meta_out, open(tmp_root / front_image_dir / "index.jsonl", "a", encoding="utf-8") as front_index_out, open(
                    tmp_root / wrist_image_dir / "index.jsonl", "a", encoding="utf-8"
                ) as wrist_index_out:
                    for out_frame_index, (_, raw_row) in enumerate(df.iterrows()):
                        row = dict(raw_row)
                        old_front_rel = str(row["image.path"])
                        old_wrist_rel = str(row["wrist_image.path"])
                        new_front_rel = front_image_dir / f"episode_{out_episode_index:06d}" / f"frame_{out_frame_index:06d}.png"
                        new_wrist_rel = wrist_image_dir / f"episode_{out_episode_index:06d}" / f"frame_{out_frame_index:06d}.png"

                        _copy_sidecar(raw_root, old_front_rel, tmp_root / new_front_rel)
                        _copy_sidecar(raw_root, old_wrist_rel, tmp_root / new_wrist_rel)

                        old_left_idx = int(row["tactile.left.zarr_index"])
                        old_right_idx = int(row["tactile.right.zarr_index"])
                        new_left_idx = int(left_start + out_frame_index)
                        new_right_idx = int(right_start + out_frame_index)

                        row["episode_index"] = int(out_episode_index)
                        row["frame_index"] = int(out_frame_index)
                        row["index"] = int(global_index)
                        row["image.path"] = str(new_front_rel)
                        row["wrist_image.path"] = str(new_wrist_rel)
                        row["tactile.left.zarr_path"] = str(TACTILE_LEFT_REL)
                        row["tactile.right.zarr_path"] = str(TACTILE_RIGHT_REL)
                        row["tactile.left.zarr_index"] = new_left_idx
                        row["tactile.right.zarr_index"] = new_right_idx

                        out_episode_rows.append(row)
                        all_rows.append(row)

                        front_index_out.write(
                            json.dumps(
                                _rewrite_image_index_record(
                                    image_index.get(old_front_rel),
                                    row,
                                    "image",
                                    new_front_rel,
                                    out_episode_index,
                                    out_frame_index,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        wrist_index_out.write(
                            json.dumps(
                                _rewrite_image_index_record(
                                    wrist_index.get(old_wrist_rel),
                                    row,
                                    "wrist_image",
                                    new_wrist_rel,
                                    out_episode_index,
                                    out_frame_index,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        left_meta_out.write(
                            json.dumps(
                                _rewrite_tactile_meta(
                                    tactile_left_meta.get(old_left_idx),
                                    row,
                                    new_left_idx,
                                    str(TACTILE_LEFT_REL),
                                    out_episode_index,
                                    out_frame_index,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        right_meta_out.write(
                            json.dumps(
                                _rewrite_tactile_meta(
                                    tactile_right_meta.get(old_right_idx),
                                    row,
                                    new_right_idx,
                                    str(TACTILE_RIGHT_REL),
                                    out_episode_index,
                                    out_frame_index,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        global_index += 1

                episode_path = tmp_root / "data" / "episodes" / f"episode_{out_episode_index:06d}.parquet"
                pd.DataFrame(out_episode_rows).to_parquet(episode_path, index=False)
                episode_info = {
                    "episode_index": int(out_episode_index),
                    "num_frames": int(len(out_episode_rows)),
                    "data_path": str(Path("data") / "episodes" / f"episode_{out_episode_index:06d}.parquet"),
                    "image_dir": str(front_image_dir / f"episode_{out_episode_index:06d}"),
                    "wrist_image_dir": str(wrist_image_dir / f"episode_{out_episode_index:06d}"),
                    "tactile_left_zarr": str(TACTILE_LEFT_REL),
                    "tactile_right_zarr": str(TACTILE_RIGHT_REL),
                    "task_index": int(out_episode_rows[0].get("task_index", first_info.get("task_index", 0))),
                    "task_description": str(first_info.get("task_description", "teleoperation")),
                    "source_raw_root": str(raw_root),
                    "source_episode_path": str(raw_episode_path.relative_to(raw_root)),
                }
                episode_infos.append(episode_info)
                with open(tmp_root / "meta" / "episodes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(episode_info, ensure_ascii=False) + "\n")

                source_summary["episodes"] = int(source_summary["episodes"]) + 1
                source_summary["frames"] = int(source_summary["frames"]) + len(out_episode_rows)
                out_episode_index += 1

            _append_file(raw_root / "raw" / "command_events.jsonl", tmp_root / "raw" / "command_events.jsonl")
            _append_file(raw_root / "raw" / "robot_state_events.jsonl", tmp_root / "raw" / "robot_state_events.jsonl")
            _append_file(raw_root / "raw" / "vr_controller_events.jsonl", tmp_root / "raw" / "vr_controller_events.jsonl")

            source_summary["episode_end"] = int(out_episode_index)
            source_summary["frame_end"] = int(global_index)
            source_summaries.append(source_summary)

        if not all_rows:
            raise RuntimeError("no rows were merged")
        pd.DataFrame(all_rows).to_parquet(tmp_root / "data" / "chunk-000" / "file-000.parquet", index=False)

        info = _merged_info(first_info, raw_roots, source_summaries, total_episodes=out_episode_index, total_frames=global_index)
        _write_json(tmp_root / "meta" / "info.json", info)
        _write_json(tmp_root / "meta" / "merge_sources.json", {"sources": source_summaries})
        first_config = raw_roots[0] / "meta" / "config.yaml"
        if first_config.exists():
            shutil.copy2(first_config, tmp_root / "meta" / "config.yaml")

        tmp_root.rename(out_root)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    return {
        "out_root": str(out_root),
        "total_episodes": int(out_episode_index),
        "total_frames": int(global_index),
        "sources": source_summaries,
    }


def _make_output_dirs(root: Path) -> None:
    for rel in [
        "data/chunk-000",
        "data/episodes",
        "image/front",
        "image/wrist",
        "tactile/tactile_left",
        "tactile/tactile_right",
        "meta",
        "raw",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _check_compatible(raw_roots: list[Path], infos: list[dict[str, Any]]) -> None:
    first = infos[0]
    keys = ["dataset_type", "tactile_msg_package", "fps", "state_dim", "action_dim"]
    for root, info in zip(raw_roots[1:], infos[1:]):
        for key in keys:
            if info.get(key) != first.get(key):
                raise ValueError(f"{root} meta/info.json {key}={info.get(key)!r} != {first.get(key)!r}")


def _merged_info(
    first_info: dict[str, Any],
    raw_roots: list[Path],
    source_summaries: list[dict[str, Any]],
    *,
    total_episodes: int,
    total_frames: int,
) -> dict[str, Any]:
    info = dict(first_info)
    info["merged_from_raw_roots"] = [str(root) for root in raw_roots]
    info["merge_sources"] = source_summaries
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["storage"] = {
        "master_parquet": "data/chunk-000/file-000.parquet",
        "per_episode_parquet_dir": "data/episodes",
        "image_dir": "image/front",
        "wrist_image_dir": "image/wrist",
        "tactile_left_zarr": str(TACTILE_LEFT_REL),
        "tactile_left_meta": "tactile/tactile_left/meta.jsonl",
        "tactile_right_zarr": str(TACTILE_RIGHT_REL),
        "tactile_right_meta": "tactile/tactile_right/meta.jsonl",
        "raw_vr_controller_events": "raw/vr_controller_events.jsonl",
        "raw_command_events": "raw/command_events.jsonl",
        "raw_robot_state_events": "raw/robot_state_events.jsonl",
    }
    return info


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_episode_parquets(raw_root: Path) -> list[Path]:
    episode_dir = raw_root / "data" / "episodes"
    paths = sorted(episode_dir.glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode parquet files under {episode_dir}")
    return paths


def _require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _load_jsonl_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[str(rec.get(key))] = rec
    return out


def _load_tactile_meta(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["zarr_index"])] = rec
    return out


def _open_zarr(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    import zarr

    return zarr.open(str(path), mode="r")


def _zarr_batch(arr: Any, indices: list[int], label: Path) -> np.ndarray:
    if not indices:
        raise ValueError(f"empty zarr index list for {label}")
    values = []
    for idx in indices:
        if idx < 0 or idx >= int(arr.shape[0]):
            raise IndexError(f"{label} index {idx} out of range 0..{int(arr.shape[0]) - 1}")
        values.append(np.asarray(arr[idx]).reshape(-1))
    return np.stack(values, axis=0)


def _append_zarr_batch(path: Path, batch: np.ndarray) -> int:
    import zarr

    batch = np.asarray(batch)
    if batch.ndim != 2:
        raise ValueError(f"expected 2D tactile batch, got {batch.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        arr = zarr.open(str(path), mode="a")
        if int(arr.shape[1]) != int(batch.shape[1]):
            raise ValueError(f"{path} frame length mismatch: {arr.shape[1]} != {batch.shape[1]}")
        if np.dtype(arr.dtype) != np.dtype(batch.dtype):
            raise ValueError(f"{path} dtype mismatch: {arr.dtype} != {batch.dtype}")
        start = int(arr.shape[0])
        arr.resize((start + int(batch.shape[0]), int(batch.shape[1])))
        arr[start : start + int(batch.shape[0]), :] = batch
        return start

    arr = zarr.open(
        str(path),
        mode="w",
        shape=(int(batch.shape[0]), int(batch.shape[1])),
        chunks=(min(1024, int(batch.shape[0])), int(batch.shape[1])),
        dtype=batch.dtype,
    )
    arr[:, :] = batch
    return 0


def _copy_sidecar(raw_root: Path, old_rel: str, out_path: Path) -> None:
    src = Path(old_rel)
    if not src.is_absolute():
        src = raw_root / src
    if not src.exists():
        raise FileNotFoundError(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)


def _rewrite_image_index_record(
    rec: dict[str, Any] | None,
    row: dict[str, Any],
    prefix: str,
    new_rel: Path,
    episode_index: int,
    frame_index: int,
) -> dict[str, Any]:
    out = dict(rec or {})
    out["episode_index"] = int(episode_index)
    out["record_frame_index"] = int(frame_index)
    out["path"] = str(new_rel)
    out.pop("tmp_path", None)
    for key in [
        "height",
        "width",
        "encoding",
        "step",
        "ros_stamp_sec",
        "ros_stamp_nanosec",
        "ros_stamp_float",
        "frame_id",
    ]:
        col = f"{prefix}.{key}"
        if col in row:
            value = row[col]
            if isinstance(value, (np.generic,)):
                value = value.item()
            out[key] = value
    return out


def _rewrite_tactile_meta(
    rec: dict[str, Any] | None,
    row: dict[str, Any],
    zarr_index: int,
    zarr_path: str,
    episode_index: int,
    frame_index: int,
) -> dict[str, Any]:
    out = dict(rec or {})
    out["episode_index"] = int(episode_index)
    out["frame_index"] = int(frame_index)
    out["zarr_index"] = int(zarr_index)
    out["zarr_path"] = zarr_path
    for key in ["timestamp", "wall_time"]:
        if key in row:
            out[key] = _json_scalar(row[key])
    return out


def _append_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fsrc, open(dst, "a", encoding="utf-8") as fdst:
        shutil.copyfileobj(fsrc, fdst)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
