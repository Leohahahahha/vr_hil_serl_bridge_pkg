"""Fixed-layout shared-memory protocol for the legacy DM-Tac W SDK.

The SDK process runs under Python 3.11 while the ROS 2 process runs under
Python 3.12.  They exchange only primitive metadata and a packed byte payload
through two mmap files in ``/dev/shm``.  No SDK or ROS package crosses the
Python ABI boundary.
"""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


MAGIC = b"DMTACW01"
PROTOCOL_VERSION = 1
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
OUTPUT_MODE_FULL = "full"
OUTPUT_MODE_SHEAR_DEPTH = "shear_depth"
OUTPUT_MODES = (OUTPUT_MODE_FULL, OUTPUT_MODE_SHEAR_DEPTH)

# ``schema_version`` describes the bytes recorded in Zarr, independently of
# the mmap transport protocol version above. Schema v2 is already used by a
# different DM-Tac layout in this repository, so the compact W layout is v3.
_SCHEMA_VERSION_BY_MODE = {
    OUTPUT_MODE_FULL: 1,
    OUTPUT_MODE_SHEAR_DEPTH: 3,
}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    dtype: np.dtype
    encoding: str
    shape: tuple[int, ...]
    channels: int
    offset: int
    nbytes: int

    @property
    def step(self) -> int:
        return IMAGE_WIDTH * self.channels * self.dtype.itemsize


_MODALITY_DEFINITIONS = {
    "raw_image": (
        np.dtype(np.uint8),
        "mono8",
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        1,
    ),
    "deformation2d": (
        np.dtype("<f4"),
        "32FC2",
        (IMAGE_HEIGHT, IMAGE_WIDTH, 2),
        2,
    ),
    "normal": (
        np.dtype("<f4"),
        "32FC1",
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        1,
    ),
    "shear": (
        np.dtype("<f4"),
        "32FC2",
        (IMAGE_HEIGHT, IMAGE_WIDTH, 2),
        2,
    ),
    "depth": (
        np.dtype("<f4"),
        "32FC1",
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        1,
    ),
}


def normalize_output_mode(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"DM-Tac output_mode must be a string, got {type(value).__name__}")
    mode = value.strip().lower()
    if mode not in OUTPUT_MODES:
        raise ValueError(
            f"unsupported DM-Tac output_mode={value!r}; expected one of {OUTPUT_MODES}"
        )
    return mode


def schema_version_for_mode(output_mode: str) -> int:
    return _SCHEMA_VERSION_BY_MODE[normalize_output_mode(output_mode)]


def _make_specs(names: tuple[str, ...]) -> tuple[ModalitySpec, ...]:
    specs: list[ModalitySpec] = []
    offset = 0
    for name in names:
        dtype, encoding, shape, channels = _MODALITY_DEFINITIONS[name]
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        specs.append(
            ModalitySpec(
                name=name,
                dtype=dtype,
                encoding=encoding,
                shape=shape,
                channels=channels,
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset += nbytes
    return tuple(specs)


_MODALITY_NAMES_BY_MODE = {
    OUTPUT_MODE_FULL: (
        "raw_image",
        "deformation2d",
        "normal",
        "shear",
        "depth",
    ),
    OUTPUT_MODE_SHEAR_DEPTH: ("shear", "depth"),
}
_MODALITY_SPECS_BY_MODE = {
    mode: _make_specs(names) for mode, names in _MODALITY_NAMES_BY_MODE.items()
}


def get_modality_specs(output_mode: str = OUTPUT_MODE_FULL) -> tuple[ModalitySpec, ...]:
    return _MODALITY_SPECS_BY_MODE[normalize_output_mode(output_mode)]


def get_payload_bytes(output_mode: str = OUTPUT_MODE_FULL) -> int:
    return sum(spec.nbytes for spec in get_modality_specs(output_mode))


# Backwards-compatible aliases: existing schema-v1 callers continue to mean
# the five-modality/full layout unless they explicitly select an output mode.
MODALITY_SPECS = get_modality_specs(OUTPUT_MODE_FULL)
MODALITY_BY_NAME = {spec.name: spec for spec in MODALITY_SPECS}
PAYLOAD_BYTES = get_payload_bytes(OUTPUT_MODE_FULL)
PACKED_IMAGE_ENCODING = "8UC1"

# magic, protocol version, header size, session id, seqlock sequence, software
# frame index, capture start ns, capture end ns, packed payload size.
HEADER_STRUCT = struct.Struct("<8sIIQQQQQQ")
HEADER_BYTES = HEADER_STRUCT.size
FRAME_BYTES = HEADER_BYTES + PAYLOAD_BYTES
SEQUENCE_OFFSET = struct.calcsize("<8sIIQ")
FRAME_METADATA_OFFSET = struct.calcsize("<8sIIQQ")
SEQUENCE_STRUCT = struct.Struct("<Q")
FRAME_METADATA_STRUCT = struct.Struct("<QQQ")


def get_frame_bytes(output_mode: str = OUTPUT_MODE_FULL) -> int:
    return HEADER_BYTES + get_payload_bytes(output_mode)


@dataclass(frozen=True)
class FrameSnapshot:
    session_id: int
    sequence: int
    frame_idx: int
    capture_start_ns: int
    capture_end_ns: int
    payload: bytes

    @property
    def capture_mid_ns(self) -> int:
        return (self.capture_start_ns + self.capture_end_ns) // 2


def packed_layout_metadata(
    output_mode: str = OUTPUT_MODE_FULL,
) -> dict[str, int]:
    """Return the fixed schema metadata stored with every recorded packed frame."""
    mode = normalize_output_mode(output_mode)
    specs = get_modality_specs(mode)
    layout: dict[str, int] = {
        "schema_version": schema_version_for_mode(mode),
        "byte_order_little_endian": 1,
        "packed_frame_bytes": get_payload_bytes(mode),
    }
    for spec in specs:
        layout[f"{spec.name}_start"] = spec.offset
        layout[f"{spec.name}_len"] = spec.nbytes
        layout[f"{spec.name}_height"] = IMAGE_HEIGHT
        layout[f"{spec.name}_width"] = IMAGE_WIDTH
        layout[f"{spec.name}_channels"] = spec.channels
        layout[f"{spec.name}_itemsize"] = spec.dtype.itemsize
    return layout


def capture_wall_ns_to_monotonic_sec(
    capture_wall_ns: int,
    *,
    receive_wall_ns: int,
    receive_monotonic_ns: int,
) -> float:
    """Map a same-host wall-clock capture timestamp into the monotonic domain.

    The mapping is sampled at callback receipt.  It removes ROS/DDS delivery
    delay while keeping the recorder grid, action events and robot-state
    timestamps in one monotonic clock domain.
    """
    if capture_wall_ns <= 0 or receive_wall_ns <= 0 or receive_monotonic_ns <= 0:
        raise ValueError("capture/receive timestamps must be positive")
    transport_ns = receive_wall_ns - capture_wall_ns
    return float(receive_monotonic_ns - transport_ns) * 1e-9


def validate_protocol_constants() -> None:
    if len(MAGIC) != 8:
        raise RuntimeError("DM-Tac W IPC magic must be exactly eight bytes")
    if PAYLOAD_BYTES != 1_920_000:
        raise RuntimeError(f"unexpected DM-Tac W payload size: {PAYLOAD_BYTES}")
    compact_bytes = get_payload_bytes(OUTPUT_MODE_SHEAR_DEPTH)
    if compact_bytes != 921_600:
        raise RuntimeError(f"unexpected DM-Tac W shear/depth payload size: {compact_bytes}")
    if MODALITY_BY_NAME["normal"].shape != (240, 320):
        raise RuntimeError("DM-Tac W normal must be a single-channel 240x320 image")


def pack_modalities(
    arrays: Mapping[str, np.ndarray],
    output_mode: str = OUTPUT_MODE_FULL,
) -> bytes:
    """Strictly validate and pack one sensor's selected SDK outputs."""
    specs = get_modality_specs(output_mode)
    spec_by_name = {spec.name: spec for spec in specs}
    payload_bytes = get_payload_bytes(output_mode)
    missing = [spec.name for spec in specs if spec.name not in arrays]
    extras = sorted(set(arrays) - set(spec_by_name))
    if missing or extras:
        raise ValueError(f"DM-Tac modalities mismatch: missing={missing}, extras={extras}")

    payload = bytearray(payload_bytes)
    for spec in specs:
        array = np.asarray(arrays[spec.name])
        if tuple(array.shape) != spec.shape:
            raise ValueError(
                f"{spec.name} shape mismatch: expected={spec.shape}, actual={array.shape}"
            )
        if array.dtype != spec.dtype:
            raise ValueError(
                f"{spec.name} dtype mismatch: expected={spec.dtype}, actual={array.dtype}"
            )
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        segment = memoryview(array).cast("B")
        start = spec.offset
        payload[start : start + spec.nbytes] = segment
    return bytes(payload)


def unpack_modalities(
    payload: bytes | bytearray | memoryview,
    output_mode: str = OUTPUT_MODE_FULL,
) -> dict[str, np.ndarray]:
    """Return read-only NumPy views backed by a packed payload."""
    specs = get_modality_specs(output_mode)
    payload_bytes = get_payload_bytes(output_mode)
    if len(payload) != payload_bytes:
        raise ValueError(
            f"DM-Tac payload size mismatch: expected={payload_bytes}, actual={len(payload)}"
        )
    result: dict[str, np.ndarray] = {}
    for spec in specs:
        view = memoryview(payload)[spec.offset : spec.offset + spec.nbytes]
        result[spec.name] = np.frombuffer(view, dtype=spec.dtype).reshape(spec.shape)
    return result


def payload_segment(
    payload: bytes,
    spec: ModalitySpec,
    output_mode: str = OUTPUT_MODE_FULL,
) -> bytes:
    payload_bytes = get_payload_bytes(output_mode)
    if len(payload) != payload_bytes:
        raise ValueError(
            f"DM-Tac payload size mismatch: expected={payload_bytes}, actual={len(payload)}"
        )
    return payload[spec.offset : spec.offset + spec.nbytes]


@dataclass(frozen=True)
class _HeaderSnapshot:
    magic: bytes
    version: int
    header_bytes: int
    session_id: int
    sequence: int
    frame_idx: int
    capture_start_ns: int
    capture_end_ns: int
    payload_bytes: int


class MMapFrameWriter:
    """Single-writer side of a fixed shared-memory frame file."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: int,
        output_mode: str = OUTPUT_MODE_FULL,
    ) -> None:
        if session_id <= 0:
            raise ValueError("DM-Tac IPC session_id must be positive")
        self.path = Path(path)
        self.session_id = session_id
        self.output_mode = normalize_output_mode(output_mode)
        self.payload_bytes = get_payload_bytes(self.output_mode)
        self.frame_bytes = get_frame_bytes(self.output_mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self._fd, self.frame_bytes)
        self._mmap = mmap.mmap(self._fd, self.frame_bytes, access=mmap.ACCESS_WRITE)
        self._sequence = 0

        # Layout and session fields are immutable after initialization. During
        # streaming only the aligned sequence word and frame metadata change,
        # which prevents a reader from seeing a torn payload-size field.
        HEADER_STRUCT.pack_into(
            self._mmap,
            0,
            MAGIC,
            PROTOCOL_VERSION,
            HEADER_BYTES,
            self.session_id,
            0,
            0,
            0,
            0,
            self.payload_bytes,
        )

    def write(
        self,
        *,
        frame_idx: int,
        capture_start_ns: int,
        capture_end_ns: int,
        payload: bytes,
    ) -> int:
        if len(payload) != self.payload_bytes:
            raise ValueError(
                "DM-Tac payload size mismatch: "
                f"expected={self.payload_bytes}, actual={len(payload)}"
            )
        if capture_start_ns <= 0 or capture_end_ns < capture_start_ns:
            raise ValueError(
                f"invalid capture interval: {capture_start_ns}..{capture_end_ns}"
            )

        # Odd sequence means a write is in progress.  The reader copies only
        # when the same even sequence is observed before and after the payload.
        odd_sequence = self._sequence + 1
        SEQUENCE_STRUCT.pack_into(self._mmap, SEQUENCE_OFFSET, odd_sequence)
        FRAME_METADATA_STRUCT.pack_into(
            self._mmap,
            FRAME_METADATA_OFFSET,
            frame_idx,
            capture_start_ns,
            capture_end_ns,
        )
        self._mmap[HEADER_BYTES : self.frame_bytes] = payload
        self._sequence = odd_sequence + 1
        SEQUENCE_STRUCT.pack_into(self._mmap, SEQUENCE_OFFSET, self._sequence)
        return self._sequence

    def close(self) -> None:
        self._mmap.close()
        os.close(self._fd)

    def __enter__(self) -> "MMapFrameWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


class MMapFrameReader:
    """Single-reader side; opens lazily while the SDK worker starts."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: int,
        output_mode: str = OUTPUT_MODE_FULL,
        read_retries: int = 3,
        max_consecutive_layout_mismatches: int = 20,
    ) -> None:
        if session_id <= 0:
            raise ValueError("DM-Tac IPC session_id must be positive")
        if read_retries <= 0:
            raise ValueError("DM-Tac IPC read_retries must be positive")
        if max_consecutive_layout_mismatches <= 0:
            raise ValueError(
                "DM-Tac IPC max_consecutive_layout_mismatches must be positive"
            )
        self.path = Path(path)
        self.session_id = session_id
        self.output_mode = normalize_output_mode(output_mode)
        self.payload_bytes = get_payload_bytes(self.output_mode)
        self.frame_bytes = get_frame_bytes(self.output_mode)
        self.read_retries = read_retries
        self.max_consecutive_layout_mismatches = max_consecutive_layout_mismatches
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._mapping_identity = "unopened"
        self._consecutive_layout_mismatches = 0
        self._last_layout_diagnostic: str | None = None

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    @property
    def consecutive_layout_mismatches(self) -> int:
        return self._consecutive_layout_mismatches

    @property
    def last_layout_diagnostic(self) -> str | None:
        return self._last_layout_diagnostic

    @staticmethod
    def _unpack_header(raw: bytes) -> _HeaderSnapshot:
        return _HeaderSnapshot(*HEADER_STRUCT.unpack(raw))

    def _record_layout_mismatch(self, detail: str) -> None:
        self._consecutive_layout_mismatches += 1
        self._last_layout_diagnostic = (
            f"{detail}; path={self.path}, mapping={self._mapping_identity}, "
            f"expected_session={self.session_id}, output_mode={self.output_mode}, "
            f"expected_header={HEADER_BYTES}, expected_payload={self.payload_bytes}, "
            f"consecutive={self._consecutive_layout_mismatches}/"
            f"{self.max_consecutive_layout_mismatches}"
        )
        if (
            self._consecutive_layout_mismatches
            >= self.max_consecutive_layout_mismatches
        ):
            raise RuntimeError(self._last_layout_diagnostic)

    def _clear_layout_mismatch(self) -> None:
        self._consecutive_layout_mismatches = 0
        self._last_layout_diagnostic = None

    def _record_current_session_size_mismatch(self, actual_bytes: int) -> None:
        if actual_bytes < HEADER_BYTES:
            return
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(HEADER_BYTES)
        except (FileNotFoundError, OSError):
            return
        if len(raw) != HEADER_BYTES:
            return
        header = self._unpack_header(raw)
        if header.magic == MAGIC and header.session_id == self.session_id:
            self._record_layout_mismatch(
                f"DM-Tac IPC file-size mismatch: actual={actual_bytes}, "
                f"expected={self.frame_bytes}"
            )

    def open_if_ready(self) -> bool:
        if self._mmap is not None:
            return True
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        if stat.st_size != self.frame_bytes:
            self._record_current_session_size_mismatch(stat.st_size)
            return False
        fd = os.open(self.path, os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            if stat.st_size != self.frame_bytes:
                os.close(fd)
                self._record_current_session_size_mismatch(stat.st_size)
                return False
            mapped = mmap.mmap(fd, self.frame_bytes, access=mmap.ACCESS_READ)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mmap = mapped
        self._mapping_identity = f"dev={stat.st_dev},ino={stat.st_ino},bytes={stat.st_size}"
        return True

    def _stable_header_is_valid(self, header: _HeaderSnapshot) -> bool:
        # A valid but foreign session is a stale file from another worker run,
        # not a layout error. The current worker will shortly initialize it.
        if header.magic == MAGIC and header.session_id != self.session_id:
            self._clear_layout_mismatch()
            return False

        problems: list[str] = []
        if header.magic != MAGIC:
            problems.append(f"magic={header.magic!r}/{MAGIC!r}")
        if header.version != PROTOCOL_VERSION:
            problems.append(f"version={header.version}/{PROTOCOL_VERSION}")
        if header.header_bytes != HEADER_BYTES:
            problems.append(f"header={header.header_bytes}/{HEADER_BYTES}")
        if header.payload_bytes != self.payload_bytes:
            problems.append(f"payload={header.payload_bytes}/{self.payload_bytes}")
        if header.session_id != self.session_id:
            problems.append(f"session={header.session_id}/{self.session_id}")
        if problems:
            self._record_layout_mismatch(
                "DM-Tac IPC stable layout mismatch: " + ", ".join(problems)
            )
            return False
        return True

    def read_new(self, last_sequence: int) -> FrameSnapshot | None:
        if not self.open_if_ready():
            return None
        assert self._mmap is not None

        for _attempt in range(self.read_retries):
            raw_before = bytes(self._mmap[:HEADER_BYTES])
            before = self._unpack_header(raw_before)
            if before.sequence % 2:
                continue

            # Validate only after observing the exact same even header twice.
            # A torn header is retried and never treated as a protocol error.
            raw_confirm = bytes(self._mmap[:HEADER_BYTES])
            confirm = self._unpack_header(raw_confirm)
            if raw_before != raw_confirm or confirm.sequence % 2:
                continue
            if not self._stable_header_is_valid(confirm):
                return None
            if confirm.sequence == last_sequence or confirm.capture_end_ns == 0:
                self._clear_layout_mismatch()
                return None

            payload = bytes(self._mmap[HEADER_BYTES : self.frame_bytes])
            raw_after = bytes(self._mmap[:HEADER_BYTES])
            after = self._unpack_header(raw_after)
            if raw_after != raw_confirm or after.sequence % 2:
                continue
            if (
                confirm.capture_start_ns <= 0
                or confirm.capture_end_ns < confirm.capture_start_ns
            ):
                self._record_layout_mismatch(
                    "DM-Tac IPC stable capture interval is invalid: "
                    f"{confirm.capture_start_ns}..{confirm.capture_end_ns}"
                )
                return None
            self._clear_layout_mismatch()
            return FrameSnapshot(
                session_id=confirm.session_id,
                sequence=confirm.sequence,
                frame_idx=confirm.frame_idx,
                capture_start_ns=confirm.capture_start_ns,
                capture_end_ns=confirm.capture_end_ns,
                payload=payload,
            )
        return None

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._mapping_identity = "closed"

    def __enter__(self) -> "MMapFrameReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


validate_protocol_constants()
