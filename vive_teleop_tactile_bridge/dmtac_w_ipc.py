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


def _make_specs() -> tuple[ModalitySpec, ...]:
    definitions = (
        ("raw_image", np.dtype(np.uint8), "mono8", (IMAGE_HEIGHT, IMAGE_WIDTH), 1),
        (
            "deformation2d",
            np.dtype("<f4"),
            "32FC2",
            (IMAGE_HEIGHT, IMAGE_WIDTH, 2),
            2,
        ),
        ("normal", np.dtype("<f4"), "32FC1", (IMAGE_HEIGHT, IMAGE_WIDTH), 1),
        (
            "shear",
            np.dtype("<f4"),
            "32FC2",
            (IMAGE_HEIGHT, IMAGE_WIDTH, 2),
            2,
        ),
        ("depth", np.dtype("<f4"), "32FC1", (IMAGE_HEIGHT, IMAGE_WIDTH), 1),
    )
    specs: list[ModalitySpec] = []
    offset = 0
    for name, dtype, encoding, shape, channels in definitions:
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


MODALITY_SPECS = _make_specs()
MODALITY_BY_NAME = {spec.name: spec for spec in MODALITY_SPECS}
PAYLOAD_BYTES = sum(spec.nbytes for spec in MODALITY_SPECS)

# magic, protocol version, header size, session id, seqlock sequence, software
# frame index, capture start ns, capture end ns, packed payload size.
HEADER_STRUCT = struct.Struct("<8sIIQQQQQQ")
HEADER_BYTES = HEADER_STRUCT.size
FRAME_BYTES = HEADER_BYTES + PAYLOAD_BYTES


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


def validate_protocol_constants() -> None:
    if len(MAGIC) != 8:
        raise RuntimeError("DM-Tac W IPC magic must be exactly eight bytes")
    if PAYLOAD_BYTES != 1_920_000:
        raise RuntimeError(f"unexpected DM-Tac W payload size: {PAYLOAD_BYTES}")
    if MODALITY_BY_NAME["normal"].shape != (240, 320):
        raise RuntimeError("DM-Tac W normal must be a single-channel 240x320 image")


def pack_modalities(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Strictly validate and pack one sensor's five SDK outputs."""
    missing = [spec.name for spec in MODALITY_SPECS if spec.name not in arrays]
    extras = sorted(set(arrays) - set(MODALITY_BY_NAME))
    if missing or extras:
        raise ValueError(f"DM-Tac modalities mismatch: missing={missing}, extras={extras}")

    payload = bytearray(PAYLOAD_BYTES)
    for spec in MODALITY_SPECS:
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


def unpack_modalities(payload: bytes | bytearray | memoryview) -> dict[str, np.ndarray]:
    """Return read-only NumPy views backed by a packed payload."""
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(
            f"DM-Tac payload size mismatch: expected={PAYLOAD_BYTES}, actual={len(payload)}"
        )
    result: dict[str, np.ndarray] = {}
    for spec in MODALITY_SPECS:
        view = memoryview(payload)[spec.offset : spec.offset + spec.nbytes]
        result[spec.name] = np.frombuffer(view, dtype=spec.dtype).reshape(spec.shape)
    return result


def payload_segment(payload: bytes, spec: ModalitySpec) -> bytes:
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(
            f"DM-Tac payload size mismatch: expected={PAYLOAD_BYTES}, actual={len(payload)}"
        )
    return payload[spec.offset : spec.offset + spec.nbytes]


class MMapFrameWriter:
    """Single-writer side of a fixed shared-memory frame file."""

    def __init__(self, path: str | Path, *, session_id: int) -> None:
        if session_id <= 0:
            raise ValueError("DM-Tac IPC session_id must be positive")
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self._fd, FRAME_BYTES)
        self._mmap = mmap.mmap(self._fd, FRAME_BYTES, access=mmap.ACCESS_WRITE)
        self._sequence = 0
        self._write_header(
            sequence=0,
            frame_idx=0,
            capture_start_ns=0,
            capture_end_ns=0,
        )

    def _write_header(
        self,
        *,
        sequence: int,
        frame_idx: int,
        capture_start_ns: int,
        capture_end_ns: int,
    ) -> None:
        HEADER_STRUCT.pack_into(
            self._mmap,
            0,
            MAGIC,
            PROTOCOL_VERSION,
            HEADER_BYTES,
            self.session_id,
            sequence,
            frame_idx,
            capture_start_ns,
            capture_end_ns,
            PAYLOAD_BYTES,
        )

    def write(
        self,
        *,
        frame_idx: int,
        capture_start_ns: int,
        capture_end_ns: int,
        payload: bytes,
    ) -> int:
        if len(payload) != PAYLOAD_BYTES:
            raise ValueError(
                f"DM-Tac payload size mismatch: expected={PAYLOAD_BYTES}, actual={len(payload)}"
            )
        if capture_start_ns <= 0 or capture_end_ns < capture_start_ns:
            raise ValueError(
                f"invalid capture interval: {capture_start_ns}..{capture_end_ns}"
            )

        # Odd sequence means a write is in progress.  The reader copies only
        # when the same even sequence is observed before and after the payload.
        odd_sequence = self._sequence + 1
        self._write_header(
            sequence=odd_sequence,
            frame_idx=frame_idx,
            capture_start_ns=capture_start_ns,
            capture_end_ns=capture_end_ns,
        )
        self._mmap[HEADER_BYTES:FRAME_BYTES] = payload
        self._sequence = odd_sequence + 1
        self._write_header(
            sequence=self._sequence,
            frame_idx=frame_idx,
            capture_start_ns=capture_start_ns,
            capture_end_ns=capture_end_ns,
        )
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

    def __init__(self, path: str | Path, *, session_id: int) -> None:
        if session_id <= 0:
            raise ValueError("DM-Tac IPC session_id must be positive")
        self.path = Path(path)
        self.session_id = session_id
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    def open_if_ready(self) -> bool:
        if self._mmap is not None:
            return True
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        if stat.st_size != FRAME_BYTES:
            return False
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mapped = mmap.mmap(fd, FRAME_BYTES, access=mmap.ACCESS_READ)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mmap = mapped
        return True

    @staticmethod
    def _unpack_header(raw: bytes) -> tuple[int, int, int, int, int]:
        (
            magic,
            version,
            header_bytes,
            session_id,
            sequence,
            frame_idx,
            capture_start_ns,
            capture_end_ns,
            payload_bytes,
        ) = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise RuntimeError(f"invalid DM-Tac IPC magic: {magic!r}")
        if version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"DM-Tac IPC version mismatch: expected={PROTOCOL_VERSION}, actual={version}"
            )
        if header_bytes != HEADER_BYTES or payload_bytes != PAYLOAD_BYTES:
            raise RuntimeError(
                "DM-Tac IPC layout mismatch: "
                f"header={header_bytes}/{HEADER_BYTES}, payload={payload_bytes}/{PAYLOAD_BYTES}"
            )
        return session_id, sequence, frame_idx, capture_start_ns, capture_end_ns

    def read_new(self, last_sequence: int) -> FrameSnapshot | None:
        if not self.open_if_ready():
            return None
        assert self._mmap is not None

        header_before = bytes(self._mmap[:HEADER_BYTES])
        # A file may exist from a previous run or may be observed between
        # ftruncate() and the worker's first header write. Treat it as not ready.
        if header_before[:8] != MAGIC:
            return None
        (
            session_id,
            sequence,
            frame_idx,
            capture_start_ns,
            capture_end_ns,
        ) = self._unpack_header(header_before)
        if session_id != self.session_id:
            return None
        if sequence == last_sequence or sequence % 2 or capture_end_ns == 0:
            return None

        payload = bytes(self._mmap[HEADER_BYTES:FRAME_BYTES])
        header_after = bytes(self._mmap[:HEADER_BYTES])
        after = self._unpack_header(header_after)
        if after != (
            session_id,
            sequence,
            frame_idx,
            capture_start_ns,
            capture_end_ns,
        ):
            return None
        if sequence % 2:
            return None
        return FrameSnapshot(
            session_id=session_id,
            sequence=sequence,
            frame_idx=frame_idx,
            capture_start_ns=capture_start_ns,
            capture_end_ns=capture_end_ns,
            payload=payload,
        )

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "MMapFrameReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


validate_protocol_constants()
