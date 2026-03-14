from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
JPEG_SOI: Final[bytes] = b"\xff\xd8"
RIFF_SIGNATURE: Final[bytes] = b"RIFF"
WEBP_SIGNATURE: Final[bytes] = b"WEBP"


@dataclass(slots=True)
class ImageFingerprint:
    sha256: str
    phash: str | None
    width: int | None
    height: int | None
    extension: str


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_extension(data: bytes, fallback: str = "bin") -> str:
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data.startswith(JPEG_SOI):
        return "jpg"
    if data.startswith(RIFF_SIGNATURE) and data[8:12] == WEBP_SIGNATURE:
        return "webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return fallback


def sniff_image_size(data: bytes) -> tuple[int | None, int | None]:
    if data.startswith(PNG_SIGNATURE) and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data.startswith(JPEG_SOI):
        return _sniff_jpeg_size(data)
    if data.startswith(RIFF_SIGNATURE) and data[8:12] == WEBP_SIGNATURE:
        return _sniff_webp_size(data)
    return None, None


def _sniff_jpeg_size(data: bytes) -> tuple[int | None, int | None]:
    stream = io.BytesIO(data)
    stream.read(2)
    while True:
        marker_prefix = stream.read(1)
        if not marker_prefix:
            return None, None
        if marker_prefix != b"\xff":
            continue
        marker = stream.read(1)
        if not marker:
            return None, None
        while marker == b"\xff":
            marker = stream.read(1)
        if marker in {b"\xd8", b"\xd9"}:
            continue
        segment_length_bytes = stream.read(2)
        if len(segment_length_bytes) != 2:
            return None, None
        segment_length = struct.unpack(">H", segment_length_bytes)[0]
        if marker in {
            b"\xc0",
            b"\xc1",
            b"\xc2",
            b"\xc3",
            b"\xc5",
            b"\xc6",
            b"\xc7",
            b"\xc9",
            b"\xca",
            b"\xcb",
            b"\xcd",
            b"\xce",
            b"\xcf",
        }:
            precision = stream.read(1)
            if not precision:
                return None, None
            height_width = stream.read(4)
            if len(height_width) != 4:
                return None, None
            height, width = struct.unpack(">HH", height_width)
            return width, height
        stream.seek(segment_length - 2, io.SEEK_CUR)


def _sniff_webp_size(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 30:
        return None, None
    chunk_type = data[12:16]
    if chunk_type == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk_type == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk_type == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    return None, None


def compute_perceptual_hash(data: bytes) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("L").resize((9, 8))
        pixels = list(image.getdata())

    bits: list[int] = []
    for row in range(8):
        row_start = row * 9
        for column in range(8):
            left = pixels[row_start + column]
            right = pixels[row_start + column + 1]
            bits.append(1 if left > right else 0)

    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return f"{value:016x}"


def fingerprint_image(data: bytes) -> ImageFingerprint:
    width, height = sniff_image_size(data)
    extension = detect_extension(data)
    return ImageFingerprint(
        sha256=compute_sha256(data),
        phash=compute_perceptual_hash(data),
        width=width,
        height=height,
        extension=extension,
    )


def store_raw_image(storage_root: Path, sha256: str, extension: str, data: bytes) -> Path:
    bucket = storage_root / "raw" / sha256[:2]
    bucket.mkdir(parents=True, exist_ok=True)
    destination = bucket / f"{sha256}.{extension}"
    if not destination.exists():
        destination.write_bytes(data)
    return destination
