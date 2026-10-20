"""Verify explicitly configured local assets without acquiring or loading models."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def verify_assets(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Asset manifest must be an existing regular file.")
    manifest_metadata = path.stat()
    if getattr(manifest_metadata, "st_flags", 0) & 0x40000000:
        raise ValueError("Asset manifest is an offline placeholder; prepare it explicitly.")
    if manifest_metadata.st_size > 1024 * 1024:
        raise ValueError("Asset manifest exceeds its size bound.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Asset manifest must be an object.")
    for field in ("source_url", "license"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError("Asset manifest requires " + field + ".")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("Asset manifest needs a nonempty bounded files hash map.")
    root = path.parent.resolve()
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or name != PurePosixPath(name).as_posix()
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None):
            raise ValueError("Invalid relative asset path or SHA-256 hash.")
        target = root / name
        current = root
        for part in PurePosixPath(name).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Asset paths must not traverse symbolic links.")
        try:
            target.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Asset path leaves the manifest directory.") from exc
        if not target.is_file():
            raise ValueError("Missing regular asset: " + name)
        metadata = target.stat()
        if getattr(metadata, "st_flags", 0) & 0x40000000:
            raise ValueError("Asset is an offline placeholder; prepare it explicitly: " + name)
        checksum = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != expected.lower():
            raise ValueError("Asset hash mismatch: " + name)
    return root, manifest
