"""Install explicitly trusted FaceChannel assets without changing model contents."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024


def install_archive(archive_path, destination, expected_sha256):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A trusted archive SHA-256 is required.")
    digest = hashlib.sha256()
    with open(archive_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError("FaceChannel archive SHA-256 mismatch.")
    destination = Path(destination).resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(m.size for m in members) > MAX_EXPANDED_BYTES:
            raise ValueError("FaceChannel archive exceeds extraction limits.")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not path.parts or path.parts[0] != "TrainedNetworks"
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe FaceChannel archive member.")
            target = destination.joinpath(*path.parts)
            try:
                target.resolve().relative_to(destination)
            except ValueError:
                raise ValueError("FaceChannel destination escapes its asset directory.")
        # Stage only regular files/directories inside TrainedNetworks; trust comes from the digest.
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            staging = Path(temporary)
            for member in members:
                target = staging.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            files = [f for f in staging.rglob("*") if f.is_file()]
            for source in files:
                target = destination / source.relative_to(staging)
                if target.exists() and (not target.is_file() or target.read_bytes() != source.read_bytes()):
                    raise ValueError("Refusing to replace an existing FaceChannel asset.")
            for source in files:
                target = destination / source.relative_to(staging)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)


def acquire_assets(url, destination):
    digest = os.environ.get("SOLVER_FACECHANNEL_ARCHIVE_SHA256", "")
    if os.environ.get("SOLVER_ALLOW_MODEL_DOWNLOAD") != "1" or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise FileNotFoundError(
            "Install the original FaceChannel model assets. Automatic acquisition requires "
            "SOLVER_ALLOW_MODEL_DOWNLOAD=1 and a trusted SOLVER_FACECHANNEL_ARCHIVE_SHA256."
        )
    # Downloads are setup operations, never an unverified implicit inference side effect.
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "trainedNetworks.tar.xz"
        size = 0
        with urlopen(url, timeout=30) as source, target.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("FaceChannel download exceeds size limit.")
                output.write(chunk)
        install_archive(target, destination, digest)
