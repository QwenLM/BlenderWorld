"""Public submission contract: one ZIP with final.blend at its root per task."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

from .data import sha256

FORMAT = "blenderworld-submission-v1"
PROJECT = "final.blend"
MANIFEST = "submission.json"
MAX_FILES = 100_000
MAX_BYTES = 20 * 1024**3


def _name(name):
    if not name or "\\" in name or "\0" in name:
        raise ValueError("Invalid archive member path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        raise ValueError("Archive member escapes its root")
    return path.as_posix().rstrip("/")


def _copy(source, destination, limit):
    total = 0
    while block := source.read(1024 * 1024):
        total += len(block)
        if total > limit:
            raise ValueError("Submission file exceeds its declared size or size limit")
        destination.write(block)
    return total


@contextmanager
def _archive(source):
    """Validate paths and file types before reading submitted files."""
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > MAX_FILES:
                raise ValueError("Too many archive members")
            names, files, total = set(), {}, 0
            for member in members:
                name = _name(member.orig_filename)
                directory = member.is_dir()
                if name == "." and directory:
                    continue
                if name in names:
                    raise ValueError("Duplicate archive member")
                names.add(name)
                kind = stat.S_IFMT(member.external_attr >> 16)
                if kind not in (0, stat.S_IFDIR if directory else stat.S_IFREG):
                    raise ValueError("Archive links and special files are not supported")
                if not directory:
                    total += member.file_size
                    if member.file_size < 0 or total > MAX_BYTES:
                        raise ValueError("Archive exceeds the uncompressed size limit")
                    files[name] = (member.file_size, member)
            for name in names:
                if any(p.as_posix() in files for p in PurePosixPath(name).parents if p.as_posix() != "."):
                    raise ValueError("Archive contains a file/directory path conflict")
            yield files, lambda name: archive.open(files[name][1])
    except (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"Invalid submission ZIP: {type(error).__name__}") from error


@contextmanager
def _directory(source):
    files, total = {}, 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Submission contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Submission contains a special file")
        size = path.stat().st_size
        total += size
        if total > MAX_BYTES or len(files) >= MAX_FILES:
            raise ValueError("Submission exceeds the size or file-count limit")
        files[_name(path.relative_to(source).as_posix())] = (size, path)
    yield files, lambda name: files[name][1].open("rb")


def _check_manifest(files, read, task_id):
    if MANIFEST not in files:
        return None
    if files[MANIFEST][0] > 16 * 1024**2:
        raise ValueError("Submission manifest is too large")
    with read(MANIFEST) as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("task_id") != task_id:
        raise ValueError("Submission manifest format or task_id does not match")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or any(not isinstance(k, str) or not isinstance(v, str)
            or not re.fullmatch(r"[0-9a-f]{64}", v) for k, v in expected.items()):
        raise ValueError("Invalid submission file checksums")
    if set(expected) != set(files) - {MANIFEST}:
        raise ValueError("Submission manifest does not cover the delivered files")
    return expected


@contextmanager
def _payload(source, task_id):
    source = Path(source)
    if source.is_symlink():
        raise ValueError("Submission source must not be a symbolic link")
    if not source.is_dir() and not source.is_file():
        raise ValueError("Submission source does not exist")
    reader = _directory(source) if source.is_dir() else _archive(source)
    with reader as (files, read):
        if PROJECT not in files:
            raise ValueError("Expected final.blend at the submission root; package the output directory's contents without a wrapping folder")
        if not files[PROJECT][0]:
            raise ValueError("Submitted final.blend is empty")
        expected = _check_manifest(files, read, task_id)
        selected = {n: info for n, info in files.items() if n != MANIFEST}
        yield selected, read, expected


def _task_id(task_id):
    if not re.fullmatch(r"bw_\d{4}", task_id):
        raise ValueError("Use the dataset's bw_XXXX task ID")


def pack_submission(source, task_id, output):
    """Package a model's output directory (or standard ZIP) with checksums."""
    _task_id(task_id)
    source, output = Path(source), Path(output).absolute()
    if source.is_symlink():
        raise ValueError("Submission source must not be a symbolic link")
    source = source.resolve()
    if output.suffix.lower() != ".zip" or output.exists():
        raise ValueError("Output must be a new .zip file")
    if source.is_dir() and output.resolve().is_relative_to(source):
        raise ValueError("Write the submission ZIP outside the source directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
        temp = Path(temporary.name)
    try:
        with _payload(source, task_id) as (files, read, expected):
            hashes = {}
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, (size, _) in sorted(files.items()):
                    digest = hashlib.sha256()
                    with read(name) as stream, archive.open(name, "w", force_zip64=True) as target:
                        written = 0
                        while block := stream.read(1024 * 1024):
                            written += len(block)
                            if written > size:
                                raise ValueError("Source changed while packaging")
                            digest.update(block)
                            target.write(block)
                    if written != size:
                        raise ValueError("Truncated delivered file")
                    hashes[name] = digest.hexdigest()
                if expected is not None and hashes != expected:
                    raise ValueError("Submission checksum mismatch")
                archive.writestr(MANIFEST, json.dumps({"format": FORMAT, "task_id": task_id,
                    "files": hashes}, indent=2) + "\n")
            inspect_submission(temp, task_id)
            # Publish atomically without overwriting another completed submission.
            os.link(temp, output)
        return {"task_id": task_id, "output": str(output), "format": FORMAT,
                "project": PROJECT, "files": len(hashes), "sha256": sha256(output)}
    finally:
        temp.unlink(missing_ok=True)


def inspect_submission(archive, task_id):
    _task_id(task_id)
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("Expected a per-task submission ZIP")
    with _payload(archive, task_id) as (files, _, expected):
        return {"format": FORMAT, "project": PROJECT, "manifest_present": expected is not None,
                "files": len(files), "bytes": sum(size for size, _ in files.values()),
                "blend_files": sorted(n for n in files if n.lower().endswith(".blend"))}


def unpack_submission(archive, task_id, destination):
    """Extract submitted files into an evaluator-owned copy, preserving paths."""
    archive, destination = Path(archive), Path(destination)
    details = inspect_submission(archive, task_id)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    try:
        with _payload(archive, task_id) as (files, read, expected):
            for name, (size, _) in sorted(files.items()):
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with read(name) as source, path.open("xb") as target:
                    written = _copy(source, target, size)
                if written != size:
                    raise ValueError("Truncated delivered file")
                hashes[name] = sha256(path)
            if expected is not None and hashes != expected:
                raise ValueError("Submission checksum mismatch")
        return {**details, "archive_sha256": sha256(archive), "file_sha256": hashes}
    except BaseException:
        shutil.rmtree(destination)
        raise
