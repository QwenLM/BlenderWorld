"""Load and verify the portable dataset without a model service."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

RESOURCE_FIELDS = ("attachments", "supplemental_attachments", "reference_images",
                   "reference_videos", "rubric_reference_images")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def load_tasks(root):
    rows = [json.loads(s) for s in (Path(root) / "dataset.jsonl").read_text().splitlines() if s.strip()]
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids) or not all(re.fullmatch(r"bw_\d{4}", s) for s in ids):
        raise ValueError("Task IDs must be unique neutral identifiers")
    return rows


def verify(root):
    root = Path(root)
    rows = load_tasks(root)
    modalities = Counter(r["modality"] for r in rows)
    if modalities != {"text": 78, "image": 130, "video": 15}:
        raise ValueError(f"Unexpected benchmark task counts: {modalities}")
    bundled = json.loads((root / "rubrics.json").read_text())["results"]
    combined = {r["task_id"]: r for r in bundled}
    if len(combined) != 223 or len(bundled) != 223:
        raise ValueError("Combined rubric index must have 223 unique tasks")
    criteria, resources = 0, 0
    for row in rows:
        task_root = root / "tasks" / row["task_id"]
        if json.loads((task_root / "task.json").read_text()) != row:
            raise ValueError(f"Task/index mismatch: {row['task_id']}")
        if resolve(root, row["query_file"]).read_text() != row["query"]:
            raise ValueError(f"Query mismatch: {row['task_id']}")
        rubric = json.loads(resolve(root, row["rubric_file"]).read_text())
        if rubric != combined[row["task_id"]] or rubric["prompt"] != row["query"]:
            raise ValueError(f"Rubric/query/index mismatch: {row['task_id']}")
        ids = [r["id"] for r in rubric["rubrics"]]
        if len(set(ids)) != len(ids) or len(ids) != row["rubric_count"]:
            raise ValueError(f"Invalid criterion IDs: {row['task_id']}")
        for criterion in rubric["rubrics"]:
            if criterion["score_values"] != [0, 1] or criterion["importance_weight"] not in (1, 2, 3):
                raise ValueError("Invalid binary criterion/importance weight")
            if criterion["evidence_type"] not in {"scene_summary", "render", "either"}:
                raise ValueError("Unknown evidence route")
        for attachment in rubric["attachments"]:
            if not resolve(task_root, attachment["file"]).is_file():
                raise ValueError(f"Missing rubric attachment: {attachment['file']}")
        if rubric.get("image_file") and not resolve(task_root, rubric["image_file"]).is_file():
            raise ValueError(f"Missing rubric image: {rubric['image_file']}")
        for field in RESOURCE_FIELDS:
            for entry in row[field]:
                path = resolve(root, entry["path"])
                if path != resolve(task_root, entry["task_relative_path"]):
                    raise ValueError("Resource path aliases disagree")
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Resource checksum mismatch: {entry['path']}")
                resources += 1
        criteria += len(ids)
    if criteria != 2651:
        raise ValueError(f"Expected 2651 criteria, got {criteria}")
    checksums = root / "SHA256SUMS"
    if not checksums.is_file():
        raise ValueError("Missing release SHA256SUMS")
    checksum_paths = set()
    for line in checksums.read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        if relative in checksum_paths or sha256(resolve(root, relative)) != checksum:
            raise ValueError(f"Duplicate or incorrect checksum: {relative}")
        checksum_paths.add(relative)
    actual_paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                    and not any(part in {".git", ".cache"} for part in p.relative_to(root).parts)}
    if actual_paths != checksum_paths | {"SHA256SUMS"}:
        raise ValueError("Release checksum inventory does not cover the complete file tree")
    manifest = json.loads((root / "manifest.json").read_text())
    if {r["path"] for r in manifest["files"]} != actual_paths - {"manifest.json", "SHA256SUMS"}:
        raise ValueError("Manifest does not cover the complete release payload")
    return {"tasks": len(rows), "modalities": dict(modalities), "criteria": criteria,
            "resources": resources, "checksums_verified": checksums.exists()}


def prepare(root, task_id, output):
    """Stage only generator-visible inputs; evaluation rubrics remain in the dataset."""
    root, output = Path(root), Path(output)
    row = next((r for r in load_tasks(root) if r["task_id"] == task_id), None)
    if row is None:
        raise ValueError(f"Unknown task: {task_id}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "query.txt").write_text(row["query"])
    inputs = []
    for field in ("attachments", "reference_images", "reference_videos"):
        for entry in row[field]:
            target = resolve(output, entry["task_relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolve(root, entry["path"]), target)
            inputs.append(entry["task_relative_path"])
    write_json(output / "input.json", {"task_id": task_id, "query_file": "query.txt", "files": inputs})
    return {"task_id": task_id, "output": str(output), "input_files": len(inputs)}
