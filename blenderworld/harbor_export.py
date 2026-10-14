"""Export the frozen dataset as self-contained Harbor task packages."""
from __future__ import annotations

import json
import importlib.metadata
import re
import shutil
from pathlib import Path

from .data import RESOURCE_FIELDS, load_tasks, prepare, resolve, sha256, verify, write_json
from .evaluation import PROTOCOL
from . import __version__
from .runtime import RUNTIME_VERSIONS, requirements
from .submissions import FORMAT, PROJECT

HARBOR_VERSION = "0.23.0"
ASSETS = Path(__file__).with_name("harbor_assets")


def _harbor():
    try:
        from harbor.models.dataset.manifest import DatasetManifest
        from harbor.models.task.task import Task
        from harbor.publisher.packager import Packager
    except ImportError as error:
        raise RuntimeError("Harbor export requires Python 3.12+ and pip install 'blenderworld[harbor]'") from error
    if importlib.metadata.version("harbor") != HARBOR_VERSION:
        raise RuntimeError(f"This exporter is validated with harbor=={HARBOR_VERSION}; install the pinned harbor extra")
    return DatasetManifest, Task, Packager


def _instruction(row):
    resources = []
    for field in ("reference_images", "reference_videos", "attachments"):
        for entry in row[field]:
            resources.append(f"- `{field}`: `/workspace/inputs/{entry['task_relative_path']}`")
    return (f"# BlenderWorld {row['task_id']}\n\n"
            "## Task\n\n" + row["query"] + "\n\n"
            "## Inputs\n\nThe original request is also in `/workspace/inputs/query.txt`.\n\n"
            + ("\n".join(resources) if resources else "This task has no reference media or declared attachments.")
            + "\n\nRead the supplied specifications and inspect the reference media before modeling. "
            "Relative filenames in the request refer to the files listed above.\n\n"
            "## Deliverables\n\nSave the editable project as `/workspace/exports/final.blend`. "
            "Put all requested images, videos, model exports, and texture dependencies under "
            "`/workspace/exports/`. Pack textures into the project or keep relative paths inside "
            "this directory so the project can be opened in a fresh environment.\n\n"
            "## Tools\n\nBlender is installed as `blender`. A Blender MCP server is configured "
            "for agents that support task MCP servers; use its advertised tools to inspect the "
            "scene, execute Blender Python, and capture viewport images. The Blender Python CLI "
            "is also available. Use Blender 4.2.10, Node.js 22.17.0, Codex CLI 0.147.0 "
            "or Claude Code 2.1.220; do not upgrade these tools. "
            "Inspect your result and save your work before finishing.\n")


def _config(row, namespace, cpus, memory_mb, gpus, agent_timeout, verifier_timeout):
    quote = json.dumps
    gpu_line = f"gpus = {gpus}\n" if gpus else ""
    return f'''schema_version = "1.3"
artifacts = ["/workspace/exports"]

[task]
name = {quote(namespace + '/BlenderWorld-' + row['task_id'])}
description = "BlenderWorld {row['task_id']} ({row['modality']})"
keywords = ["blender", "3d", "{row['modality']}"]

[metadata]
task_id = "{row['task_id']}"
modality = "{row['modality']}"
rubric_count = {row['rubric_count']}
protocol_version = {PROTOCOL['version']}
code_version = "{__version__}"
blender_version = "{RUNTIME_VERSIONS['blender']}"
node_version = "{RUNTIME_VERSIONS['node']}"
codex_version = "{RUNTIME_VERSIONS['codex']}"
claude_code_version = "{RUNTIME_VERSIONS['claude_code']}"
submission_format = "{FORMAT}"
submission_project = "{PROJECT}"

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = 3600
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 102400
{gpu_line}workdir = "/workspace"

[[environment.mcp_servers]]
name = "blender"
transport = "stdio"
command = "/usr/local/bin/bw-blender-mcp"
args = []

[verifier]
timeout_sec = {verifier_timeout}
environment_mode = "separate"

[verifier.environment]
build_timeout_sec = 3600
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 102400
{gpu_line}workdir = "/workspace"

[verifier.env]
ANTHROPIC_BASE_URL = "${{BW_AGENT_JUDGE_ENDPOINT}}"
ANTHROPIC_API_KEY = "${{BW_AGENT_JUDGE_API_KEY}}"
JUDGE_API_ENDPOINT = "${{BW_VISION_JUDGE_ENDPOINT}}"
JUDGE_API_KEY = "${{BW_VISION_JUDGE_API_KEY}}"
RENDER_AGENT_MODEL = "${{BW_RENDER_MODEL:-gpt-5.6-terra}}"
AGENT_JUDGE_MODEL = "${{BW_AGENT_JUDGE_MODEL:-gpt-5.6-terra}}"
JUDGE_MODEL = "${{BW_VISION_JUDGE_MODEL:-gpt-5.6-terra}}"
'''


def _copy_dataset_task(dataset, row, destination):
    task_root = destination / "tasks" / row["task_id"]
    task_root.mkdir(parents=True)
    write_json(task_root / "task.json", row)
    for key in ("query_file", "rubric_file"):
        target = resolve(destination, row[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(dataset, row[key]), target)
    for field in RESOURCE_FIELDS:
        for entry in row[field]:
            target = resolve(destination, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolve(dataset, entry["path"]), target)
    (destination / "dataset.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    rubric = json.loads(resolve(dataset, row["rubric_file"]).read_text())
    write_json(destination / "rubrics.json", {"results": [rubric]})


def export_harbor(dataset, output, namespace="Qwen", task_ids=None, cpus=8,
                  memory_mb=16384, gpus=0, agent_timeout=14400, verifier_timeout=28800):
    """Create a new export; the source dataset and existing task packages are never edited."""
    Manifest, Task, Packager = _harbor()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", namespace) or ".." in namespace:
        raise ValueError("Invalid Harbor organization namespace")
    if min(cpus, memory_mb, agent_timeout, verifier_timeout) <= 0 or gpus < 0:
        raise ValueError("Resources and timeouts must be positive; GPU count may be zero")
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Export directory must be outside the frozen dataset")
    verify(dataset)
    rows = load_tasks(dataset)
    if task_ids:
        selected = set(task_ids)
        rows = [r for r in rows if r["task_id"] in selected]
        if {r["task_id"] for r in rows} != selected:
            raise ValueError("Unknown task ID in selection")
    output.mkdir(parents=True, exist_ok=False)
    manifest_tasks, audit_tasks = [], []
    code_root = Path(__file__).parent
    # Only Python runtime files go into the verifier, never into the agent image.
    code_files = sorted(code_root.rglob("*.py"))
    for row in rows:
        task = output / row["task_id"]
        env, tests = task / "environment", task / "tests"
        env.mkdir(parents=True)
        tests.mkdir()
        (task / "instruction.md").write_text(_instruction(row))
        (task / "task.toml").write_text(_config(row, namespace, cpus, memory_mb, gpus,
                                                agent_timeout, verifier_timeout))
        prepare(dataset, row["task_id"], env / "inputs")
        for name in ("Dockerfile.agent", "mcp_bridge.py", "blender_startup.py", "install_blender.sh", "requirements-tools.txt", "LICENSE"):
            shutil.copyfile(ASSETS / name, env / ("Dockerfile" if name == "Dockerfile.agent" else name))
        for name in ("Dockerfile.verifier", "test.sh", "install_blender.sh", "requirements-verifier.txt", "LICENSE"):
            shutil.copyfile(ASSETS / name, tests / ("Dockerfile" if name == "Dockerfile.verifier" else name))
        (tests / "test.sh").chmod(0o755)
        for path in code_files:
            relative = path.relative_to(code_root)
            if "harbor_assets" in relative.parts:
                continue
            target = tests / "blenderworld" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        _copy_dataset_task(dataset, row, tests / "dataset")
        Task(task)  # Validate with the same schema used by the installed Harbor release.
        digest, files = Packager.compute_content_hash(task)
        manifest_tasks.append({"name": f"{namespace}/BlenderWorld-{row['task_id']}",
                               "digest": "sha256:" + digest})
        audit_tasks.append({"task_id": row["task_id"], "modality": row["modality"],
                            "digest": "sha256:" + digest, "files": len(files),
                            "query_sha256": sha256(env / "inputs/query.txt"),
                            "rubric_sha256": sha256(resolve(tests / "dataset", row["rubric_file"]))})
    manifest = Manifest.model_validate({"dataset": {"name": f"{namespace}/BlenderWorld",
                                       "version": "1.2.0", "description": "BlenderWorld: 3D creation in Blender",
                                       "keywords": ["blender", "3d", "multimodal", "agent"]},
                                       "tasks": manifest_tasks})
    (output / "dataset.toml").write_text(manifest.to_toml())
    for name in ("README.md", "verifier.env.example", "LICENSE"):
        shutil.copyfile(ASSETS / name, output / name)
    report = {"format": "blenderworld-harbor-v1", "harbor_version": HARBOR_VERSION,
              "code_version": __version__, "runtime_requirements": requirements(),
              "submission_format": FORMAT,
              "dataset_name": manifest.dataset.name, "task_count": len(rows),
              "dataset_digest": "sha256:" + manifest.compute_content_hash(),
              "source_dataset_sha256": sha256(dataset / "dataset.jsonl"),
              "protocol": PROTOCOL, "tasks": audit_tasks}
    write_json(output / "export_manifest.json", report)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}\n" for p in files))
    return {k: v for k, v in report.items() if k != "tasks"}
