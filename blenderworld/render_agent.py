"""Adaptive eight-view evidence rendering using a configured coding agent."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .data import write_json
from .evaluators.evidence_checks import cli_log_status

VIEW_COUNT = 8
RESOLUTION = 1024
SAMPLES = 64
MAX_CORRECTIVE_RERENDERS = 2


def normalize_blender_path(value):
    """Resolve caller-relative paths before agents change their working directory.

    A bare command may stay unresolved during a dry run without Blender installed.
    """
    requested = os.path.expanduser(str(value))
    found = shutil.which(requested)
    if found:
        return str(Path(found).resolve())
    if os.sep in requested:
        return str(Path(requested).resolve())
    return requested


def prompt_for(task, blender):
    return f"""You collect rendering evidence for BlenderWorld.
Open the primary submitted project .eval_workspace/final.blend with {blender}. Read query.txt,
rubric.json, and references/. These describe the target; they do not prove that the
submission satisfies it. Render exactly {VIEW_COUNT} complementary views into
evaluation/multiview/. Start with the submitted camera when it shows the subject,
then choose different wide views and at least two detail views relevant to the rubrics.
Use short unique filenames ending in .png. Fit cameras to the subject, ignoring
large background/floor planes. For interiors, position cameras inside enclosing walls.

Every view must use CYCLES, {RESOLUTION}x{RESOLUTION}, {SAMPLES} samples. Prefer OPTIX,
then CUDA; use CPU if no supported GPU exists. Record the actual device. Retain the
scene's denoising. Preserve geometry, materials, textures, object visibility, lighting,
animation, and color management, including exposure and gamma. Never repair the
submitted scene. Do not save over a .blend or any submitted file. The provided Blender
wrapper restores missing image paths only from this task's bundled workspace.

After each render, open and inspect the PNG. Correct evaluator camera/framing/clipping
or incorrect evaluator render settings when necessary; allow at most
{MAX_CORRECTIVE_RERENDERS} corrective re-renders per view (one initial render plus
two corrections, three total). Reuse the same filename. Dark or oddly colored
authored scenes must remain that way. Do not alter lights or materials to improve them.
Keep all Blender calls in the foreground and wait for completion. The runtime logs
render settings and rejects more than three attempts for the same view.

Write evaluation/multiview/render_agent_report.json with:
{{"ok": true, "view_count": 8, "engine": "CYCLES", "device": "OPTIX|CUDA|CPU",
"image_files": ["view_01.png", "..."], "notes": "view selection and corrections"}}.
All PNGs must be newly rendered from this submission. Do not copy author exports or
reference images into the evaluator views. Do not write outside the task directory.

Task query:
{task['query']}
"""


def validate_views(task_dir):
    root = Path(task_dir)
    folder = root / "evaluation/multiview"
    images = sorted(folder.glob("*.png"))
    if len(images) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} fresh evaluator renders, found {len(images)}")
    report = json.loads((folder / "render_agent_report.json").read_text())
    if report.get("ok") is not True or report.get("engine") != "CYCLES" or report.get("view_count") != VIEW_COUNT:
        raise ValueError("Invalid render-agent report")
    if sorted(report.get("image_files", [])) != [p.name for p in images]:
        raise ValueError("Render report does not list the actual views")
    attempts = json.loads((folder / "render_attempts.json").read_text())
    for path in images:
        with Image.open(path) as image:
            if image.size != (RESOLUTION, RESOLUTION):
                raise ValueError(f"Incorrect resolution: {path.name}")
            image.verify()
        history = attempts.get(str(path.resolve()), [])
        if not history or len(history) > 1 + MAX_CORRECTIVE_RERENDERS:
            raise ValueError(f"Missing or excessive render attempts: {path.name}")
        final = history[-1]
        if final.get("engine") != "CYCLES" or final.get("samples") != SAMPLES:
            raise ValueError(f"Incorrect render settings: {path.name}")
        if final.get("completed") is not True:
            raise ValueError(f"No successful Blender render: {path.name}")
    return report


def run(task, task_dir, blender_bin, model, timeout=5400):
    root = Path(task_dir).resolve()
    folder = root / "evaluation/multiview"
    folder.mkdir(parents=True, exist_ok=False)
    real_blender = shutil.which(normalize_blender_path(blender_bin))
    if not real_blender:
        raise FileNotFoundError(f"Blender not found: {blender_bin}")
    bootstrap = Path(__file__).with_name("blender_runtime.py").resolve()
    wrapper = root / "evaluation/blender"
    wrapper.write_text("#!/usr/bin/env python3\nimport os,sys\nos.execv(" + repr(real_blender)
                       + ", [" + repr(real_blender) + ", '--background', '--python-exit-code', '2', '--python', "
                       + repr(str(bootstrap)) + "] + sys.argv[1:])\n")
    wrapper.chmod(0o755)
    prompt = prompt_for(task, str(wrapper))
    (root / "evaluation/render_prompt.txt").write_text(prompt)
    env = os.environ.copy()
    env.update(BLENDER_EVAL_ASSET_ROOT=str(root), BLENDER_EVAL_VIEW_ROOT=str(folder),
               BLENDER_EVAL_RUNTIME_AUDIT=str(root / "evaluation/blender_runtime.jsonl"),
               DISABLE_AUTOUPDATER="1")
    if model:
        env["ANTHROPIC_MODEL"] = model
    log_path = root / "evaluation/render_agent.jsonl"
    with log_path.open("w") as log:
        result = subprocess.run(["claude", "--verbose", "--output-format", "stream-json",
                                 "--max-turns", "500", "--allowedTools", "Bash", "Read", "Write",
                                 "Edit", "Glob", "Grep", "--disallowedTools", "WebFetch", "WebSearch",
                                 "-p", prompt], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    event, refused = cli_log_status(log_path)
    if result.returncode or refused or not event or event.get("is_error"):
        raise RuntimeError("Render agent failed; see evaluation/render_agent.jsonl")
    return validate_views(root)
