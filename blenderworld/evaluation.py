"""Local evidence preparation, rendering, judging, and benchmark aggregation."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import render_agent
from .data import RESOURCE_FIELDS, load_tasks, resolve, sha256, write_json
from .evaluators import judge_blender_binary_rubrics as vision
from .evaluators.evidence_checks import cli_log_status
from .runtime import RuntimeValidationError, inspect_runtime, record_runtime, requirements
from .submissions import FORMAT, inspect_submission, unpack_submission

PROTOCOL = {"version": 1, "views": 8, "resolution": 1024, "samples": 64,
            "engine": "CYCLES", "max_corrective_rerenders": 2,
            "dimension_weight": 0.5, "score_cap": None,
            "structural_evidence": ["scene_summary"], "visual_evidence": ["render", "either"]}


def tree_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def sample_video(source, destination, count):
    destination.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", str(source)], check=True, capture_output=True, text=True, timeout=60)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Video has no positive duration: {source.name}")
    # Sample across the whole clip, including both ends without seeking past its last frame.
    for i in range(count):
        timestamp = max(0, duration - min(0.1, duration / count)) * i / max(1, count - 1)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(timestamp), "-i", str(source),
                        "-frames:v", "1", str(destination / f"frame_{i:03d}.png")],
                       check=True, capture_output=True, timeout=120)
    if len(list(destination.glob("*.png"))) != count:
        raise ValueError(f"Incomplete video sampling: {source.name}")


def prepare_evaluation(dataset, row, submission, task_dir):
    task_dir.mkdir(parents=True, exist_ok=False)
    snapshot = task_dir / ".eval_workspace"
    # The public ZIP contract is independent of the model's generation harness.
    manifest = unpack_submission(submission, row["task_id"], snapshot)
    write_json(task_dir / "submission_archive.json", manifest)
    (task_dir / "query.txt").write_text(row["query"])
    rubric = json.loads(resolve(dataset, row["rubric_file"]).read_text())
    write_json(task_dir / "rubric.json", rubric)
    input_hashes = {entry["sha256"] for field in RESOURCE_FIELDS for entry in row[field]}
    write_json(task_dir / "input_media_sha256.json", sorted(input_hashes))
    for field in RESOURCE_FIELDS:
        for entry in row[field]:
            path = resolve(task_dir, entry["task_relative_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolve(dataset, entry["path"]), path)
    judge_refs = task_dir / "image_assets"
    judge_refs.mkdir()
    image_entries = row["rubric_reference_images"] or row["reference_images"]
    for entry in image_entries:
        shutil.copyfile(resolve(dataset, entry["path"]), judge_refs / Path(entry["path"]).name)
    for i, entry in enumerate(row["reference_videos"]):
        sample_video(resolve(dataset, entry["path"]), judge_refs / f"video_{i:02d}", 4)
    for i, video in enumerate(sorted(snapshot.rglob("*"))):
        if (video.is_file() and video.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
                and sha256(video) not in input_hashes):
            sample_video(video, task_dir / "animation_frames" / f"clip_{i:03d}", 16)
    write_json(task_dir / "submission_sha256.json", tree_hashes(snapshot))
    write_json(task_dir / "rubrics_batch.json", {"results": [{**rubric, "task_dir": str(task_dir.resolve())}]})


def run_module(module, args, log):
    with log.open("w") as stream:
        completed = subprocess.run([sys.executable, "-m", "blenderworld.evaluators." + module, *args],
                                   env={**os.environ, "DISABLE_AUTOUPDATER": "1"},
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{module} failed ({completed.returncode}); see {log.name}")


def validate_score(row, rubric, result):
    if result.get("ok") is not True or result.get("dry_run"):
        raise ValueError("Judge did not produce a successful scored result")
    if result.get("cap_applied") or not result.get("rendered_image_files"):
        raise ValueError("Result does not follow the public scoring protocol")
    scores = result.get("rubric_scores", [])
    expected = {r["id"] for r in rubric["rubrics"]}
    if {r["id"] for r in scores} != expected or len(scores) != len(expected):
        raise ValueError("Incomplete or duplicate rubric scores")
    if any(type(r.get("score")) is not int or r["score"] not in (0, 1) for r in scores):
        raise ValueError("Rubric scores must be binary integers")
    dimensions = result.get("dimension_scores", [])
    expected_dims = {r["id"] for r in vision.DIMENSIONS}
    if {r["id"] for r in dimensions} != expected_dims or len(dimensions) != len(expected_dims):
        raise ValueError("Incomplete or duplicate dimension scores")
    if any(type(r.get("score")) is not int or not 1 <= r["score"] <= 10 for r in dimensions):
        raise ValueError("Dimension scores must be integers from 1 to 10")
    rubric_score = vision.build_score_summary(rubric["rubrics"], scores)["weighted_score_0_100"]
    dimension_score = 10 * sum(d["score"] for d in dimensions) / len(dimensions)
    fused = 0.5 * (rubric_score + dimension_score)
    if abs(fused - result["fused_score_0_100"]) > 1e-6:
        raise ValueError("Fused score differs from the paper's uncapped 1:1 formula")
    return {"task_id": row["task_id"], "modality": row["modality"], "ok": True,
            "score": fused, "rubric_score": rubric_score, "dimension_score": dimension_score,
            "error": ""}


def score_task(row, task_dir, options):
    common = ["--rubrics-batch", str(task_dir / "rubrics_batch.json"),
              "--image-glob", "evaluation/multiview/*.png", "--max-images", "8",
              "--reference-image-glob", "image_assets/**/*", "--max-reference-images", "4",
              "--extra-image-glob", ".eval_workspace/**/*.png,.eval_workspace/**/*.jpg,.eval_workspace/**/*.jpeg,.eval_workspace/**/*.webp"]
    rubric = json.loads((task_dir / "rubric.json").read_text())
    if any(r["evidence_type"] == "scene_summary" for r in rubric["rubrics"]):
        run_module("judge_blender_agent", common + ["--agent-evidence-types", "scene_summary",
                   "--no-dimension-pass", "--blender-bin", options.blender_bin,
                   "--model", options.agent_judge_model, "--task-concurrency", "1",
                   "--batch-output", str(task_dir / "agent_batch.json")], task_dir / "agent_judge.log")
        folder = task_dir / "evaluation/agent_judge_multiview_eval"
        agent_result = json.loads((folder / "judgement.json").read_text())
        if agent_result.get("agent_returncode") != 0:
            raise ValueError("Structural judge did not exit successfully")
        expected_logs = [folder / "agent_judge_stdout.jsonl"] + [
            folder / f"agent_judge_stdout.attempt{i}.jsonl"
            for i in range(2, agent_result.get("agent_attempts", 1) + 1)]
        if any(not p.is_file() for p in expected_logs):
            raise ValueError("Structural judge attempt logs are incomplete")
        for log in folder.glob("agent_judge_stdout*.jsonl"):
            terminal, refused = cli_log_status(log)
            if refused:
                raise ValueError("Structural judge encountered an upstream content refusal")
        terminal, refused = cli_log_status(resolve(task_dir, agent_result["agent_stdout_path"]))
        if not terminal or terminal.get("is_error"):
            raise ValueError("Structural judge did not terminate successfully")
        for key in ("agent_raw_path", "bpy_dump_path"):
            if not json.loads(resolve(task_dir, agent_result[key]).read_text()):
                raise ValueError("Structural judge has no inspectable raw evidence")
    run_module("judge_blender_binary_rubrics", common + ["--evidence-routing",
               "--model", options.judge_model, "--api-endpoint", options.judge_api_endpoint,
               "--anim-frame-glob", "animation_frames/**/*.png", "--max-anim-frames", "16",
               "--task-concurrency", "1", "--batch-output", str(task_dir / "vision_batch.json")],
               task_dir / "vision_judge.log")
    result = json.loads((task_dir / "evaluation/judge_vlm/judgement.json").read_text())
    if tree_hashes(task_dir / ".eval_workspace") != json.loads((task_dir / "submission_sha256.json").read_text()):
        raise ValueError("Evaluator modified the submitted workspace")
    return validate_score(row, rubric, result)


def summarize(rows, records):
    def aggregate(selected):
        count = len(selected)
        known = [records[r["task_id"]] for r in selected if r["task_id"] in records]
        return {"task_count": count, "completed": len(known), "succeeded": sum(r["ok"] for r in known),
                "mean_score": sum(r["score"] for r in known) / count if count else None}
    return {"complete": len(records) == len(rows), "overall": aggregate(rows),
            "text": aggregate([r for r in rows if r["modality"] == "text"]),
            "multimodal": aggregate([r for r in rows if r["modality"] != "text"]),
            "results": [records[r["task_id"]] for r in rows if r["task_id"] in records]}


def evaluate(options):
    dataset, submissions, output = map(lambda p: Path(p).resolve(), (options.dataset, options.submissions, options.output))
    options.blender_bin = render_agent.normalize_blender_path(getattr(options, "blender_bin", "blender"))
    rows = load_tasks(dataset)
    single_zip = submissions.suffix.lower() == ".zip" and not submissions.is_dir()
    if single_zip and not options.task_id:
        options.task_id = [submissions.stem]
    if options.task_id:
        requested = set(options.task_id)
        rows = [r for r in rows if r["task_id"] in requested]
        if {r["task_id"] for r in rows} != requested:
            raise ValueError("Unknown task ID in selection")
    if single_zip and len(rows) != 1:
        raise ValueError("A single submission ZIP must select exactly one task")
    if output == submissions or (submissions.is_dir() and output.is_relative_to(submissions)):
        raise ValueError("Evaluation output must be outside the submissions directory")
    archives = {r["task_id"]: submissions if single_zip else submissions / (r["task_id"] + ".zip") for r in rows}
    output.mkdir(parents=True, exist_ok=True)
    # A run has one owner. A live owner must never be mistaken for a stopped run.
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run is already active") from error
        plan = {"dataset_sha256": sha256(dataset / "dataset.jsonl"),
                "rubrics_sha256": sha256(dataset / "rubrics.json"),
                "code_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p)
                                for p in sorted(Path(__file__).parent.rglob("*.py"))},
                "task_ids": [r["task_id"] for r in rows], "protocol": PROTOCOL,
                "models": {"render": options.render_agent_model, "agent": options.agent_judge_model,
                           "vision": options.judge_model}, "submissions": str(submissions),
                "endpoints": {"vision": getattr(options, "judge_api_endpoint", ""),
                              "agent": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")},
                "blender_bin": getattr(options, "blender_bin", "blender"),
                "submission_format": FORMAT, "runtime_requirements": requirements(),
                "submission_hashes": {tid: sha256(path) if path.is_file() and not path.is_symlink() else None
                                      for tid, path in archives.items()}}
        plan_file = output / "plan.json"
        if plan_file.exists():
            if not options.resume:
                raise ValueError("Run exists; use --resume or choose a new output directory")
            if json.loads(plan_file.read_text()) != plan:
                raise ValueError("Inputs, models, code, or protocol changed; start a new run")
        else:
            write_json(plan_file, plan)
        if options.dry_run:
            inventory, errors = {}, {}
            for task_id, path in archives.items():
                try:
                    inventory[task_id] = inspect_submission(path, task_id)
                except (OSError, ValueError) as error:
                    errors[task_id] = str(error)
            return {"dry_run": True, "tasks": len(rows), "plan": str(plan_file),
                    "missing_submissions": [tid for tid, path in archives.items() if not path.is_file()],
                    "submission_inventory": inventory, "submission_errors": errors}
        runtime_checked = False
        records = {}
        for row in rows:
            task_id = row["task_id"]
            task_dir = output / "tasks" / task_id
            outcome = task_dir / "result.json"
            if outcome.exists():
                records[task_id] = json.loads(outcome.read_text())
                continue
            if task_dir.exists():
                # Preserve incomplete evidence; resume in a new attempt directory.
                n = 1
                while (output / "incomplete" / f"{task_id}.{n}").exists():
                    n += 1
                saved = output / "incomplete" / f"{task_id}.{n}"
                saved.parent.mkdir(parents=True, exist_ok=True)
                task_dir.rename(saved)
            print(f"Evaluating {task_id}", flush=True)
            try:
                submission = archives[task_id]
                if not submission.is_file():
                    raise ValueError("Missing submission ZIP: " + task_id + ".zip")
                if sha256(submission) != plan["submission_hashes"][task_id]:
                    raise ValueError("Submission ZIP changed after planning")
                prepare_evaluation(dataset, row, submission, task_dir)
                if not runtime_checked:
                    record_runtime(output, inspect_runtime(options.blender_bin))
                    runtime_checked = True
                render_agent.run(row, task_dir, options.blender_bin, options.render_agent_model)
                result = score_task(row, task_dir, options)
                if sha256(submission) != plan["submission_hashes"][task_id]:
                    raise ValueError("Original submission ZIP changed during evaluation")
            except RuntimeValidationError:
                # A wrong evaluator runtime must not become 223 failed agent scores.
                raise
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result = {"task_id": task_id, "modality": row["modality"], "ok": False,
                          "score": 0.0, "rubric_score": None, "dimension_score": None, "error": str(error)}
            write_json(outcome, result)
            records[task_id] = result
            write_json(output / "summary.json", summarize(rows, records))
        summary = summarize(rows, records)
        write_json(output / "summary.json", summary)
        with (output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task_id", "modality", "ok", "score", "rubric_score", "dimension_score", "error"])
            writer.writeheader()
            writer.writerows(summary["results"])
        return {k: v for k, v in summary.items() if k != "results"}
