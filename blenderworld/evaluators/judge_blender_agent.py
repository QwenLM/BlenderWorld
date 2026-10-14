#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-as-Judge for Blender trajectory eval (runs in parallel with the GPT-5.6-terra vision judge).

Unlike ``judge_blender_binary_rubrics.py`` (a pure-vision pass that can only score what a
static render shows), this judge spawns a coding agent per task -- by default the same
GPT-5.6-terra model the render agent already uses, via the ``claude`` CLI pointed at the
configured Anthropic-compatible endpoint -- that can OPEN THE SUBMITTED ``.blend`` WITH ``bpy``.
This lets it verify rubrics whose ``evidence_type`` is ``scene_summary`` / ``either``
(shape keys, drivers, Geometry-Nodes params, triangle budgets, UV layout, material node
graphs, ``scene.unit_settings`` / ``render.engine`` / ``cycles.samples``, exact mm
dimensions, physics setup) that the vision judge structurally cannot check, while still
using the rendered PNGs for ``render``-type (visual) rubrics.

The agent emits, per rubric, two cross-check sub-verdicts (``bpy_check`` for the .blend internals
and ``visual_check`` for its own renders, each pass/fail/na) + method + evidence, and per dimension a
1-10 impression plus a ``render_conflict`` flag. ALL scoring arithmetic (weighted rubric pass-rate,
dimension score, the ``0.5*dim + 0.5*rubric`` fusion, and the uncapped fusion) is done here
in Python, reusing the exact helpers from ``judge_blender_binary_rubrics.py``. Two worst-of-both
rules make the visible render authoritative: a rubric passes only if EVERY applicable sub-verdict
passes (so a correct bpy value cannot rescue a wrong-looking render), and any dimension the agent
flags with ``render_conflict`` is clamped into the low band. Both are backward-compatible (agent
outputs lacking the new fields score exactly as before) and can be disabled via CLI. The top-level
per-task ``judgement.json`` and batch schema stay IDENTICAL to the vision judge (the new fields are
additive inside each ``rubric_scores``/``dimension_scores`` entry; the final per-rubric ``score`` is
still integer 0/1), so the Stage-7 aggregate in ``main.sh`` and downstream comparisons are unchanged.

Audit trail: the agent is required to persist ``agent_bpy_dump.json`` (the bpy query script
+ its raw output) and ``agent_judgement_raw.json`` next to the judgement, so every
scene-metadata verdict is checkable after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the vision judge's pure, model-agnostic helpers so the output schema and scoring
# math stay bit-for-bit consistent between the two judges.
from . import judge_blender_binary_rubrics as vj
from . import evidence_checks

try:
    from json_repair import repair_json
except Exception:  # noqa: BLE001
    def repair_json(s: str) -> str:
        return s


DEFAULT_OUTPUT_DIR = "evaluation/agent_judge_multiview_eval"
DEFAULT_IMAGE_GLOB = "evaluation/multiview/*.png"
RAW_NAME = "agent_judgement_raw.json"
BPY_DUMP_NAME = "agent_bpy_dump.json"


# --------------------------------------------------------------------------------------
# Agent output parsing
# --------------------------------------------------------------------------------------
def cli_api_error(path: Path) -> str:
    """Read the CLI's terminal API failure before attempting JSON output repair."""
    if not path.exists():
        return ""
    event, refused = evidence_checks.cli_log_status(path)
    if refused:
        return "agent_judge_upstream_content_policy_violation"
    if event:
        status = event.get("api_error_status")
        if event.get("is_error") is True and status:
            return f"agent judge API error {status}"
    return ""


def _coerce_binary(value: Any) -> int | None:
    if value in (0, "0", "0.0", False):
        return 0
    if value in (1, "1", "1.0", True):
        return 1
    return None


def _coerce_check(value: Any) -> str:
    """Normalize a per-rubric sub-verdict to ``pass`` / ``fail`` / ``na``.

    Missing or unrecognized values collapse to ``na`` (not applicable), which makes the
    cross-check enforcement a no-op for that source and lets old outputs (with no ``bpy_check`` /
    ``visual_check`` keys) fall back to the model's self-reported ``score`` unchanged."""
    if value is None:
        return "na"
    s = str(value).strip().lower()
    if s in ("pass", "passed", "ok", "yes", "true", "1", "1.0"):
        return "pass"
    if s in ("fail", "failed", "no", "false", "0", "0.0"):
        return "fail"
    return "na"  # na / n/a / not_applicable / unknown / "" / anything else


def _truthy(value: Any) -> bool:
    """Loose truthiness for a model-emitted boolean flag (accepts true/1/yes/'true'...)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("true", "1", "1.0", "yes", "y")


def normalize_agent_rubrics(
    obj: dict[str, Any] | None,
    rubrics: list[dict[str, Any]],
    *,
    enforce_cross_check: bool = True,
) -> tuple[bool, list[dict[str, Any]], str]:
    """Map the agent's per-rubric verdicts onto the canonical rubric_scores shape.

    Mirrors ``vj.normalize_scores`` (id/score/evidence/source_views/confidence) and adds a
    ``method`` field (bpy|visual). Returns (ok, scores, error); ok is False if any rubric id
    is unanswered (build_score_summary still counts those as 0, matching the vision judge).

    Cross-check enforcement (``enforce_cross_check``, default True): each verdict may carry two
    sub-verdicts, ``bpy_check`` (does the requirement hold in the .blend internals) and
    ``visual_check`` (does it hold in the agent's own renders), each ``pass``/``fail``/``na``. The
    final ``score`` is derived as 1 iff EVERY applicable (non-``na``) sub-verdict passes -- so a
    correct bpy value can never rescue a rubric whose render contradicts it (and vice-versa). When
    neither sub-verdict applies (both ``na``, or the keys are absent), we fall back to the model's
    self-reported ``score`` -- keeping old agent outputs bit-for-bit identical. The final ``score``
    stays an integer 0/1 (relied on by ``vj.build_score_summary`` / ``vj.is_complete_judgement``);
    the sub-verdicts + ``model_score`` + ``cross_check_applied`` are stored as an audit trail."""
    if not obj:
        return False, [], "agent output was not valid JSON"
    raw_scores = obj.get("rubric_scores")
    if not isinstance(raw_scores, list):
        return False, [], "missing rubric_scores list"

    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", "")).strip()
        if not rid:
            continue
        model_score = _coerce_binary(item.get("score"))
        bpy_check = _coerce_check(item.get("bpy_check"))
        visual_check = _coerce_check(item.get("visual_check"))
        applicable = [c for c in (bpy_check, visual_check) if c in ("pass", "fail")]
        if enforce_cross_check and applicable:
            score = 1 if all(c == "pass" for c in applicable) else 0
        else:
            score = model_score
        if score is None:
            # No usable verdict (no valid score AND no applicable sub-check) -> treat as missing,
            # matching the pre-existing "score is None -> skip" behavior.
            continue
        by_id[rid] = {
            "id": rid,
            "score": score,
            "evidence": str(item.get("evidence", "")).strip(),
            "method": str(item.get("method", "")).strip() or "unknown",
            "bpy_check": bpy_check,
            "visual_check": visual_check,
            "model_score": model_score,
            "cross_check_applied": bool(applicable) and model_score is not None and score != model_score,
            "source_views": item.get("source_views") if isinstance(item.get("source_views"), list) else [],
            "confidence": str(item.get("confidence", "medium")).strip() or "medium",
        }

    normalized: list[dict[str, Any]] = []
    missing: list[str] = []
    for rubric in rubrics:
        rid = str(rubric.get("id", "")).strip()
        if rid in by_id:
            normalized.append(by_id[rid])
        else:
            missing.append(rid)
    if missing:
        return False, normalized, f"missing scores for ids: {missing[:10]}"
    return True, normalized, ""


def finalize_judgement(
    task_dir: Path,
    rubrics: list[dict[str, Any]],
    raw_obj: dict[str, Any] | None,
    cfg: dict[str, Any],
    *,
    render_images: list[Path] | None = None,
    extra_images: list[Path] | None = None,
    reference_images: list[Path] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the canonical judgement object from the agent's raw verdicts.

    Pure/deterministic given ``raw_obj``: reuses ``vj.build_score_summary`` /
    ``vj.score_dimensions`` and the same uncapped fusion logic as the vision judge
    (``judge_blender_binary_rubrics.py`` ~1636-1662). Unit-testable without spawning an agent."""
    render_images = render_images or []
    extra_images = extra_images or []
    reference_images = reference_images or []
    images = render_images + extra_images

    ok, scores, error = normalize_agent_rubrics(
        raw_obj, rubrics, enforce_cross_check=bool(cfg.get("cross_check", True))
    )
    total = sum(item["score"] for item in scores)
    max_score = len(rubrics)
    score_summary = vj.build_score_summary(rubrics, scores)

    rubric_score_0_100 = score_summary["weighted_score_0_100"]
    dim_result = vj.score_dimensions(raw_obj, vj.DIMENSIONS) if cfg.get("dimension_pass", True) else None
    if cfg.get("dimension_pass", True) and (not dim_result or not dim_result["ok"]):
        ok = False
        error = "; ".join(x for x in [error, "dimension judge failed"] if x)

    # Worst-of-both for the 1-10 dimensions: if the agent flagged a dimension's .blend internals as
    # looking correct/better than what its renders actually show (`render_conflict`), the visible
    # result governs -- clamp that dimension into the low band. This mirrors the rubric cross-check
    # and runs before score fusion. The routed public protocol uses the VLM for dimensions.
    dim_conflict_cap = cfg.get("dimension_conflict_cap")
    if dim_result and dim_result.get("ok") and dim_conflict_cap is not None:
        raw_dims = (raw_obj or {}).get("dimensions")
        conflict_ids = {
            str(d.get("id", "")).strip()
            for d in (raw_dims if isinstance(raw_dims, list) else [])
            if isinstance(d, dict) and _truthy(d.get("render_conflict"))
        }
        conflict_ids.discard("")
        capped_any = False
        for row in dim_result.get("dimension_scores", []):
            if str(row.get("id", "")).strip() in conflict_ids:
                row["render_conflict"] = True
                if int(row.get("score", 0)) > dim_conflict_cap:
                    row["original_score"] = row.get("score")
                    row["score"] = dim_conflict_cap
                    issues = row.get("issues")
                    if not isinstance(issues, list):
                        issues = []
                    issues.append(
                        f"render conflicts with .blend internals; capped to <= {dim_conflict_cap}"
                    )
                    row["issues"] = issues
                    capped_any = True
        if capped_any:
            denom = 10 * len(vj.DIMENSIONS)
            total_dim = sum(int(r.get("score", 0)) for r in dim_result["dimension_scores"])
            dim_result["dimension_score_0_100"] = (100.0 * total_dim / denom) if denom else 0.0

    w = float(cfg.get("dimension_weight", 0.5))
    if dim_result and dim_result["ok"]:
        dimension_score_0_100 = dim_result["dimension_score_0_100"]
        fused_score_0_100 = w * dimension_score_0_100 + (1.0 - w) * rubric_score_0_100
    else:
        dimension_score_0_100 = None
        fused_score_0_100 = rubric_score_0_100  # fall back to rubric-only

    cap_applied = False
    cap_reason = ""

    obj: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_dir": str(task_dir),
        "judge_kind": "agent",
        "model": cfg.get("model", ""),
        "reference_image_files": [vj.rel(p, task_dir) for p in reference_images],
        "image_files": [vj.rel(p, task_dir) for p in images],
        "rendered_image_files": [vj.rel(p, task_dir) for p in render_images],
        "extra_image_files": [vj.rel(p, task_dir) for p in extra_images],
        "rubric_count": len(rubrics),
        "score": total,
        "max_score": max_score,
        "normalized_score": (total / max_score) if max_score else 0.0,
        "raw_pass": score_summary["raw_pass"],
        "raw_total": score_summary["raw_total"],
        "raw_score_0_100": score_summary["raw_score_0_100"],
        "weighted_pass": score_summary["weighted_pass"],
        "weighted_total": score_summary["weighted_total"],
        "weighted_score_0_100": score_summary["weighted_score_0_100"],
        "score_summary": score_summary,
        "passed_count": total,
        "failed_count": max_score - total,
        "rubric_scores": scores,
        "overall_notes": str((raw_obj or {}).get("overall_notes", "")).strip(),
        # the agent judge does not run a separate blind pass
        "blind_pass": False,
        "blind_description": None,
        "blind_reference_description": None,
        "rubric_batch_size": 0,
        "rubric_chunk_count": 1,
        # --- holistic dimension scoring + fusion ---
        "dimension_pass": bool(cfg.get("dimension_pass", True)),
        "dimension_weight": w,
        "rubric_score_0_100": rubric_score_0_100,
        "dimension_score_0_100": dimension_score_0_100,
        "fused_score_0_100": fused_score_0_100,
        "cap_applied": cap_applied,
        "cap_reason": cap_reason,
        "dimension_scores": (dim_result or {}).get("dimension_scores", []),
        "core_objects_recognizable": (dim_result or {}).get("core_objects_recognizable"),
        "catastrophic_failure": (dim_result or {}).get("catastrophic_failure"),
        "dimension_missing": (dim_result or {}).get("missing_dimensions", []),
        "dimension_notes": str((raw_obj or {}).get("dimension_notes", (raw_obj or {}).get("overall_notes", ""))).strip(),
        "ok": ok,
    }
    if dim_result and not dim_result["ok"]:
        obj["dimension_error"] = dim_result.get("error", "")
    if not ok:
        obj["error"] = error
    if extra_meta:
        obj.update(extra_meta)
    if obj.get("agent_error"):
        obj["ok"] = False
        obj["error"] = obj["agent_error"]
    return obj


# --------------------------------------------------------------------------------------
# Agent prompt + invocation
# --------------------------------------------------------------------------------------
def render_rubrics_for_agent(rubrics: list[dict[str, Any]]) -> str:
    lines = []
    for item in rubrics:
        lines.append(
            json.dumps(
                {
                    "id": item.get("id"),
                    "category": item.get("category") or item.get("category_v2"),
                    "evidence_type": item.get("evidence_type"),
                    "criterion": item.get("criterion"),
                    "pass_condition": item.get("pass_condition"),
                    "fail_condition": item.get("fail_condition"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def build_prompt(
    task_dir: Path,
    prompt_text: str,
    rubrics: list[dict[str, Any]],
    *,
    blend_files: list[Path],
    render_images: list[Path],
    extra_images: list[Path],
    reference_images: list[Path],
    blender_bin: str,
    output_rel: str,
    use_bpy: bool,
    render_required: bool = True,
) -> str:
    def rels(paths: list[Path]) -> str:
        return "\n".join(f"  - {vj.rel(p, task_dir)}" for p in paths) or "  (none)"

    inventory = evidence_checks.file_inventory(task_dir / ".eval_workspace")
    inventory_path = task_dir / output_rel / "submission_manifest.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    # The full deterministic manifest stays available even for long sequences.
    inventory_preview = json.dumps({**inventory, "files": inventory["files"][:80]}, ensure_ascii=False)
    checks_script = str(Path(evidence_checks.__file__).resolve())
    dims = "\n".join(f"  - {d['id']}: {d['name']} — {d['desc']}" for d in vj.DIMENSIONS)
    raw_rel = f"{output_rel}/{RAW_NAME}"
    bpy_rel = f"{output_rel}/{BPY_DUMP_NAME}"
    renders_rel = f"{output_rel}/agent_renders"

    # RENDERING POLICY (both modes): rendering is expensive and, in this headless GPU container,
    # EEVEE/Workbench silently degrade or fail — any scoring render MUST go through Cycles. When
    # render_required is False (evidence routing: this agent only owns structural / scene_summary
    # rubrics decided from the .blend internals) the agent should NOT render unless a specific
    # rubric genuinely cannot be settled from bpy.
    cycles_rule = (
        "Whenever you render, you MUST use Cycles — never EEVEE/Workbench, which are unreliable "
        "headless here and degrade quality. In your render script set "
        "`scene.render.engine='CYCLES'`, select the GPU "
        "(`bpy.context.preferences.addons['cycles'].preferences.compute_device_type='OPTIX'` or "
        "`'CUDA'`, refresh devices, enable them, then `scene.cycles.device='GPU'`), and use enough "
        "samples (>=64) so the image is not noisy. Confirm the engine is actually CYCLES before "
        "trusting a render."
    )
    export_scan_rule = (
        "When a rubric concerns an EXPORTED deliverable (a PNG image, a rendered still, an "
        "animation/video such as .mp4/.mov/.gif/.webm, or a baked texture), verify its existence "
        "by scanning the WHOLE submission export tree — check `exports/`, `output/`, `renders/`, "
        "and the submission root recursively (e.g. `find . -type f \\( -iname '*.png' -o -iname "
        "'*.mp4' -o -iname '*.mov' -o -iname '*.gif' -o -iname '*.webm' \\)`). Do NOT assume one "
        "fixed folder; score such a rubric on whether the file actually exists somewhere under the "
        "export paths."
    )
    if render_required:
        render_intro = (
            "Author-delivered images are listed below. To judge any appearance/visual "
            "rubric inspect the relevant author file, or render targeted scene evidence in STEP 3 when needed. If you "
            "have neither inspected a relevant author image nor produced a relevant render, do not pass a visual rubric. " + cycles_rule
        )
        step3 = (
            "STEP 3 — Render MULTIPLE targeted images yourself, GUIDED by the rubric + prompt + "
            "dimensions. " + cycles_rule + " Open a COPY of the .blend and render as many views as "
            "you need — an overview PLUS the specific angles, close-ups and framings that expose what "
            "the rubrics and each dimension actually check (e.g. a close-up on a part a rubric names; "
            "an angle showing support/contact for `physics`; a wide shot for "
            "`layout_spatial`/`completeness`; a material close-up for `lighting_material`/`style_color`). "
            "You may adjust temporary cameras in your COPY. Preserve geometry, materials, lights, animation, visibility and color management. Save all images to " + renders_rel +
            "/ with descriptive names and note what each one shows."
        )
    else:
        render_intro = (
            "You are scoring STRUCTURAL / scene_summary rubrics that are decided from the .blend "
            "INTERNALS via bpy, NOT from appearance. You normally do NOT need to render anything, and "
            "you should NOT re-render unless a specific rubric genuinely cannot be settled from the "
            ".blend internals. " + cycles_rule
        )
        step3 = (
            "STEP 3 — Rendering is OPTIONAL for these rubrics and usually unnecessary. Do NOT render "
            "just to have an image. Render your OWN image(s) into " + renders_rel + "/ ONLY if a "
            "specific rubric truly requires visual confirmation that bpy cannot give; if you do, "
            + cycles_rule + " Preserve geometry, materials, lights, animation, visibility and color management. If no rubric needs a render, skip this step entirely (that is expected "
            "and not an error)."
        )

    return f"""You are an AGENT-AS-JUDGE for a Blender scene-reconstruction benchmark. You are NOT solving
the task. You SCORE an already-submitted result (a .blend) against a fixed rubric list and a set of
holistic dimensions. Unlike an image-only judge you can (a) open the submitted .blend to inspect the
"original code" — its objects, modifiers, geometry-node graphs, shape keys, drivers, materials, scene
settings, and any embedded text/script data-blocks — AND (b) render your OWN targeted images from it.

# Task prompt given to the solver (the TARGET; the submission is what you judge)
{prompt_text}

# Submitted artifacts (paths relative to your working directory)
Submitted .blend file(s) — the "original code" (READ-ONLY: work on a COPY; never edit/re-save the original):
{rels(blend_files)}
Reference / target image(s) (the goal; do NOT credit the submission for something only here):
{rels(reference_images)}
Author-delivered images already discovered (these are original files, not your renders):
{rels(extra_images)}
Original file inventory (first 80 records; includes PNG IHDR bit depth):
{inventory_preview}
Read the FULL inventory at `{vj.rel(inventory_path, task_dir)}` when needed. Inspect the whole `.eval_workspace/` tree, including root-level exports, nested frame sequences, FBX/EXR and accompanying textures. A configured absolute output path in the .blend may be stale: it is not evidence that a delivered file is absent. Do not count evaluator renders, input references or sampled video frames as author-delivered files. The hashes of supplied inputs are listed in `input_media_sha256.json` when present.
Blender executable: {blender_bin}
Read-only evidence helper: `{checks_script}`
To inspect camera optics and rigid-body units, run the original file with autoexec disabled:
`{blender_bin} --background <original.blend> --disable-autoexec --python {checks_script} -- --output <probe.json>`
For a second-UV non-overlap condition add `--uv-layer <actual-layer-name>`. Inspect the relevant low-poly meshes; an incomplete scan is unknown, never proof of no overlap. The helper checks original mesh UVs; inspect evaluated geometry too when modifiers generate or alter the deliverable UVs.

{render_intro}

Work in this EXACT order:

STEP 1 — Plan. Read the task prompt above, EVERY rubric below (note each rubric's `evidence_type`),
and the {len(vj.DIMENSIONS)} holistic dimensions with their descriptions. Decide which scene facts to
extract from the .blend and which views / close-ups / angles will best reveal the rubric criteria and
each dimension.

STEP 2 — Inspect the .blend internals (the "original code") with bpy. Run Blender headless
(`{blender_bin} --background <copy-of-blend> --python /tmp/probe.py`) to extract exactly what the
rubrics/dimensions need: object list & names, mesh triangle counts, modifiers / Geometry-Nodes
node-group inputs & values, shape-key/blendshape names & counts, drivers, object bounding-box
dimensions (mm), scene unit settings (`scene.unit_settings`), render engine/samples/resolution,
material node graphs (Principled/SSS/IOR), UV layout & overlap, physics / rigid-body / Dynamic-Paint
settings, collections, and any embedded Text data-blocks (scripts). Save your probe script + raw
output to `{bpy_rel}` (JSON) as an audit trail. DO NOT modify or re-save the original .blend.
{export_scan_rule}

Evidence rules:
- PNG bit depth comes from the original PNG IHDR, not Pillow's RGB/RGBA mode; Pillow commonly presents 16-bit RGB files as RGB. Use the manifest or evidence helper.
- For alpha, inspect the actual channel and composite over a checkerboard. A black-looking RGB preview neither proves nor disproves transparency. Validate silhouettes in the actual alpha image.
- Convert rigid-body substeps_per_frame using effective fps = fps/fps_base. For example, 10 substeps/frame at 24 fps equals 240 steps/second; record time_scale separately. Do not compare different units directly.
- Camera downward pitch is measured from the world optical axis (local -Z) to the horizontal plane, not from Euler X. Euler X about 65 degrees can correspond to downward pitch about 25 degrees.
- UV layer names do not prove non-overlap. Check positive-area triangle intersections within the requested layer, allowing shared boundaries. Check actual coordinates, not beauty renders or labels.
- Preserve the original .blend as the source of settings. If using a copy for rendering, do not report changed render/world settings as original facts.
- Final-image composition, focus, noise and required deliverable appearance must be checked against the designated author final file. Previews or your rerenders cannot replace that file. Scene existence/geometry can use targeted views.
- Exact numerical facts, metadata and UV tests are structural checks: use visual_check=na where pixels cannot establish the numeric fact. A dimension that looks plausible is not an independent numeric measurement.

{step3}

STEP 4 — Score EVERY rubric by CROSS-CHECKING both evidence sources. Do NOT pick a single method and
stop. For each rubric produce TWO independent sub-verdicts, each `pass` / `fail` / `na`:
- `bpy_check`: does the requirement hold in the .blend INTERNALS (bpy)? Use `na` ONLY when the rubric
  has no structural/internal aspect (e.g. "reads as a real chair" — a purely visual judgment).
- `visual_check`: does the requirement hold in YOUR rendered images? Use `na` ONLY when it is
  genuinely invisible in any render you could make (e.g. triangle-count / UV-overlap / a driver).
At least one sub-verdict must be non-`na`. Whenever BOTH apply — colour, material, a part being
present AND correctly shaped, a dimension that is visible, layout — you MUST fill BOTH, and the rubric
passes ONLY if BOTH pass.

THE VISIBLE RENDER IS GROUND TRUTH for appearance. A correct internal value NEVER rescues a wrong
appearance, and vice-versa. Concretely: if the Principled BSDF base colour is red in the node graph
(`bpy_check`="pass") but the object renders grey/wrong (`visual_check`="fail"), the rubric FAILS —
because the material is on the wrong slot, unassigned to the visible faces, overridden by emission /
vertex colour, or the UVs are broken. Any such mismatch is a FAIL, not a pass.

Set the final `score`: 1 only if EVERY applicable (non-`na`) sub-verdict is `pass`, otherwise 0 (the
harness re-derives this from your sub-verdicts, so make them consistent). Score what is ACTUALLY
present in the submission (.blend + your renders), never what the prompt merely asked for. If truly
unverifiable even with bpy + targeted renders, score 0 and say why. Also set `method`="bpy" or
"visual" for the source that was decisive, and give a short concrete `evidence` string that cites the
bpy fact AND/OR the render file for each sub-verdict you made.

RUBRICS (one JSON object per line):
{render_rubrics_for_agent(rubrics)}

STEP 5 — Rate these {len(vj.DIMENSIONS)} dimensions as integers 1-10, scored by WHAT THE RENDER
VISIBLY ACHIEVES. The .blend internals may only CONFIRM or LOWER a dimension — a correct internal
value can NEVER raise a dimension above the visible result. Be strict & calibrated: 9-10 only for
essentially faithful results; 7-8 good with minor issues; 5-6 clear problems; 3-4 major failures;
1-2 absent/broken. Do NOT cluster at 7-8. Each dimension needs concrete `reasons` (cite views) + an
`issues` list.
{dims}
For EACH dimension also set a boolean `render_conflict`: true when the .blend internals look
correct/better than what the renders actually show, or the two disagree (e.g. materials/colours/parts
that are right in bpy but wrong or missing on screen). When `render_conflict` is true you MUST put
that dimension in the LOW band (the visible result governs; the harness additionally clamps such
dimensions).
Also two booleans: `core_objects_recognizable` (main requested objects independently recognizable) and
`catastrophic_failure` (scene fundamentally fails / unusable as requested).

STEP 6 — Write your verdict as a single JSON object to `{raw_rel}` (create the folder if needed),
EXACTLY this shape and nothing else in the file:
{{
  "rubric_scores": [
    {{"id": "R01", "score": 0 or 1, "bpy_check": "pass"/"fail"/"na", "visual_check": "pass"/"fail"/"na", "method": "bpy" or "visual", "evidence": "short reason"}}
    // one entry per rubric id above, all of them
  ],
  "dimensions": [
    {{"id": "scene_fidelity", "score": 1-10, "render_conflict": true or false, "reasons": "...", "issues": ["..."]}}
    // one entry per dimension id above, all of them
  ],
  "core_objects_recognizable": true or false,
  "catastrophic_failure": true or false,
  "overall_notes": "short summary of the main strengths and failures"
}}

Rules: include every rubric id and every dimension id exactly once; integer scores; each rubric's
`score` must equal 1 iff every applicable (non-`na`) sub-verdict is `pass`; valid JSON (no trailing
commas/comments). When `{raw_rel}` is written you are done.
"""


def retry_preface(attempt: int, max_attempts: int, raw_rel: str, reason: str) -> str:
    """Preface prepended to the judge prompt on a retry.

    Instead of silently dropping a non-conforming agent judgement to the VLM the moment it fails
    to parse, we re-tell the agent exactly why its previous ``agent_judgement_raw.json`` was
    rejected and ask it to re-emit a complete, valid JSON verdict."""
    return (
        f"RETRY {attempt}/{max_attempts}: your previous write to `{raw_rel}` was REJECTED because: "
        f"{reason}. Common causes: the file was not valid JSON, or some rubric ids were "
        f"missing/duplicated. Re-read EVERY rubric id listed below and re-emit the COMPLETE JSON "
        f"object exactly per the STEP 6 shape — include every rubric id and every dimension id "
        f"exactly once, integer scores, valid JSON only (no comments / no trailing commas), and "
        f"nothing else in the file. Overwrite `{raw_rel}` with the corrected JSON.\n\n"
    )


def run_one_task(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    task_dir = Path(str(result.get("task_dir") or "")).expanduser()
    rubrics = result.get("rubrics") or []
    tname = task_dir.name
    # Under evidence_type routing this judge only scores a subset (e.g. scene_summary). Filter
    # immediately so is_complete/normalize/finalize all key off the same subset id-set, and the
    # VLM judge owns the rest. Empty set = score every rubric (legacy behavior).
    ev_types = cfg.get("agent_evidence_types") or set()
    if ev_types:
        rubrics = [r for r in rubrics if str(r.get("evidence_type") or "").strip().lower() in ev_types]
        if not rubrics:
            vj.log.info("[agent-judge] skip %s: no rubrics with evidence_type in %s", tname, sorted(ev_types))
            return None
    if not task_dir.is_dir():
        return {"task_dir": str(task_dir), "ok": False, "error": "task_dir not found", "rubric_scores": []}
    if not rubrics:
        return {"task_dir": str(task_dir), "ok": False, "error": "no rubrics", "rubric_scores": []}

    out_path = vj.output_path(task_dir, cfg["output_dir"])
    rubric_ids = {str(r.get("id")) for r in rubrics}
    if not cfg["overwrite"] and vj.is_complete_judgement(out_path, rubric_ids):
        vj.log.info("[agent-judge] skip complete %s", tname)
        return json.loads(out_path.read_text(encoding="utf-8"))

    render_images = vj.find_render_images(task_dir, cfg["image_glob"], cfg["max_images"])
    extra_images = vj.find_extra_submission_images(task_dir, cfg["extra_image_globs"], cfg["max_extra_images"])
    reference_images = (
        [] if cfg["no_reference_images"]
        else vj.find_reference_images(task_dir, cfg["reference_image_glob"], cfg["max_reference_images"])
    )
    # The public submission contract names the primary project explicitly.
    primary_project = task_dir / ".eval_workspace" / "final.blend"
    blend_files = [primary_project] if primary_project.is_file() else []

    output_rel = Path(cfg["output_dir"])
    output_rel = output_rel if not output_rel.is_absolute() else Path(tname)
    output_rel_str = output_rel.as_posix()
    (task_dir / output_rel).mkdir(parents=True, exist_ok=True)
    raw_path = task_dir / output_rel / RAW_NAME
    if cfg["overwrite"] and raw_path.exists():
        raw_path.unlink()

    # Renders are required only when this judge owns visual rubrics. Under evidence routing it owns
    # only scene_summary (structural, decided from bpy) → do NOT force a render (that render is
    # wasteful and, if it fails headless, would falsely error the whole scene_summary verdict).
    render_required = (not ev_types) or bool(ev_types & {"either", "render", "visual", "image"})
    prompt = build_prompt(
        task_dir, str(result.get("prompt", "")), rubrics,
        blend_files=blend_files, render_images=render_images, extra_images=extra_images,
        reference_images=reference_images, blender_bin=cfg["blender_bin"],
        output_rel=output_rel_str, use_bpy=cfg["use_bpy"] and bool(blend_files),
        render_required=render_required,
    )

    base_prompt = prompt
    raw_rel = f"{output_rel_str}/{RAW_NAME}"
    max_attempts = max(1, int(cfg.get("max_attempts", 3)))
    returncode: int | None = None
    agent_error = ""
    raw_obj: dict[str, Any] | None = None
    attempts_used = 0
    last_reason = ""
    stdout_rel = f"{output_rel_str}/agent_judge_stdout.jsonl"

    if cfg["dry_run"]:
        (task_dir / output_rel / "agent_judge_prompt.txt").write_text(base_prompt, encoding="utf-8")
    else:
        # Instead of dropping non-conforming agent output straight to the VLM, re-prompt the agent
        # with what was wrong and let it re-score, up to max_attempts. Only JSON-conformance misses
        # are retried; a timeout / missing-claude / spawn error is infrastructure, not the agent's
        # scoring, so we stop and let the VLM fallback (load_agent_verdicts guard) handle it.
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            if attempt == 1:
                prompt_n = base_prompt
                stdout_rel = f"{output_rel_str}/agent_judge_stdout.jsonl"
            else:
                prompt_n = retry_preface(attempt, max_attempts, raw_rel, last_reason) + base_prompt
                stdout_rel = f"{output_rel_str}/agent_judge_stdout.attempt{attempt}.jsonl"
                if raw_path.exists():
                    raw_path.unlink()  # drop the previous bad file so it can't be misread as success
            cmd = [
                "claude", "--verbose",
                "--max-turns", str(cfg["max_turns"]),
                "--output-format", "stream-json",
                "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "LS",
                "--disallowedTools", "WebFetch", "WebSearch",
                "-p", prompt_n,
            ]
            env = os.environ.copy()
            env["ANTHROPIC_BASE_URL"] = cfg["anthropic_base_url"]
            env.setdefault("ANTHROPIC_API_KEY", cfg["anthropic_api_key"])
            env.setdefault("ANTHROPIC_AUTH_TOKEN", "")
            if cfg["model"]:
                env["ANTHROPIC_MODEL"] = cfg["model"]
            agent_error = ""
            try:
                with (task_dir / stdout_rel).open("w", encoding="utf-8") as stdout:
                    completed = subprocess.run(
                        cmd, cwd=str(task_dir), env=env,
                        stdout=stdout, stderr=subprocess.STDOUT,
                        text=True, timeout=cfg["timeout"], check=False,
                    )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                agent_error = f"agent judge timed out after {cfg['timeout']}s"
            except FileNotFoundError:
                returncode = 127
                agent_error = "claude command not found"
            except Exception as exc:  # noqa: BLE001
                returncode = 1
                agent_error = repr(exc)

            # A terminal API failure is not a malformed judgement. In particular, an explicit
            # content refusal must stop here, without a JSON repair retry or VLM fallback.
            agent_error = agent_error or cli_api_error(task_dir / stdout_rel)
            if agent_error:
                break

            # Parse this attempt's raw output (with json_repair fallback).
            raw_obj = None
            if raw_path.exists():
                try:
                    text = raw_path.read_text(encoding="utf-8")
                    try:
                        raw_obj = json.loads(text)
                    except Exception:  # noqa: BLE001
                        raw_obj = json.loads(repair_json(text))
                except Exception as exc:  # noqa: BLE001
                    raw_obj = None
                    last_reason = f"could not parse {RAW_NAME}: {exc!r}"
            else:
                last_reason = f"agent did not write {RAW_NAME}"

            # Conformance probe (reuses the same normalization finalize will apply): ok iff valid
            # JSON with a score for every rubric id.
            if raw_obj is not None:
                probe_ok, _, probe_reason = normalize_agent_rubrics(
                    raw_obj, rubrics, enforce_cross_check=bool(cfg.get("cross_check", True))
                )
                if probe_ok:
                    break  # valid, complete scoring — done
                last_reason = probe_reason or "agent scoring incomplete"

            if attempt < max_attempts:
                vj.log.info(
                    "[agent-judge] task=%s attempt=%d/%d invalid (%s) — retrying",
                    tname, attempt, max_attempts, last_reason,
                )
                continue
            # Attempts exhausted, still non-conforming: flag the error so finalize records it and
            # load_agent_verdicts' guard falls back to VLM (the last resort).
            agent_error = f"agent output invalid after {max_attempts} attempts: {last_reason}"

    # Enforce a self-render ONLY when this judge owns visual rubrics (render_required). Under
    # evidence routing it scores structural / scene_summary rubrics from bpy, so an empty
    # agent_renders/ is expected and must NOT be treated as an error.
    renders_dir = task_dir / output_rel / "agent_renders"
    agent_render_count = len(list(renders_dir.glob("*.png"))) if renders_dir.exists() else 0
    if render_required and agent_render_count == 0 and not agent_error:
        agent_error = "agent produced no self-rendered images (agent_renders/ empty) — visual rubrics unverified"

    extra_meta = {
        "agent_returncode": returncode,
        "agent_stdout_path": stdout_rel,
        "agent_raw_path": f"{output_rel_str}/{RAW_NAME}",
        "bpy_dump_path": f"{output_rel_str}/{BPY_DUMP_NAME}",
        "agent_renders_path": f"{output_rel_str}/agent_renders",
        "agent_render_count": agent_render_count,
        "agent_attempts": attempts_used,
    }
    if agent_error:
        extra_meta["agent_error"] = agent_error

    obj = finalize_judgement(
        task_dir, rubrics, raw_obj, cfg,
        render_images=render_images, extra_images=extra_images, reference_images=reference_images,
        extra_meta=extra_meta,
    )
    obj["output_path"] = str(out_path)
    vj.atomic_write_json(out_path, obj)
    vj.log.info(
        "[agent-judge] task=%s ok=%s fused=%.1f rubric=%.1f dim=%s self_renders=%d attempts=%d rc=%s",
        tname, obj.get("ok"), obj.get("fused_score_0_100") or 0.0,
        obj.get("weighted_score_0_100") or 0.0, obj.get("dimension_score_0_100"),
        agent_render_count, attempts_used, returncode,
    )
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent-as-Judge for Blender trajectory eval")
    p.add_argument("--rubrics-batch", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-output", default="")
    p.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB)
    p.add_argument("--extra-image-glob", action="append", default=[])
    p.add_argument("--reference-image-glob", default="**/image_assets/*")
    p.add_argument("--no-reference-images", action="store_true")
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument("--max-extra-images", type=int, default=4)
    p.add_argument("--max-reference-images", type=int, default=4)
    p.add_argument("--model", default=os.environ.get("AGENT_JUDGE_MODEL") or os.environ.get("MODEL", ""))
    p.add_argument("--max-turns", type=int, default=int(os.environ.get("AGENT_JUDGE_MAX_TURNS", "500")))
    p.add_argument(
        "--max-attempts", type=int,
        default=int(os.environ.get("AGENT_JUDGE_MAX_ATTEMPTS", "3")),
        help="Max agent-judge attempts per task. When the agent produces non-conforming JSON "
             "(missing file / unparseable / missing rubric ids), it is re-prompted to re-score "
             "up to this many times before the VLM fallback is used as a last resort.",
    )
    p.add_argument("--timeout", type=int, default=int(os.environ.get("AGENT_JUDGE_TIMEOUT", "14400")))
    p.add_argument("--task-concurrency", type=int, default=int(os.environ.get("AGENT_JUDGE_TASK_CONCURRENCY", "2")))
    p.add_argument("--dimension-weight", type=float, default=0.5)
    p.add_argument("--no-dimension-pass", action="store_true")
    p.add_argument("--dimension-conflict-cap", type=int, default=4,
                   help="Clamp any dimension the agent flags with render_conflict=true to <= this "
                        "1-10 value (worst-of-both; the visible render governs). Default 4.")
    p.add_argument("--no-dimension-conflict-cap", action="store_true",
                   help="Disable the render_conflict dimension clamp.")
    p.add_argument("--no-rubric-cross-check", action="store_true",
                   help="Disable conjunctive rubric scoring; fall back to the model's self-reported "
                        "0/1 score (ignore bpy_check/visual_check).")
    p.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    p.add_argument("--no-bpy", action="store_true")
    p.add_argument("--anthropic-base-url", default=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    p.add_argument("--anthropic-api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--agent-evidence-types",
        default="",
        help=(
            "Comma list of evidence_type values this agent judge should score. Empty (default) "
            "means score every rubric (legacy behavior). Under evidence_type routing set this to "
            "'scene_summary' so the agent only judges those rubrics and the VLM handles the rest."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rubrics_batch = Path(args.rubrics_batch)
    results = vj.load_rubric_results(rubrics_batch)
    if args.limit > 0:
        results = results[: args.limit]

    extra_globs = args.extra_image_glob or list(vj.DEFAULT_EXTRA_IMAGE_GLOBS)
    cfg = {
        "output_dir": args.output_dir,
        "image_glob": args.image_glob,
        "extra_image_globs": extra_globs,
        "reference_image_glob": args.reference_image_glob,
        "no_reference_images": args.no_reference_images,
        "max_images": args.max_images,
        "max_extra_images": args.max_extra_images,
        "max_reference_images": args.max_reference_images,
        "model": args.model,
        "max_turns": args.max_turns,
        "max_attempts": args.max_attempts,
        "timeout": args.timeout,
        "dimension_weight": args.dimension_weight,
        "dimension_pass": not args.no_dimension_pass,
        "dimension_conflict_cap": None if args.no_dimension_conflict_cap else args.dimension_conflict_cap,
        "cross_check": not args.no_rubric_cross_check,
        "blender_bin": args.blender_bin,
        "use_bpy": not args.no_bpy,
        "anthropic_base_url": args.anthropic_base_url,
        "anthropic_api_key": args.anthropic_api_key,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
        "agent_evidence_types": vj.parse_type_set(args.agent_evidence_types),
    }

    outputs: list[dict[str, Any]] = []
    workers = max(1, int(args.task_concurrency))
    if workers == 1:
        for result in results:
            out = run_one_task(result, cfg)
            if out is not None:
                outputs.append(out)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one_task, result, cfg): result for result in results}
            for fut in as_completed(futures):
                try:
                    out = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = futures[fut]
                    out = {"task_dir": str(r.get("task_dir")), "ok": False,
                           "error": repr(exc), "rubric_scores": []}
                if out is not None:
                    outputs.append(out)

    outputs.sort(key=lambda o: str(o.get("task_dir")))
    batch_output = Path(args.batch_output) if args.batch_output else rubrics_batch.parent / "judge_agent_batch.json"
    ok_count = sum(1 for item in outputs if item.get("ok"))
    batch = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rubrics_batch": str(rubrics_batch),
        "model": args.model,
        "judge_kind": "agent",
        "task_count": len(outputs),
        "ok_count": ok_count,
        "score_summary": vj.build_batch_score_summary(outputs),
        "results": outputs,
    }
    vj.atomic_write_json(batch_output, batch)
    vj.log.info("[agent-judge] wrote batch %s ok=%d/%d", batch_output, ok_count, len(outputs))
    return 0 if ok_count == len(outputs) else 2


if __name__ == "__main__":
    sys.exit(main())
