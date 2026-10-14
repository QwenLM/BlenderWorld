"""Restore bundled image paths in memory when Blender opens an archived project.

The saved project is never rewritten. Ambiguous filenames remain unresolved;
an evaluator must not silently substitute another task's texture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

_handler = None


def workspace_for(blend_path, root):
    root = Path(root).resolve()
    blend = Path(blend_path).resolve()
    if not blend.is_relative_to(root):
        return None
    for parent in blend.parents:
        if not parent.is_relative_to(root):
            break
        if parent.name == ".eval_workspace":
            return parent
        candidate = parent / ".eval_workspace"
        if candidate.is_dir():
            return candidate
    return None


def find_asset(saved_path, workspace):
    """Prefer the longest matching archived suffix; never pick an ambiguous name."""
    workspace = Path(workspace).resolve()
    parts = PurePosixPath(str(saved_path).replace("\\", "/")).parts
    if not parts or ".." in parts:
        return None
    files = [
        p
        for p in workspace.rglob(parts[-1])
        if p.is_file() and p.resolve().is_relative_to(workspace)
    ]
    # Glob metacharacters in a legitimate filename must not select another file.
    files = [p for p in files if p.name == parts[-1]]
    for count in range(len(parts), 0, -1):
        suffix = parts[-count:]
        matches = [p for p in files if p.relative_to(workspace).parts[-count:] == suffix]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
    return None


def restore_images(bpy, root):
    if not bpy.data.filepath:
        return []
    workspace = workspace_for(bpy.data.filepath, root)
    if workspace is None:
        return []
    changes = []
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file or not image.filepath:
            continue
        old = image.filepath
        if Path(bpy.path.abspath(old, library=image.library)).is_file():
            continue
        replacement = find_asset(old, workspace)
        if replacement is None:
            changes.append({"action": "unresolved_image", "image": image.name, "path": old})
            continue
        image.filepath = str(replacement)
        try:
            image.reload()
            if not all(image.size):
                raise RuntimeError("image has no readable pixels")
        except (RuntimeError, OSError, ValueError) as exc:
            image.filepath = old
            changes.append(
                {
                    "action": "image_reload_failed",
                    "image": image.name,
                    "path": old,
                    "error": str(exc),
                }
            )
        else:
            changes.append(
                {
                    "action": "restored_image",
                    "image": image.name,
                    "from": old,
                    "to": str(replacement),
                }
            )
    return changes


def compatible_denoising(scenes):
    changes = []
    for scene in scenes:
        cycles = scene.cycles
        if cycles.use_denoising and cycles.denoiser == "OPTIX":
            # OptiX rendering can work even when its denoiser cannot initialize.
            # OIDN on CPU preserves denoising without changing render samples.
            cycles.denoiser = "OPENIMAGEDENOISE"
            if hasattr(cycles, "denoising_use_gpu"):
                cycles.denoising_use_gpu = False
            changes.append(
                {
                    "action": "denoiser_compatibility",
                    "scene": scene.name,
                    "from": "OPTIX",
                    "to": "OPENIMAGEDENOISE",
                }
            )
    return changes


def apply_runtime(bpy):
    root = os.environ.get("BLENDER_EVAL_ASSET_ROOT")
    if not root or not getattr(bpy.data, "filepath", ""):
        return
    if workspace_for(bpy.data.filepath, root) is None:
        return
    changes = restore_images(bpy, root) + compatible_denoising(bpy.data.scenes)
    if not changes:
        return
    event = {"blend": bpy.data.filepath, "changes": changes}
    line = json.dumps(event, ensure_ascii=False)
    print("BLENDER_EVAL_RUNTIME " + line, flush=True)
    audit = os.environ.get("BLENDER_EVAL_RUNTIME_AUDIT")
    if audit:
        with Path(audit).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def register():
    import bpy
    from bpy.app.handlers import persistent

    global _handler
    if _handler is not None:
        return

    @persistent
    def on_load_or_render(*_args):
        apply_runtime(bpy)

    # Both hooks are needed: scripts may select another image or scene after load.
    bpy.app.handlers.load_post.append(on_load_or_render)
    bpy.app.handlers.render_pre.append(on_load_or_render)
    _handler = on_load_or_render


def unregister():
    import bpy

    global _handler
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.render_pre):
        if _handler in handlers:
            handlers.remove(_handler)
    _handler = None
