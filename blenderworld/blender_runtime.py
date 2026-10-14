"""Loaded by Blender before the render agent's script; records real render attempts."""
import fcntl
import json
import os
import sys
from pathlib import Path

import bpy
from bpy.app.handlers import persistent

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluators"))
import asset_paths

if bpy.app.version != (4, 2, 10):
    raise RuntimeError("BlenderWorld evaluation requires Blender 4.2.10")

asset_paths.register()


def update_attempt(scene, completed=False):
    folder = Path(os.environ["BLENDER_EVAL_VIEW_ROOT"]).resolve()
    target = Path(bpy.path.abspath(scene.render.filepath)).resolve()
    if not target.suffix and scene.render.use_file_extension:
        target = target.with_suffix(".png")
    if not target.is_relative_to(folder) or target.suffix.lower() != ".png":
        raise RuntimeError("Evaluator renders must be PNGs inside evaluation/multiview")
    state_path = folder / "render_attempts.json"
    with (folder / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        history = state.setdefault(str(target), [])
        if completed:
            if history:
                history[-1]["completed"] = True
        else:
            if len(history) >= 3:
                raise RuntimeError("Two corrective re-renders per view exhausted")
            if (scene.render.engine != "CYCLES" or scene.cycles.samples != 64
                    or scene.render.resolution_x != 1024 or scene.render.resolution_y != 1024
                    or scene.render.resolution_percentage != 100):
                raise RuntimeError("Required: Cycles, 64 samples, 1024x1024 at 100%")
            history.append({"engine": scene.render.engine, "samples": scene.cycles.samples,
                            "device": scene.cycles.device, "completed": False})
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)


@persistent
def before(scene, *_):
    try:
        update_attempt(scene)
    except Exception as error:
        # Blender ignores ordinary handler exceptions; abort this render subprocess.
        print("BLENDERWORLD_RENDER_REJECTED", str(error), flush=True)
        os._exit(2)


@persistent
def after(scene, *_):
    update_attempt(scene, completed=True)


bpy.app.handlers.render_pre.append(before)
bpy.app.handlers.render_write.append(after)
