"""Validate the release runtime before spending evaluator model calls."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .data import sha256


class RuntimeValidationError(RuntimeError):
    """An evaluator installation error, not a failed agent submission."""

RUNTIME_VERSIONS = {
    "blender": "4.2.10",
    "node": "22.17.0",
    "claude_code": "2.1.220",
    "codex": "0.147.0",
}

BLENDER_PROBE = '''import bpy, json
backends = []
prefs = bpy.context.preferences.addons['cycles'].preferences
for backend in ('OPTIX', 'CUDA'):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        devices = sorted({d.name for d in prefs.devices if d.type == backend})
        if devices:
            backends.append({'backend': backend, 'devices': devices})
    except Exception:
        pass
build = bpy.app.build_hash
if isinstance(build, bytes):
    build = build.decode('ascii', errors='replace')
print('BLENDERWORLD_RUNTIME=' + json.dumps({
    'version': '.'.join(map(str, bpy.app.version)),
    'build_hash': build, 'gpu_backends': backends}))
'''


def requirements():
    return {"versions": dict(RUNTIME_VERSIONS),
            "evaluation_tools": ["blender", "node", "claude_code"],
            "generation_tools": ["blender", "node", "codex", "claude_code"]}


def _version(command, name):
    found = shutil.which(str(command))
    if not found:
        raise RuntimeValidationError(f"BlenderWorld runtime requires {name} {RUNTIME_VERSIONS[name]}; executable not found: {command}")
    path = Path(found).resolve()
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True,
                                env={**os.environ, "DISABLE_AUTOUPDATER": "1"},
                                text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeValidationError(f"Could not check {name} version: {error}") from error
    text = result.stdout.strip()
    match = re.search(r"\bv?(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)\b", text)
    actual = match.group(1) if match else "unknown"
    if result.returncode or actual != RUNTIME_VERSIONS[name]:
        raise RuntimeValidationError(f"BlenderWorld runtime requires {name} {RUNTIME_VERSIONS[name]}, found {actual} at {path}")
    return {"path": str(path), "version": actual,
            "sha256": sha256(path)}


def inspect_runtime(blender_bin="blender"):
    """Pin evaluator tools; Codex is installed/checked by the rollout image."""
    report = {"requirements": requirements()}
    report["blender"] = _version(blender_bin, "blender")
    report["node"] = _version("node", "node")
    report["claude_code"] = _version("claude", "claude_code")
    try:
        result = subprocess.run([report["blender"]["path"], "--background", "--factory-startup",
                                 "--python-exit-code", "2", "--python-expr", BLENDER_PROBE],
                                capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeValidationError(f"Could not inspect Blender's Cycles runtime: {error}") from error
    lines = [s.removeprefix("BLENDERWORLD_RUNTIME=") for s in result.stdout.splitlines()
             if s.startswith("BLENDERWORLD_RUNTIME=")]
    if result.returncode or len(lines) != 1:
        raise RuntimeValidationError("Blender runtime inspection failed; check the Blender installation and GPU driver")
    try:
        details = json.loads(lines[0])
    except ValueError as error:
        raise RuntimeValidationError("Invalid Blender runtime report") from error
    if details.get("version") != RUNTIME_VERSIONS["blender"]:
        raise RuntimeValidationError("Blender executable and bpy runtime versions disagree")
    report["blender"].update(details)
    return report


def record_runtime(output, report):
    path = Path(output) / "runtime.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise RuntimeValidationError("Runtime versions, executables, or GPU changed; start a new run")
    if not path.exists():
        path.write_text(json.dumps(report, indent=2) + "\n")
