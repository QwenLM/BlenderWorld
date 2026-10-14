#!/usr/bin/env python3
"""Read-only evidence helpers; no task IDs or verdict overrides.

Normal Python: --inventory-root <materialized submission> --output <json>
Blender: open the original with --disable-autoexec, then --python this_file --
         --output <json> [--uv-layer <name>]. No scene is saved or rendered.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path


def cli_event_content_refusal(event):
    """Recognize provider errors, never a prompt merely mentioning filtering."""
    if not isinstance(event, dict):
        return False
    kind = event.get('type')
    status = event.get('api_error_status')
    if kind == 'result' and event.get('is_error') is True:
        text = str(event.get('result', ''))
    elif kind == 'assistant' and event.get('error'):
        message = event.get('message', {})
        content = message.get('content', []) if isinstance(message, dict) else []
        text = '\n'.join(str(block.get('text', '')) for block in content
                         if isinstance(block, dict) and block.get('type') == 'text')
    elif kind == 'system' and event.get('subtype') == 'api_error':
        error = event.get('error', {})
        if not isinstance(error, dict):
            return False
        status = error.get('status', error.get('status_code', status))
        text = str(error.get('message', ''))
    elif kind == 'system' and event.get('subtype') in {'task_updated', 'task_notification'}:
        details = event.get('patch', {}) if event['subtype'] == 'task_updated' else event
        if not isinstance(details, dict) or details.get('status') != 'failed':
            return False
        text = str(details.get('error', details.get('summary', '')))
        prefix = 'Agent terminated early due to an API error: '
        if not text.startswith(prefix):
            return False
        text = text[len(prefix):]
    elif kind == 'user' and isinstance(event.get('tool_use_result'), str):
        # Claude Code emits this metadata for failed child tasks, alongside
        # system task notifications. Ordinary user messages have no such field.
        prefix = 'Error: Agent terminated early due to an API error: '
        if not event['tool_use_result'].startswith(prefix):
            return False
        content = event.get('message', {}).get('content', [])
        if not isinstance(content, list) or not any(
            isinstance(block, dict) and block.get('type') == 'tool_result'
            and block.get('is_error') is True for block in content
        ):
            return False
        text = event['tool_use_result'][len(prefix):]
    else:
        return False
    return bool(
        (status == 400 or re.match(r'\s*API Error:\s*400\b', text))
        and re.search(r'\b(?:content_policy_violation|content_filter)\b', text)
    )


def cli_log_status(path):
    """Inspect every CLI event, including child errors before parent success."""
    terminal, refused = None, False
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        refused = refused or cli_event_content_refusal(event)
        if isinstance(event, dict) and event.get('type') == 'result':
            terminal = event
    return terminal, refused


def png_metadata(path):
    with Path(path).open('rb') as f:
        header = f.read(33)
    if len(header) != 33 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid PNG IHDR')
    width, height, bits, color = struct.unpack('>IIBB', header[16:26])
    result = {'format': 'PNG', 'width': width, 'height': height,
              'bits_per_channel': bits, 'png_color_type': color,
              'has_alpha_channel': color in (4, 6)}
    if color in (4, 6):
        try:
            from PIL import Image
            with Image.open(path) as image:
                # Pillow downconverts RGB(A)16 for display; alpha extrema here are
                # display-scale values, NOT a claim about native file bit depth.
                alpha = image.convert('RGBA').getchannel('A')
                lo, hi = alpha.getextrema()
                result['alpha_display_range_0_255'] = [lo, hi]
                result['has_visible_transparency'] = lo < 255
        except (ImportError, OSError, ValueError) as exc:
            result['alpha_probe_error'] = type(exc).__name__
    return result


def file_inventory(root):
    root = Path(root).resolve()
    rows = []
    for folder in ('exports', 'output'):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            row = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                   'provenance': 'author_submission', 'suffix': path.suffix.lower()}
            try:
                if path.suffix.lower() == '.png':
                    row.update(png_metadata(path))
                elif path.suffix.lower() in ('.jpg', '.jpeg', '.webp'):
                    from PIL import Image
                    with Image.open(path) as im:
                        row.update(format=im.format, width=im.width, height=im.height)
            except (ImportError, OSError, ValueError) as exc:
                row['metadata_error'] = type(exc).__name__
            rows.append(row)
    return {'root': str(root), 'provenance': 'original files materialized from workspace_snapshot',
            'file_count': len(rows), 'files': rows,
            'note': 'Includes auxiliary textures and previews. Presence alone does not identify the final deliverable.'}


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points, points[1:]+points[:1]))) * .5


def triangle_intersection_area(subject, triangle):
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if cross(*triangle) < 0:
        triangle = triangle[::-1]
    output = list(subject)
    for a,b in zip(triangle, triangle[1:]+triangle[:1]):
        source, output = output, []
        if not source:
            break
        prev = source[-1]; dp = cross(a,b,prev)
        for cur in source:
            dc = cross(a,b,cur)
            if (dp >= 0) != (dc >= 0):
                t = dp/(dp-dc)
                output.append((prev[0]+t*(cur[0]-prev[0]), prev[1]+t*(cur[1]-prev[1])))
            if dc >= 0:
                output.append(cur)
            prev, dp = cur, dc
    return polygon_area(output)


def find_uv_overlap(triangles, max_pairs=2_000_000):
    """Positive-area overlap within one mesh; shared edges do not count.

    A bounded sweep returns incomplete/unknown when its budget is exhausted;
    an incomplete scan must never be used to assert no overlap.
    """
    boxes = []
    degenerate = 0
    for i,t in enumerate(triangles):
        if polygon_area(t) <= 1e-12:
            degenerate += 1
            continue
        boxes.append((min(p[0] for p in t), max(p[0] for p in t),
                      min(p[1] for p in t), max(p[1] for p in t), i))
    boxes.sort()
    active = []; checked = 0
    for box in boxes:
        xmin,xmax,ymin,ymax,i = box
        active = [b for b in active if b[1] > xmin]
        for b in active:
            checked += 1
            if checked > max_pairs:
                return {'complete': False, 'overlap': None, 'pairs_checked': checked-1,
                        'reason': 'candidate pair budget exceeded', 'degenerate_triangles': degenerate}
            if b[3] <= ymin or ymax <= b[2]:
                continue
            area = triangle_intersection_area(triangles[b[4]], triangles[i])
            if area > 1e-10:
                return {'complete': True, 'overlap': True, 'triangle_indices': [b[4],i],
                        'intersection_area': area, 'pairs_checked': checked,
                        'degenerate_triangles': degenerate}
        active.append(box)
    return {'complete': True, 'overlap': False, 'pairs_checked': checked,
            'degenerate_triangles': degenerate}


def scene_facts(uv_layer=None):
    import bpy
    from mathutils import Vector
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    result = {'blender_version': bpy.app.version_string, 'blend_file': bpy.data.filepath,
              'engine': scene.render.engine, 'effective_fps': fps, 'cameras': []}
    world = scene.rigidbody_world
    if world:
        result['rigid_body_world'] = {
            'substeps_per_frame': world.substeps_per_frame,
            'steps_per_second_at_scene_fps': world.substeps_per_frame * fps,
            'solver_iterations': world.solver_iterations, 'time_scale': world.time_scale}
    for camera in (o for o in scene.objects if o.type == 'CAMERA'):
        direction = camera.matrix_world.to_quaternion() @ Vector((0,0,-1))
        result['cameras'].append({'name': camera.name, 'active': camera == scene.camera,
            'world_z': camera.matrix_world.translation.z,
            'look_direction_world': list(direction),
            'down_pitch_from_horizontal_degrees': math.degrees(math.atan2(-direction.z, math.hypot(direction.x,direction.y))),
            'lens_mm': camera.data.lens, 'dof_enabled': camera.data.dof.use_dof,
            'fstop': camera.data.dof.aperture_fstop})
    if uv_layer:
        result['uv_checks'] = []
        for obj in scene.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            row = {'object': obj.name, 'collections': [c.name for c in obj.users_collection],
                   'hide_render': obj.hide_render, 'layer': uv_layer}
            uv = mesh.uv_layers.get(uv_layer)
            if uv is None:
                row.update(layer_present=False)
            else:
                mesh.calc_loop_triangles()
                triangles = [[tuple(uv.data[i].uv) for i in t.loops] for t in mesh.loop_triangles]
                row.update(layer_present=True, triangle_count=len(triangles), **find_uv_overlap(triangles))
            result['uv_checks'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--uv-layer')
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    data = file_inventory(args.inventory_root) if args.inventory_root else scene_facts(args.uv_layer)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
