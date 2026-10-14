"""Start the bundled, versioned MCP addon in an interactive Blender session."""
import importlib.util
import os
import sys

import bpy

if bpy.app.version != (4, 2, 10):
    raise RuntimeError("BlenderWorld requires Blender 4.2.10")

os.environ["BLENDERMCP_NO_UPDATE_CHECK"] = "1"
path = os.environ.get("BLENDER_MCP_ADDON_PATH", "/opt/bw-mcp/addon.py")
spec = importlib.util.spec_from_file_location("blendermcp", path)
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
addon.register()
bpy.context.scene.blendermcp_port = int(os.environ["BLENDER_PORT"])
bpy.context.scene.blendermcp_auto_start_server = True
bpy.context.scene.blendermcp_use_polyhaven = True
print("BlenderWorld MCP addon registered", flush=True)
