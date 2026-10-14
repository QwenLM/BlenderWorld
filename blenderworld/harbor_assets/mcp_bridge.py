#!/usr/local/bin/python
"""Own a Blender session and expose its MCP server over clean stdio."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    env = os.environ.copy()
    env.setdefault("PATH", "/opt/blender:/usr/local/bin:/usr/bin:/bin")
    env["BLENDERMCP_NO_UPDATE_CHECK"] = "1"
    env["BLENDER_MCP_DISABLE_TELEMETRY"] = "1"
    env.setdefault("BLENDER_MCP_ADDON_PATH", "/opt/bw-mcp/addon.py")
    # Separate ports allow multiple MCP clients without borrowing another client's scene.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env.update(BLENDER_HOST="127.0.0.1", BLENDER_PORT=str(port))
    log = Path(env.get("BW_BLENDER_LOG", "/logs/artifacts/blender-mcp.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    blender = None
    server = None

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        with log.open("a") as stream:
            blender = subprocess.Popen(
                ["xvfb-run", "-a", env.get("BLENDER_BIN", "/opt/blender/blender"),
                 "--factory-startup", "--python-exit-code", "1", "--python",
                 str(Path(__file__).resolve().with_name("blender_startup.py"))],
                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + float(env.get("BW_MCP_STARTUP_TIMEOUT", "120"))
            while True:
                if blender.poll() is not None:
                    raise RuntimeError(f"Blender exited during startup; see {log}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Blender MCP startup timed out; see {log}")
                    time.sleep(0.2)
            # stdin stays connected to the MCP client. Only this process writes to stdout.
            server = subprocess.Popen([env.get("BW_MCP_COMMAND", "/opt/bw-mcp/bin/mcp-for-blender")],
                                      env=env, stdin=sys.stdin, stdout=sys.stdout,
                                      stderr=sys.stderr, start_new_session=True)
            return server.wait()
    finally:
        for process in (server, blender):
            if process is None:
                continue
            # xvfb-run may exit before its descendants; terminate the owned process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"BlenderWorld MCP: {error}", file=sys.stderr)
        raise SystemExit(1)
