"""Run a Python file inside the Unreal Editor the user has open (live edits, no commandlet).

Needs Edit > Project Settings > Plugins > Python > Enable Remote Execution. Uses the
remote_execution.py client that ships with the engine's Python plugin. The editor keeps
the changes unsaved; saving is left to the user. The code is sent as a temporary .py
file: sent as text, the editor treats everything up to the first ".py" in it as a file
name (PythonScriptPlugin's TryExtractPathnameAndCommand), so a message mentioning
"gtb.py" broke a whole script.

    python -m ue.remote script.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ue.pipeline import find_unreal  # noqa: E402


def run(code, cfg=None):
    """Execute code in the first editor found; returns its printed output."""
    import os
    import tempfile
    engine = find_unreal(cfg or {}).parents[3]      # .../Engine/Binaries/Win64/x.exe -> UE_5.x
    sys.path.insert(0, str(engine / "Engine" / "Plugins" / "Experimental" / "PythonScriptPlugin"
                           / "Content" / "Python"))
    import remote_execution as rx_mod
    rx = rx_mod.RemoteExecution()
    rx.start()
    try:
        for _ in range(50):
            if rx.remote_nodes:
                break
            time.sleep(0.1)
        if not rx.remote_nodes:
            raise RuntimeError("no Unreal Editor found (is Python Remote Execution enabled?)")
        rx.open_command_connection(rx.remote_nodes[0]["node_id"])
        fd, script = tempfile.mkstemp(suffix=".py", prefix="gtb_remote_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            r = rx.run_command(f'"{Path(script).as_posix()}"', unattended=True, exec_mode=rx_mod.MODE_EXEC_FILE)
        finally:
            os.unlink(script)
        out = "".join(o["output"] if o["output"].endswith("\n") else o["output"] + "\n"
                      for o in r.get("output", []))
        if not r.get("success"):
            raise RuntimeError(f"{out}{r.get('result')}")
        return out
    finally:
        rx.stop()


if __name__ == "__main__":
    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    print(run(Path(sys.argv[1]).read_text(encoding="utf-8"), cfg), end="")
