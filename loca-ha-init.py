#!/usr/bin/env python3
"""
loca-ha-init.py — Push sunny_online = off to HA at boot.

Called by loca-watcher.service ExecStartPre so HA starts with a known
offline state before loca-talk has a chance to declare itself online.
Exits 0 always — failure must never prevent loca-watcher from starting.
"""
import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.realpath(_sys.argv[0]))
_venv = (_os.environ.get("LOCA_VENV")
         or (_os.path.join(_here, ".venv") if _os.path.isdir(_os.path.join(_here, ".venv")) else None)
         or _os.path.expanduser("~/hf-venv"))
_vpython = _os.path.join(_venv, "bin", "python3")
if _os.path.exists(_vpython) and _os.path.normpath(_sys.prefix) != _os.path.normpath(_venv):
    _os.execv(_vpython, [_vpython] + _sys.argv)
del _sys, _os, _here, _venv, _vpython

import re
import sys

try:
    import httpx
    from config import CFG

    ha_url   = CFG.get("ha_url", "")
    ha_token = CFG.get("ha_token", "")
    entity   = CFG.get("triggers", {}).get("online_entity", "binary_sensor.sunny_online")
    name     = CFG.get("llm", {}).get("assistant_name", "Assistant")

    if ha_url and ha_token and entity:
        ha_base = re.sub(r"/api/.*$", "", ha_url)
        httpx.post(
            f"{ha_base}/api/states/{entity}",
            json={"state": "off", "attributes": {"friendly_name": f"{name} Online"}},
            headers={"Authorization": f"Bearer {ha_token}"},
            timeout=5.0,
        )
except Exception:
    pass

sys.exit(0)
