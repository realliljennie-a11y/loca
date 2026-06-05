#!/usr/bin/env python3
"""
loca-proxy.py — Fan-controlling reverse proxy in front of Ollama.

Listens on 0.0.0.0:11434, forwards every request to 127.0.0.1:11435.
Spins the AMD GPU fan to 80% when any request arrives; returns to auto
when no requests are in flight.
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

from pathlib import Path
import aiohttp
from aiohttp import web

# ── Configuration ─────────────────────────────────────────────────────────────

LISTEN_HOST  = "0.0.0.0"
LISTEN_PORT  = 11434
UPSTREAM_URL = "http://127.0.0.1:11435"

# ── AMD GPU fan control ───────────────────────────────────────────────────────

_AMDGPU_HWMON: Path | None = None

def _find_amdgpu_hwmon() -> Path | None:
    global _AMDGPU_HWMON
    if _AMDGPU_HWMON is not None:
        return _AMDGPU_HWMON
    try:
        for hwmon in Path('/sys/class/hwmon').iterdir():
            if (hwmon / 'name').read_text().strip() == 'amdgpu':
                _AMDGPU_HWMON = hwmon
                return hwmon
    except OSError:
        pass
    return None

def _gpu_fan_set(pct: int = 80) -> None:
    hwmon = _find_amdgpu_hwmon()
    if not hwmon:
        return
    try:
        (hwmon / 'pwm1_enable').write_text('1')
        (hwmon / 'pwm1').write_text(str(int(pct / 100 * 255)))
    except OSError:
        pass

def _gpu_fan_auto() -> None:
    hwmon = _find_amdgpu_hwmon()
    if not hwmon:
        return
    try:
        (hwmon / 'pwm1_enable').write_text('2')
    except OSError:
        pass

# ── Fan reference counter ─────────────────────────────────────────────────────
# asyncio is single-threaded so no lock needed; the counter is modified only
# from the event loop.

_active_requests: int = 0

def _fan_acquire() -> None:
    global _active_requests
    _active_requests += 1
    if _active_requests == 1:
        _gpu_fan_set(80)

def _fan_release() -> None:
    global _active_requests
    _active_requests -= 1
    if _active_requests == 0:
        _gpu_fan_auto()

# ── Proxy handler ─────────────────────────────────────────────────────────────

_HOP_BY_HOP = frozenset({
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailers', 'transfer-encoding', 'upgrade',
})

async def _proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    forward_headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in _HOP_BY_HOP and k.lower() != 'host'}

    _fan_acquire()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=f"{UPSTREAM_URL}{request.path_qs}",
                headers=forward_headers,
                data=body,
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(status=upstream.status)
                for k, v in upstream.headers.items():
                    if k.lower() not in _HOP_BY_HOP:
                        response.headers[k] = v
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
    finally:
        _fan_release()

    return response

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = web.Application()
    app.router.add_route('*', '/{path_info:.*}', _proxy)
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, print=None)

if __name__ == '__main__':
    main()
