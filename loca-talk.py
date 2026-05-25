#!/usr/bin/env python3
"""
loca-talk.py — Voice conversation pipeline over Classic BT HFP.

Requires:
  - Device connected in HFP mode: loca-watcher handles this automatically
  - Ollama running: systemd service
  - pip install faster-whisper httpx pyyaml

Usage:
    ./loca-talk.py                # full voice loop
    ./loca-talk.py --type         # type input instead of speaking
    ./loca-talk.py --silent       # print output instead of using TTS
    ./loca-talk.py --greet        # have the assistant say hello on startup
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

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import asyncio
import json
import math
import random
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import httpx
from faster_whisper import WhisperModel
from kokoro import KPipeline
from config import CFG

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

# ── Configuration ─────────────────────────────────────────────────────────────

_audio_in       = CFG.get("audio", {}).get("input", {})
_audio_out      = CFG.get("audio", {}).get("output", {})
AUDIO_SOURCE    = _audio_in.get("source", "")
AUDIO_SINK      = _audio_out.get("sink", "")
BT_MAC          = (_audio_out.get("bt_mac") or _audio_in.get("bt_mac", ""))
MIC_RATE        = 16000
MIC_CHANNELS    = 1

DEFAULT_VOICE   = CFG["tts"]["voice"]
SPEED           = CFG["tts"]["speed"]

_ollama_llm_cfg = CFG.get("ollama", {})
OLLAMA_URL      = _ollama_llm_cfg.get("url", "http://localhost:11434/api/chat")
OLLAMA_MODEL    = _ollama_llm_cfg.get("model", "")
OLLAMA_OPTIONS  = _ollama_llm_cfg.get("options", {})
DEFAULT_LLM     = CFG.get("llm", {}).get("provider", "ollama")

_llm_cfg                = CFG.get("llm", {})
ASSISTANT_NAME          = _llm_cfg.get("assistant_name", "Assistant")
USER_NAME               = _llm_cfg.get("user_name", "User")
_user_pro               = _llm_cfg.get("user_pronouns", "they/them/their").split("/")
USER_PRONOUN_SUBJ       = _user_pro[0] if len(_user_pro) > 0 else "they"
USER_PRONOUN_OBJ        = _user_pro[1] if len(_user_pro) > 1 else "them"
USER_PRONOUN_POSS       = _user_pro[2] if len(_user_pro) > 2 else "their"
_asst_pro               = _llm_cfg.get("assistant_pronouns", "they/them/their").split("/")
ASST_PRONOUN_SUBJ       = _asst_pro[0] if len(_asst_pro) > 0 else "they"
ASST_PRONOUN_OBJ        = _asst_pro[1] if len(_asst_pro) > 1 else "them"
ASST_PRONOUN_POSS       = _asst_pro[2] if len(_asst_pro) > 2 else "their"
def _substitute(text: str) -> str:
    """Substitute persona variables in config strings.

    Supported placeholders: {assistant_name}, {user_name},
    {user_pronoun_subj/obj/poss}, {assistant_pronoun_subj/obj/poss}.
    Unknown placeholders are left unchanged.
    """
    if not text:
        return text
    class _PassThrough(dict):
        def __missing__(self, key): return "{" + key + "}"
    return text.format_map(_PassThrough({
        "assistant_name":       ASSISTANT_NAME,
        "user_name":            USER_NAME,
        "user_pronoun_subj":    USER_PRONOUN_SUBJ,
        "user_pronoun_obj":     USER_PRONOUN_OBJ,
        "user_pronoun_poss":    USER_PRONOUN_POSS,
        "assistant_pronoun_subj": ASST_PRONOUN_SUBJ,
        "assistant_pronoun_obj":  ASST_PRONOUN_OBJ,
        "assistant_pronoun_poss": ASST_PRONOUN_POSS,
    }))

MEMORY_HEADER           = _substitute(_llm_cfg.get("memory_header",
                              "What {assistant_name} remembers about {user_name}:"))
CHARACTER_REINFORCEMENT = _substitute(_llm_cfg.get("character_reinforcement", "").strip())
_REFUSAL_REDIRECTS      = [_substitute(s) for s in _llm_cfg.get("refusal_redirects", [
                              "Let's talk about something else!",
                              "How about we change the subject?",
                              "I'd rather not go there — what else is on your mind?",
                              "Let's try a different topic.",
                          ])]

CLAUDE_MODEL         = CFG.get("claude", {}).get("model", "claude-haiku-4-5")
CLAUDE_SYSTEM_PROMPT = _substitute(CFG.get("claude", {}).get("system_prompt") or "")  or None

OPENROUTER_URL          = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL        = CFG.get("openrouter", {}).get("model", "gryphe/mythomax-l2-13b")
OPENROUTER_SYSTEM_PROMPT = _substitute(CFG.get("openrouter", {}).get("system_prompt") or "") or None

WHISPER_MODEL        = CFG["stt"]["whisper_model"]
WHISPER_DEVICE       = CFG["stt"]["whisper_device"]
WHISPER_COMPUTE      = CFG["stt"]["whisper_compute"]
NO_SPEECH_THRESHOLD  = CFG["stt"].get("no_speech_threshold", 0.6)
LOGPROB_THRESHOLD    = CFG["stt"].get("logprob_threshold", -1.0)
_ww_cfg              = CFG["stt"].get("wake_word_filter", {})
WAKE_WORD_THRESHOLD  = _ww_cfg.get("threshold", 0.85)
WAKE_WORD_VARIANTS   = [re.sub(r"[^a-z0-9]", "", v.lower())
                        for v in _ww_cfg.get("variants", [])]

SILENCE_SECONDS = CFG["vad"]["silence_seconds"]
MIN_SPEECH_SECS = CFG["vad"]["min_speech_secs"]
MAX_RECORD_SECS = CFG["vad"]["max_record_secs"]
NOISE_HEADROOM  = CFG["vad"]["noise_headroom"]

HA_URL          = CFG["ha_url"]
HA_TOKEN        = CFG["ha_token"]

_ollama_cfg     = CFG.get("ollama", {})
GREETING        = _substitute(_ollama_cfg.get("greeting") or CFG.get("greeting", ""))
SYSTEM_PROMPT   = _substitute(_ollama_cfg.get("system_prompt") or CFG.get("system_prompt", ""))

TRIGGERS        = CFG.get("triggers", {})
OVOS_ENTITY     = TRIGGERS.get("ovos_speaking_entity", "")
AWAKE_ENTITY    = TRIGGERS.get("awake_entity", "")
SLEEP_PHRASES   = [p.lower().strip() for p in TRIGGERS.get("sleep_phrases",
                   ["go to sleep", "goodnight", "sleep time"])]
WAKE_PHRASES    = [p.lower().strip() for p in TRIGGERS.get("wake_phrases",
                   ["wake up", "good morning", "rise and shine"])]
SLEEP_RESPONSE  = _substitute(TRIGGERS.get("sleep_response") or
                               "I'm going to sleep now.")
WAKE_RESPONSE   = _substitute(TRIGGERS.get("wake_response") or GREETING)

MAX_RECENT_TURNS = CFG.get("memory", {}).get("max_recent_turns", 20)
COMPACT_EVERY    = CFG.get("memory", {}).get("compact_every", 10)

MEMORY_DIR        = Path(__file__).parent / "memory"
LONG_TERM_FILE    = MEMORY_DIR / "long_term.txt"
RECENT_TURNS_FILE = MEMORY_DIR / "recent_turns.json"

# ── Shutdown flag ─────────────────────────────────────────────────────────────

_shutdown_requested = False

# ── OVOS speaking gate ────────────────────────────────────────────────────────

_ovos_speaking: bool = False

# ── Sleep mode flag ───────────────────────────────────────────────────────────

_awake: bool = True

# ── Speaking gate (prevents mic from hearing Sunny's own voice) ───────────────

_sunny_speaking: bool = False

def _handle_signal(signum, frame):
    """Handle SIGTERM from systemd — set flag, let the loop exit cleanly."""
    global _shutdown_requested
    print("Shutdown requested.", end="", flush=True)
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_signal)

# ── GPU fan control ───────────────────────────────────────────────────────────

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

# ── Memory ────────────────────────────────────────────────────────────────────

def load_memory() -> tuple[list[dict], str]:
    """Load recent turns and long-term memory from disk."""
    history = []
    if RECENT_TURNS_FILE.exists():
        try:
            history = json.loads(RECENT_TURNS_FILE.read_text())
        except Exception:
            pass
    long_term = ""
    if LONG_TERM_FILE.exists():
        try:
            long_term = LONG_TERM_FILE.read_text().strip()
        except Exception:
            pass
    return history, long_term

def save_recent_turns(history: list[dict]) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    trimmed = history[-MAX_RECENT_TURNS:]
    RECENT_TURNS_FILE.write_text(json.dumps(trimmed, indent=2))

async def compact_memory(history: list[dict],
                          long_term: str,
                          http: httpx.AsyncClient,
                          verbose: bool = False,
                          claude_client=None,
                          openrouter_cfg: dict | None = None) -> str:
    """Summarise recent conversation into long_term.txt. Returns new memory text."""
    conversation = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in history[-20:])
    memory_prompt = (
        (f"You are {ASSISTANT_NAME}, and these are your existing notes about {USER_NAME}:\n{long_term}\n\n" if long_term else "")
        + f"Recent conversation:\n{conversation}\n\n"
        f"As {ASSISTANT_NAME}, write 2-4 sentences of updated memory notes about {USER_NAME}. "
        f"Include what matters to {USER_PRONOUN_OBJ}, things {USER_PRONOUN_SUBJ} mentioned, and patterns you noticed. "
        f"Refer to {USER_PRONOUN_OBJ} as '{USER_NAME}' and use {USER_PRONOUN_POSS} correct pronouns. "
        f"Be concise. Output only the notes, nothing else."
    )
    try:
        if claude_client is not None:
            base_prompt = CLAUDE_SYSTEM_PROMPT or SYSTEM_PROMPT
            resp = await claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=[{"type": "text", "text": base_prompt,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": memory_prompt}],
            )
            new_memory = resp.content[0].text.strip()
        elif openrouter_cfg is not None:
            base_prompt = openrouter_cfg.get("system_prompt") or SYSTEM_PROMPT
            headers = {
                "Authorization": f"Bearer {openrouter_cfg['api_key']}",
                "X-Title": "loca-talk",
            }
            payload = {
                "model":    openrouter_cfg["model"],
                "messages": [{"role": "system", "content": base_prompt},
                             {"role": "user",   "content": memory_prompt}],
                "stream":   False,
            }
            resp = await http.post(OPENROUTER_URL, json=payload,
                                   headers=headers, timeout=60.0)
            new_memory = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            payload = {
                "model":   OLLAMA_MODEL,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user",   "content": memory_prompt}],
                "stream":  False,
                "options": {"num_predict": 300},
            }
            _gpu_fan_set()
            try:
                resp = await http.post(OLLAMA_URL, json=payload, timeout=60.0)
                new_memory = resp.json()["message"]["content"].strip()
            finally:
                _gpu_fan_auto()
        MEMORY_DIR.mkdir(exist_ok=True)
        LONG_TERM_FILE.write_text(new_memory)
        if verbose:
            print(f"  [memory: {new_memory[:80]}...]")
        return new_memory
    except Exception as e:
        print(f"  [memory compact failed: {type(e).__name__}: {e}]")
        return long_term

# ── Connection watchdog ───────────────────────────────────────────────────────

def device_is_connected() -> bool:
    """Check if device's HFP audio sink is present."""
    result = subprocess.run(
        ["pactl", "list", "sinks", "short"],
        capture_output=True, text=True)
    mac_underscored = BT_MAC.replace(":", "_")
    return mac_underscored in result.stdout

async def wait_for_device(timeout: float = 300.0) -> bool:
    """Wait for device's audio sink to appear. Returns True when found."""
    print("Waiting for device to connect...", end="", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _shutdown_requested:
            return False
        if device_is_connected():
            print(" connected.")
            return True
        print(".", end="", flush=True)
        await asyncio.sleep(5)
    print(" timed out.")
    return False

# ── Text processing ───────────────────────────────────────────────────────────

def sanitise(text: str) -> str:
    """Strip markup and stage directions before TTS."""
    text = re.sub(r'\*[^*]*\*', '', text)
    text = re.sub(r'_[^_]*_', '', text)
    text = re.sub(r'\[/?INST\]', '', text)
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'^[A-Za-z\s]+:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_to_complete_sentences(text: str) -> str:
    """Strip any incomplete sentence from the end of the text."""
    match = re.search(r'[.!?][^.!?]*$', text.strip())
    if not match:
        return text
    last_punct = text.rfind(match.group()[0], 0, match.start() + 1)
    return text[:last_punct + 1].strip()

# ── Noise floor calibration ───────────────────────────────────────────────────

def calibrate_noise_floor(source: str = AUDIO_SOURCE,
                           calibration_secs: float = 2.0) -> float:
    """Measure ambient noise floor and return a suitable speech threshold."""
    chunk_bytes   = int(MIC_RATE * 0.05) * 2
    chunks_needed = int(calibration_secs / 0.05)

    proc = subprocess.Popen(
        ["pw-record",
         f"--target={source}",
         "--channels=1",
         f"--rate={MIC_RATE}",
         "--format=s16",
         "--latency=50",
         "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    rms_values = []
    try:
        for _ in range(chunks_needed):
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            rms_values.append(rms)
    finally:
        proc.terminate()
        proc.wait()

    if not rms_values:
        return 1000
    rms_values.sort()
    noise_floor = rms_values[len(rms_values) // 4]  # 25th percentile
    threshold   = noise_floor + NOISE_HEADROOM
    print(f"  (noise floor: {noise_floor:.0f}, threshold: {threshold:.0f})")
    return threshold

# ── Home Assistant helpers ────────────────────────────────────────────────────

def lux_to_band(lx: float) -> str:
    if lx < 10:   return "very dark"
    if lx < 30:   return "dim"
    if lx < 100:  return "moderately lit"
    return "bright"

def illum_transition_text(from_band: str, to_band: str) -> str | None:
    if to_band == "very dark":
        return "The nursery just went dark."
    if from_band == "very dark" and to_band in ("moderately lit", "bright"):
        return "The nursery lights just came on."
    return None  # ignore minor transitions (dim↔moderately lit, etc.)

async def sensor_watcher(http: httpx.AsyncClient,
                          event_queue: asyncio.Queue,
                          verbose: bool = False) -> None:
    """Background task: subscribe to HA WebSocket state changes for sensors and illuminance."""
    import websockets

    sensor_cfgs  = TRIGGERS.get("sensors", [])
    illum_entity = TRIGGERS.get("illuminance_entity")

    entities = list({s["entity"] for s in sensor_cfgs})
    if illum_entity:
        entities.append(illum_entity)
    if not entities:
        return

    # entity → list of (watch_state, event_text) pairs
    sensor_map: dict[str, list[tuple[str, str]]] = {}
    for s in sensor_cfgs:
        sensor_map.setdefault(s["entity"], []).append(
            (s["watch_state"].lower(), s["event_text"]))

    ha_base = re.sub(r"/api/.*$", "", HA_URL)
    ws_url  = ha_base.replace("https://", "wss://").replace("http://", "ws://") \
              + "/api/websocket"

    # Fetch initial illuminance band so first transition has a baseline.
    last_illum_band: str | None = None
    if illum_entity:
        try:
            r = await http.get(f"{HA_URL}/{illum_entity}",
                               headers={"Authorization": f"Bearer {HA_TOKEN}"}, timeout=3.0)
            if r.status_code == 200:
                last_illum_band = lux_to_band(float(r.json().get("state", 0)))
        except Exception:
            pass

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_required":
                    raise RuntimeError(f"unexpected HA WS message: {msg}")
                await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_ok":
                    raise RuntimeError("HA WebSocket authentication failed")

                await ws.send(json.dumps({
                    "id": 1,
                    "type": "subscribe_trigger",
                    "trigger": {"platform": "state", "entity_id": entities},
                }))
                await ws.recv()  # result confirmation

                if verbose:
                    print(f"  [sensor_watcher] subscribed to {entities}")

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "event":
                        continue
                    trigger    = (msg.get("event", {})
                                     .get("variables", {})
                                     .get("trigger", {}))
                    entity_id  = trigger.get("entity_id", "")
                    from_state = trigger.get("from_state", {}).get("state", "").lower()
                    to_state   = trigger.get("to_state",   {}).get("state", "").lower()

                    if verbose:
                        print(f"  [sensor_watcher] {entity_id}: {from_state!r} → {to_state!r}")

                    if entity_id == illum_entity:
                        try:
                            band = lux_to_band(float(to_state))
                        except ValueError:
                            continue
                        if last_illum_band is not None and band != last_illum_band:
                            text = illum_transition_text(last_illum_band, band)
                            if text:
                                await event_queue.put(text)
                        last_illum_band = band
                        continue

                    for watch_state, event_text in sensor_map.get(entity_id, []):
                        if from_state not in ("unavailable", "unknown") \
                                and from_state != to_state \
                                and to_state == watch_state:
                            if verbose:
                                print(f"  [sensor_watcher] queuing: {event_text!r}")
                            await event_queue.put(event_text)

        except Exception as e:
            if verbose:
                print(f"  [sensor_watcher] {type(e).__name__}: {e} — reconnecting in 5s")
            await asyncio.sleep(5)


async def bedtime_watcher(event_queue: asyncio.Queue) -> None:
    """Background task: queue a bedtime event when the clock reaches bedtime_hour."""
    bedtime_hour = TRIGGERS.get("bedtime_hour")
    if bedtime_hour is None:
        return
    last_hour = time.localtime().tm_hour
    while True:
        await asyncio.sleep(30)
        hour = time.localtime().tm_hour
        if last_hour != hour and hour == bedtime_hour:
            await event_queue.put(f"It just turned {bedtime_hour}:00 — bedtime.")
        last_hour = hour

async def ovos_watcher(verbose: bool = False) -> None:
    """Background task: subscribe to HA WebSocket state changes for OVOS speaking."""
    global _ovos_speaking
    import websockets

    ha_base = re.sub(r"/api/.*$", "", HA_URL)
    ws_url  = ha_base.replace("https://", "wss://").replace("http://", "ws://") \
              + "/api/websocket"

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_required":
                    raise RuntimeError(f"unexpected HA WS message: {msg}")
                await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_ok":
                    raise RuntimeError("HA WebSocket authentication failed")

                await ws.send(json.dumps({
                    "id": 1,
                    "type": "subscribe_trigger",
                    "trigger": {"platform": "state", "entity_id": OVOS_ENTITY},
                }))
                await ws.recv()  # result confirmation

                if verbose:
                    print(f"  [ovos_watcher] subscribed to {OVOS_ENTITY}")

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "event":
                        to_state = (msg.get("event", {})
                                       .get("variables", {})
                                       .get("trigger", {})
                                       .get("to_state", {}))
                        state = to_state.get("state", "").lower()
                        _ovos_speaking = state in ("on", "true")
                        if verbose:
                            print(f"  [ovos_watcher] OVOS speaking: {_ovos_speaking}")

        except Exception as e:
            if verbose:
                print(f"  [ovos_watcher] {type(e).__name__}: {e} — reconnecting in 5s")
            _ovos_speaking = False
            await asyncio.sleep(5)

async def awake_watcher(verbose: bool = False, silent: bool = False) -> None:
    """Background task: subscribe to HA WebSocket state changes for sleep mode entity."""
    global _awake, _sunny_speaking
    import websockets

    ha_base = re.sub(r"/api/.*$", "", HA_URL)
    ws_url  = ha_base.replace("https://", "wss://").replace("http://", "ws://") \
              + "/api/websocket"

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_required":
                    raise RuntimeError(f"unexpected HA WS message: {msg}")
                await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_ok":
                    raise RuntimeError("HA WebSocket authentication failed")

                await ws.send(json.dumps({
                    "id": 1,
                    "type": "subscribe_trigger",
                    "trigger": {"platform": "state", "entity_id": AWAKE_ENTITY},
                }))
                await ws.recv()  # result confirmation

                if verbose:
                    print(f"  [awake_watcher] subscribed to {AWAKE_ENTITY}")

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "event":
                        continue
                    to_state = (msg.get("event", {})
                                   .get("variables", {})
                                   .get("trigger", {})
                                   .get("to_state", {}))
                    state       = to_state.get("state", "").lower()
                    new_awake = state in ("on", "true")
                    if new_awake == _awake:
                        continue
                    _awake = new_awake
                    if verbose:
                        print(f"  [awake_watcher] awake: {_awake}")
                    if not silent:
                        response = WAKE_RESPONSE if _awake else SLEEP_RESPONSE
                        _sunny_speaking = True  # set now, before executor starts
                        try:
                            print(f"{ASSISTANT_NAME} (non-LLM): {response}")
                            await asyncio.get_running_loop().run_in_executor(
                                None, lambda r=response: speak(
                                    r, voice=DEFAULT_VOICE, speed=SPEED, sink=AUDIO_SINK))
                        except Exception:
                            _sunny_speaking = False

        except Exception as e:
            if verbose:
                print(f"  [awake_watcher] {type(e).__name__}: {e} — reconnecting in 5s")
            await asyncio.sleep(5)

# ── Home Assistant context ────────────────────────────────────────────────────

async def get_ha_context(http: httpx.AsyncClient,
                          verbose: bool = False) -> str:
    """Fetch nursery state from Home Assistant."""
    lines = []

    now  = time.localtime()
    lines.append(f"The time is {now.tm_hour:02d}:{now.tm_min:02d}.")

    illum_entity = TRIGGERS.get("illuminance_entity", "")
    if HA_URL and HA_TOKEN and illum_entity:
        try:
            headers = {"Authorization": f"Bearer {HA_TOKEN}"}
            r = await http.get(
                f"{HA_URL}/{illum_entity}",
                headers=headers, timeout=3.0)
            if r.status_code == 200:
                lx = float(r.json().get("state", 0))
                lines.append(f"The nursery is {lux_to_band(lx)} ({lx:.0f} lx).")
        except Exception:
            pass

    if verbose:
        print("\n".join(lines))
    return "\n".join(lines)

# ── TTS pipeline (module-level singleton) ─────────────────────────────────────

_tts_pipeline: KPipeline | None = None

def init_tts() -> None:
    global _tts_pipeline
    _tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

def _get_tts() -> KPipeline:
    global _tts_pipeline
    if _tts_pipeline is None:
        _tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _tts_pipeline

# ── TTS → speaker ─────────────────────────────────────────────────────────────

# Kokoro mispronounces certain words; substitute phonetic spellings before synthesis.
_PRONUNCIATION_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bdiapees\b', re.IGNORECASE), 'dypees'),
    (re.compile(r'\bdiapee\b',  re.IGNORECASE), 'dypee'),
    (re.compile(r'\bdiapers\b', re.IGNORECASE), 'dypers'),
    (re.compile(r'\bdiaper\b',  re.IGNORECASE), 'dyper'),
    (re.compile(r'\bdipees\b',  re.IGNORECASE), 'dypees'),
    (re.compile(r'\bdipee\b',   re.IGNORECASE), 'dypee'),
    (re.compile(r'\bdiapies\b', re.IGNORECASE), 'dypees'),
    (re.compile(r'\bdiapie\b',  re.IGNORECASE), 'dypee'),
    (re.compile(r'\bmybaby\b',          re.IGNORECASE), 'my baby'),
    (re.compile(r'\bmysweetlittleone\b', re.IGNORECASE), 'my sweet little one'),
]

def _apply_pronunciation_fixes(text: str) -> str:
    for pattern, replacement in _PRONUNCIATION_FIXES:
        def _replace(m, r=replacement):
            return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
        text = pattern.sub(_replace, text)
    return text


def speak(text: str,
          voice: str = DEFAULT_VOICE,
          speed: float = SPEED,
          sink: str = AUDIO_SINK) -> None:
    """Render text with Kokoro and play directly to PipeWire sink."""
    global _sunny_speaking
    text = _apply_pronunciation_fixes(text)
    _sunny_speaking = True
    try:
        pipeline = _get_tts()
        chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
        if not chunks:
            return
        pcm_float = np.concatenate(chunks)
        pcm_int16 = (np.clip(pcm_float, -1.0, 1.0) * 32767).astype(np.int16)
        # Prepend silence to avoid HFP transport clipping the start
        silence = np.zeros(int(24000 * 0.15), dtype=np.int16)
        pcm_bytes = np.concatenate([silence, pcm_int16]).tobytes()
        pw = subprocess.Popen(
            ["pw-play",
             f"--target={sink}",
             "--channels=1",
             "--rate=24000",
             "--format=s16",
             "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pw.communicate(input=pcm_bytes)
    finally:
        _sunny_speaking = False

# ── Microphone recording with VAD ─────────────────────────────────────────────

def record_utterance(source: str = AUDIO_SOURCE,
                     threshold: float = 1000.0,
                     verbose: bool = False) -> bytes | None:
    """Record from source until silence is detected. Returns raw PCM."""
    chunk_secs   = 0.05
    chunk_frames = int(MIC_RATE * chunk_secs)
    chunk_bytes  = chunk_frames * MIC_CHANNELS * 2
    silence_chunks_needed = int(SILENCE_SECONDS / chunk_secs)
    max_chunks   = int(MAX_RECORD_SECS / chunk_secs)

    proc = subprocess.Popen(
        ["pw-record",
         f"--target={source}",
         f"--channels={MIC_CHANNELS}",
         f"--rate={MIC_RATE}",
         "--format=s16",
         "--latency=50",
         "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frames        = []
    silent_chunks = 0
    recording     = False
    total_chunks  = 0

    if verbose:
        print(f"  [listening, threshold={threshold:.0f}...]",
              end="", flush=True)

    try:
        while total_chunks < max_chunks:
            if _ovos_speaking or _sunny_speaking:
                frames = []  # discard anything captured so far
                break
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk or len(chunk) < chunk_bytes:
                break
            total_chunks += 1
            samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))

            if rms > threshold:
                if not recording:
                    if verbose:
                        print(f"\n  [recording, RMS={rms:.0f}]",
                              end="", flush=True)
                    recording = True
                silent_chunks = 0
                frames.append(chunk)
            elif recording:
                silent_chunks += 1
                frames.append(chunk)
                if silent_chunks >= silence_chunks_needed:
                    break
    finally:
        proc.terminate()
        proc.wait()

    if verbose:
        print()

    if not frames:
        return None
    pcm = b"".join(frames)
    if len(pcm) / (MIC_RATE * MIC_CHANNELS * 2) < MIN_SPEECH_SECS:
        return None
    return pcm

# ── STT ───────────────────────────────────────────────────────────────────────

def transcribe(pcm: bytes, model: WhisperModel) -> str:
    """Transcribe raw PCM bytes using faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(MIC_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(MIC_RATE)
            wf.writeframes(pcm)
        segments, _ = model.transcribe(
            wav_path, language="en", vad_filter=True)
        parts = [
            s.text.strip() for s in segments
            if s.no_speech_prob <= NO_SPEECH_THRESHOLD
            and s.avg_logprob   >= LOGPROB_THRESHOLD
        ]
        return " ".join(parts).strip()
    finally:
        Path(wav_path).unlink(missing_ok=True)

def _is_wake_word(text: str) -> bool:
    """Return True if the beginning of text fuzzy-matches a foreign wake word variant.

    Only the start of the utterance is checked so that longer sentences which
    happen to contain a wake-word-like sound are not accidentally dropped.
    """
    import difflib
    normalized = re.sub(r"[^a-z0-9]", "", text.lower())
    for variant in WAKE_WORD_VARIANTS:
        # Compare a prefix of the transcription — variant length plus a small
        # buffer for Whisper length variation — against the known variant.
        prefix = normalized[:len(variant) + 4]
        if difflib.SequenceMatcher(None, prefix, variant).ratio() >= WAKE_WORD_THRESHOLD:
            return True
    return False

# ── LLM ───────────────────────────────────────────────────────────────────────

async def llm_respond(history: list[dict],
                      http: httpx.AsyncClient,
                      long_term: str = "",
                      ha_context: str | None = None,
                      verbose: bool = False,
                      claude_client=None,
                      openrouter_cfg: dict | None = None) -> str:
    """Stream a response from Ollama or Claude API. Returns the full response text."""
    if ha_context is None:
        ha_context = await get_ha_context(http, verbose=verbose)
    memory_section = (f"\n\n## {MEMORY_HEADER}\n{long_term}" if long_term else "")
    reinforcement  = (f"\n\n## Important\n{CHARACTER_REINFORCEMENT}"
                      if CHARACTER_REINFORCEMENT else "")

    if claude_client is not None:
        dynamic_suffix = (
            memory_section
            + f"\n\n## Current situation:\n{ha_context}"
            + reinforcement
        )
        base_prompt = CLAUDE_SYSTEM_PROMPT or SYSTEM_PROMPT
        system_blocks = [
            {"type": "text", "text": base_prompt,
             "cache_control": {"type": "ephemeral"}},
        ]
        if dynamic_suffix.strip():
            system_blocks.append({"type": "text", "text": dynamic_suffix})
        response_text = ""
        async with claude_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_blocks,
            messages=history,
        ) as stream:
            async for text in stream.text_stream:
                response_text += text
        return _trim_to_last_sentence(response_text)

    if openrouter_cfg is not None:
        base_prompt = openrouter_cfg.get("system_prompt") or SYSTEM_PROMPT
        dynamic_system = base_prompt + memory_section + f"\n\n## Current situation:\n{ha_context}" + reinforcement
        headers = {
            "Authorization": f"Bearer {openrouter_cfg['api_key']}",
            "X-Title": "loca-talk",
        }
        payload = {
            "model":    openrouter_cfg["model"],
            "messages": [{"role": "system", "content": dynamic_system}] + history,
            "stream":   True,
        }
        response_text = ""
        async with http.stream("POST", OPENROUTER_URL, json=payload,
                               headers=headers) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    token = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if token:
                        response_text += token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return _trim_to_last_sentence(response_text)

    dynamic_system = SYSTEM_PROMPT + memory_section + f"\n\n## Current situation:\n{ha_context}" + reinforcement
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "system", "content": dynamic_system}]
                     + history,
        "stream":   True,
        "options":  OLLAMA_OPTIONS,
    }
    response_text = ""
    _gpu_fan_set()
    try:
        async with http.stream("POST", OLLAMA_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    response_text += token
    finally:
        _gpu_fan_auto()
    return _trim_to_last_sentence(response_text)


def _trim_to_last_sentence(text: str) -> str:
    """Trim to the last complete sentence, removing trailing fragments and second-voice content."""
    text = text.strip()
    # Strip special token artifacts: <s>, [control_N], [TOOL_CALLS], etc.
    text = re.sub(r'^(?:\s*(?:<[^>]+>|\[[^\]]+\])\s*)+', '', text)
    # Strip stochastic word/char prefix before real sentence, e.g. "qpoint Oh..." → "Oh..."
    text = re.sub(r'^[A-Za-z]+\s+(?=[A-Z])', '', text)
    # Fix CamelCase word fusions the model produces under context pressure, e.g. "MyBabyJennie"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Find the last sentence-ending punctuation
    last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_end > 0:
        text = text[:last_end + 1].strip()
    return text

_REFUSAL_PHRASES = (
    "i can't continue", "i cannot continue",
    "i'm not comfortable", "i am not comfortable",
    "i need to be honest",
    "content policy", "guidelines",
    "infantilization", "unhealthy pattern",
    "not the right tool", "other communities",
    "psychological regression", "beyond innocent",
    "i won't be able to",
    "i need to pause", "step out of character",
    "your wellbeing", "not something i can continue",
    "doesn't feel right for me to encourage",
    "might not be healthy", "talk about this differently",
)


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)

# ── Stdin reader (non-blocking, cancellable) ──────────────────────────────────

# ── Main conversation loop ────────────────────────────────────────────────────

async def conversation_loop(args):
    global _shutdown_requested, _awake, _sunny_speaking

    loop = asyncio.get_running_loop()
    _this_task = asyncio.current_task()
    def _sigint_handler():
        print("\n[Ctrl-C — shutting down...]", flush=True)
        _this_task.cancel()
    loop.add_signal_handler(signal.SIGINT, _sigint_handler)

    # Attach stdin to an asyncio StreamReader so readline() is non-blocking
    # and cancellable without any Python-buffering / select mismatch.
    _stdin_reader = None
    if args.type:
        _stdin_reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(_stdin_reader),
            sys.stdin)

    if not args.silent:
        if not device_is_connected():
            if not await wait_for_device():
                print("Device not available. Exiting.")
                return

    whisper = None
    if not args.type:
        print("Loading Whisper model...", end=" ", flush=True)
        whisper = WhisperModel(WHISPER_MODEL,
                               device=WHISPER_DEVICE,
                               compute_type=WHISPER_COMPUTE)
        print("ready.")

    if not args.silent:
        print("Loading TTS model...", end=" ", flush=True)
        init_tts()
        print("ready.")

    noise_threshold = 1000.0
    if not args.type:
        print("Calibrating microphone noise floor...", end=" ", flush=True)
        noise_threshold = calibrate_noise_floor()
        print(f"threshold={noise_threshold:.0f}")

    voice    = args.voice
    speed    = args.speed

    claude_client   = None
    openrouter_cfg  = None
    if args.llm == "claude":
        if _anthropic is None:
            print("Error: 'anthropic' package not installed. Run: pip install anthropic")
            return
        api_key = CFG.get("claude", {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Claude API key not set. Add claude.api_key to hfp_secret.yaml or set ANTHROPIC_API_KEY.")
            return
        claude_client = _anthropic.AsyncAnthropic(api_key=api_key)
        print(f"Using Claude API ({CLAUDE_MODEL})")
    elif args.llm == "openrouter":
        api_key = CFG.get("openrouter", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OpenRouter API key not set. Add openrouter.api_key to hfp_secret.yaml or set OPENROUTER_API_KEY.")
            return
        openrouter_cfg = {
            "api_key":       api_key,
            "model":         OPENROUTER_MODEL,
            "system_prompt": OPENROUTER_SYSTEM_PROMPT,
        }
        print(f"Using OpenRouter ({OPENROUTER_MODEL})")
    elif args.llm == "ollama":
        print(f"Using Ollama ({OLLAMA_MODEL})")
    else:
        print(f"Error: unknown LLM backend '{args.llm}'. Known backends: claude, openrouter, ollama")
        return

    history, long_term = load_memory()
    if history:
        print(f"  (resuming: {len(history)} turns from last session)")
    if long_term:
        print(f"  (memory: {long_term[:72]}{'...' if len(long_term) > 72 else ''})")
    turns_since_compact = 0

    event_queue: asyncio.Queue = asyncio.Queue()
    _readline_task   = None
    _bedtime_watcher = None

    async with httpx.AsyncClient(timeout=120) as http:
        # Fetch initial sleep state before greeting so we know whether to speak.
        if AWAKE_ENTITY:
            try:
                r = await http.get(f"{HA_URL}/{AWAKE_ENTITY}",
                                   headers={"Authorization": f"Bearer {HA_TOKEN}"}, timeout=3.0)
                if r.status_code == 200:
                    _awake = r.json().get("state", "").lower() == "on"
                    if args.verbose:
                        print(f"  [init] awake={_awake}")
            except Exception:
                pass

        if args.greet and _awake:
            print(f"{ASSISTANT_NAME}: {GREETING}")
            if not args.silent:
                try:
                    speak(GREETING, voice=voice, speed=speed)
                except Exception as e:
                    print(f"  [greet failed: {e}]")

        print("\nConversation started. Ctrl-C to exit.\n")

        _sensor_watcher = asyncio.create_task(
                      sensor_watcher(http, event_queue, verbose=args.verbose)) \
                  if TRIGGERS else None
        _bedtime_watcher = asyncio.create_task(
                       bedtime_watcher(event_queue)) \
                   if TRIGGERS else None
        _ovos_watcher = asyncio.create_task(
                            ovos_watcher(verbose=args.verbose)) \
                        if OVOS_ENTITY else None
        _awake_watcher = asyncio.create_task(
                             awake_watcher(verbose=args.verbose, silent=args.silent)) \
                         if AWAKE_ENTITY else None

        while not _shutdown_requested:
            try:
                # ── Connection check ───────────────────────────────────────
                if not args.silent and not device_is_connected():
                    print("\nDevice disconnected.")
                    save_recent_turns(history)
                    if not await wait_for_device():
                        print("Device did not return. Exiting.")
                        break
                    print("Re-calibrating...", end=" ", flush=True)
                    noise_threshold = calibrate_noise_floor()
                    print(f"threshold={noise_threshold:.0f}")
                    history = []

                # ── Sensor-triggered events ────────────────────────────────
                while not event_queue.empty():
                    event_text = event_queue.get_nowait()
                    if not _awake:
                        print(f"\n[event] {event_text} (ignored; sleeping)")
                        continue
                    print(f"\n[event] {event_text}")
                    history.append({"role": "user", "content": event_text})
                    response_text = await llm_respond(
                        history[-MAX_RECENT_TURNS:], http,
                        long_term=long_term,
                        verbose=args.verbose,
                        claude_client=claude_client,
                        openrouter_cfg=openrouter_cfg)
                    if (claude_client is not None or openrouter_cfg is not None) and _is_refusal(response_text):
                        if args.verbose:
                            print("  [refusal detected — substituting redirect]")
                        response_text = random.choice(_REFUSAL_REDIRECTS)
                    print(f"{ASSISTANT_NAME}: {response_text}")
                    if response_text.strip():
                        history.append({"role": "assistant",
                                        "content": response_text})
                        save_recent_turns(history)
                        turns_since_compact += 1
                        if turns_since_compact >= COMPACT_EVERY:
                            turns_since_compact = 0
                            long_term = await compact_memory(
                                history, long_term, http,
                                verbose=args.verbose,
                                claude_client=claude_client,
                                openrouter_cfg=openrouter_cfg)
                        if len(history) > 40:
                            history = history[-40:]
                        if not args.silent:
                            clean = sanitise(response_text)
                            clean = truncate_to_complete_sentences(clean)
                            if clean:
                                try:
                                    await asyncio.get_event_loop() \
                                        .run_in_executor(
                                            None,
                                            lambda c=clean: speak(
                                                c, voice=voice,
                                                speed=speed, sink=AUDIO_SINK))
                                except Exception as e:
                                    print(f"  [speak failed: {e}]")

                # ── Get user input ─────────────────────────────────────────
                if args.type:
                    if _readline_task is None or _readline_task.done():
                        _readline_task = asyncio.ensure_future(
                            _stdin_reader.readline())
                        print("You: ", end="", flush=True)
                    _event_wait = asyncio.ensure_future(event_queue.get())
                    done, _ = await asyncio.wait(
                        {_readline_task, _event_wait},
                        return_when=asyncio.FIRST_COMPLETED)
                    if _readline_task in done:
                        if _event_wait in done:
                            event_queue.put_nowait(_event_wait.result())
                        else:
                            _event_wait.cancel()
                        data = _readline_task.result()
                        _readline_task = None
                        if not data:
                            break
                        user_text = data.decode().rstrip("\n").strip()
                        if not user_text:
                            continue
                    else:
                        # Sensor event arrived — requeue it and let the drain handle it
                        event_queue.put_nowait(_event_wait.result())
                        _readline_task.cancel()
                        _readline_task = None
                        print()
                        continue
                else:
                    if _ovos_speaking or _sunny_speaking:
                        await asyncio.sleep(0.2)
                        continue
                    t_start = time.monotonic()
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: record_utterance(
                            source=AUDIO_SOURCE,
                            threshold=noise_threshold,
                            verbose=args.verbose))
                    t_recorded = time.monotonic()

                    if not pcm or _sunny_speaking:
                        continue

                    if args.verbose:
                        print(f"  [recording: {t_recorded - t_start:.2f}s, "
                              f"{len(pcm):,} bytes, "
                              f"{len(pcm)/(MIC_RATE*2):.2f}s audio]")

                    t0 = time.monotonic()
                    user_text = transcribe(pcm, whisper)
                    if args.verbose:
                        print(f"  [STT: {time.monotonic()-t0:.2f}s]")

                    if not user_text:
                        if args.verbose:
                            print("  [no speech detected]")
                        continue

                    if WAKE_WORD_VARIANTS and _is_wake_word(user_text):
                        if args.verbose:
                            print(f"  [wake word filtered: {user_text!r}]")
                        continue

                    print(f"You:   {user_text}")

                # ── Sleep / wake voice commands ────────────────────────────
                _norm = re.sub(r"[^a-z0-9 ]", "", user_text.lower()).strip()
                if WAKE_PHRASES and any(p in _norm for p in WAKE_PHRASES):
                    if not _awake:
                        _awake = True
                        if args.verbose:
                            print("  [woken by voice command]")
                        if not args.silent:
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: speak(WAKE_RESPONSE,
                                                        voice=voice, speed=speed,
                                                        sink=AUDIO_SINK))
                            except Exception:
                                pass
                    continue

                if SLEEP_PHRASES and any(p in _norm for p in SLEEP_PHRASES):
                    if _awake:
                        _awake = False
                        if args.verbose:
                            print("  [sleeping by voice command]")
                        if not args.silent:
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: speak(SLEEP_RESPONSE,
                                                        voice=voice, speed=speed,
                                                        sink=AUDIO_SINK))
                            except Exception:
                                pass
                    continue

                if not _awake:
                    if args.verbose:
                        print(f"  [sleeping — ignoring: {user_text!r}]")
                    continue

                history.append({"role": "user", "content": user_text})

                # ── LLM response ───────────────────────────────────────────
                ha_context = await get_ha_context(http, verbose=args.verbose)
                response_text = ""
                for _attempt in range(3):
                    t0 = time.monotonic()
                    response_text = await llm_respond(
                        history[-MAX_RECENT_TURNS:], http,
                        long_term=long_term,
                        ha_context=ha_context,
                        verbose=args.verbose,
                        claude_client=claude_client,
                        openrouter_cfg=openrouter_cfg)
                    if args.verbose:
                        print(f"  [LLM: {time.monotonic()-t0:.2f}s]")
                    if response_text.strip():
                        break

                if not response_text.strip():
                    history.pop()   # remove unanswered user message
                    continue

                if claude_client is not None and _is_refusal(response_text):
                    if args.verbose:
                        print("  [refusal detected — substituting redirect]")
                    response_text = random.choice(_REFUSAL_REDIRECTS)

                print(f"{ASSISTANT_NAME}: {response_text}")

                history.append({"role": "assistant",
                                 "content": response_text})

                save_recent_turns(history)
                turns_since_compact += 1
                if turns_since_compact >= COMPACT_EVERY:
                    turns_since_compact = 0
                    long_term = await compact_memory(
                        history, long_term, http, verbose=args.verbose,
                        claude_client=claude_client,
                        openrouter_cfg=openrouter_cfg)

                if len(history) > 40:
                    history = history[-40:]

                # ── TTS → speaker ──────────────────────────────────────────
                if not args.silent:
                    clean = sanitise(response_text)
                    clean = truncate_to_complete_sentences(clean)
                    if clean:
                        t0 = time.monotonic()
                        try:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: speak(
                                    clean,
                                    voice=voice,
                                    speed=speed,
                                    sink=AUDIO_SINK))
                        except Exception as e:
                            print(f"  [speak failed: {e}]")
                        if args.verbose:
                            print(f"  [TTS+play: {time.monotonic()-t0:.2f}s]")

            except (KeyboardInterrupt, asyncio.CancelledError):
                # User-initiated exit — save memory then say goodbye
                print()
                save_recent_turns(history)
                if history:
                    long_term = await compact_memory(
                        history, long_term, http, verbose=args.verbose,
                        claude_client=claude_client,
                        openrouter_cfg=openrouter_cfg)
                if device_is_connected():
                    try:
                        speak("Goodbye, sweetheart. Sweet dreams.",
                              voice=voice, speed=speed)
                    except Exception:
                        pass
                break

            except Exception as e:
                print(f"  [error: {type(e).__name__}: {e}]")
                continue

        if _sensor_watcher:
            _sensor_watcher.cancel()
        if _bedtime_watcher:
            _bedtime_watcher.cancel()

    # SIGTERM path — save memory, no goodbye
    save_recent_turns(history)
    if history:
        async with httpx.AsyncClient(timeout=30) as http:
            await compact_memory(history, long_term, http, verbose=False,
                                 claude_client=claude_client,
                                 openrouter_cfg=openrouter_cfg)
    print("Shutting down.")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="loca-talk",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--type", action="store_true",
        help="Type input instead of speaking")
    p.add_argument("--silent", action="store_true",
        help="Print responses only, no TTS or audio (implies no device needed)")
    p.add_argument("--greet", action="store_true",
        help="Have the assistant say hello on startup")
    p.add_argument("--voice", default=DEFAULT_VOICE,
        metavar="NAME",
        help=f"Kokoro voice name (default: {DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=SPEED,
        metavar="F",
        help=f"TTS speed, <1.0 slower (default: {SPEED})")
    p.add_argument("--verbose", "-v", action="store_true",
        help="Print timing and debug information")
    p.add_argument("--llm", default=None,
        metavar="BACKEND",
        help="LLM backend: ollama, claude, openrouter (default: llm.provider from config)")

    args = p.parse_args()
    if args.llm is None:
        args.llm = DEFAULT_LLM
    asyncio.run(conversation_loop(args))

if __name__ == "__main__":
    main()
