#!/usr/bin/env python3
"""
loca-shell.py — Interactive BLE REPL for the device.

Connects once and stays connected. Type 'help' for commands.
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

import asyncio
import base64
import httpx
import json
import re
import readline  # enables arrow keys and history at the prompt
import secrets
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner
from kokoro import KPipeline
from config import CFG

# ── Configuration ─────────────────────────────────────────────────────────────

BLE_MAC       = CFG["ble"]["mac"]
DEFAULT_VOICE = CFG["tts"]["voice"]
SPEED         = CFG["tts"]["speed"]
PAD_DURATION  = CFG["tts"]["pad_duration"]
MP3_BITRATE   = CFG["tts"]["mp3_bitrate"]
OLLAMA_URL    = CFG["llm"]["ollama_url"]
OLLAMA_MODEL  = CFG["llm"]["ollama_model"]
SYSTEM_PROMPT = CFG["system_prompt"]

# ── TTS pipeline (lazy singleton) ─────────────────────────────────────────────

_tts_pipeline: KPipeline | None = None

def _get_tts() -> KPipeline:
    global _tts_pipeline
    if _tts_pipeline is None:
        _tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _tts_pipeline

# ── GATT UUIDs ────────────────────────────────────────────────────────────────

ACK_CHAR  = "0000AE01-0000-1000-8000-00805f9b34fb"
UPL_CHAR  = "0000AE02-0000-1000-8000-00805f9b34fb"
CMD_CHAR  = "0000AE03-0000-1000-8000-00805f9b34fb"
ANS_CHAR  = "0000AE04-0000-1000-8000-00805f9b34fb"
IND_CHAR  = "0000FEE5-0000-1000-8000-00805f9b34fb"
FF61_CHAR = "0000ff61-0000-1000-8000-00805f9b34fb"
FF62_CHAR = "0000ff62-0000-1000-8000-00805f9b34fb"

# ── Packet framing ────────────────────────────────────────────────────────────

HEAD    = bytes([0x4A, 0x4C])
TCMD    = bytes([0x40, 0x01])
TUPLOAD = bytes([0x20, 0x01])
TAIL    = bytes([0xFF])

AUDIO_CHUNK = 4096
MTU_SIZE    = 509
MAX_RETRY   = 3

STATUS_OK      = "0"
STATUS_PENDING = "1"
STATUS_EMPTY   = "2"

# ── CRC ───────────────────────────────────────────────────────────────────────

def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
    crc &= 0xFFFF
    return bytes([crc >> 8, crc & 0xFF])

# ── Packet builders ───────────────────────────────────────────────────────────

def make_cmd(json_str: str) -> bytes:
    payload   = json_str.encode("utf-8")
    length    = len(payload) + 2
    len_bytes = struct.pack(">H", length)
    return HEAD + len_bytes + TCMD + payload + b'\x00\x00' + TAIL

def make_upload_packet(audio_chunk: bytes, title: str,
                       pos: int, total: int) -> bytes:
    if total == 1:
        pack_pos = 2
    elif pos == 0:
        pack_pos = 0
    elif pos == total - 1:
        pack_pos = 2
    else:
        pack_pos = 1
    meta_json = f'{{"ringName":"{title}","re":"{pack_pos}"}}'.encode("utf-8")
    audio_crc = crc16(audio_chunk)
    inner_len  = len(audio_chunk) + len(audio_crc) + len(meta_json)
    len_field4 = bytes([0, 0, (inner_len >> 8) & 0xFF, inner_len & 0xFF])
    meta_len   = struct.pack(">H", len(meta_json))
    return (HEAD + len_field4 + TUPLOAD +
            meta_len + meta_json +
            audio_chunk + audio_crc +
            TAIL)

def split_for_mtu(data: bytes, mtu_size: int = MTU_SIZE) -> list:
    return [data[i:i+mtu_size] for i in range(0, len(data), mtu_size)]

def new_file_id() -> str:
    raw = secrets.token_bytes(15)
    b64 = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"U-{b64[:19]}"

def strip_ext(file_id: str) -> str:
    return re.sub(r'\.mp3$', '', file_id, flags=re.IGNORECASE)

# ── Response reassembly ───────────────────────────────────────────────────────

class NotifBuffer:
    def __init__(self):
        self.data         = bytearray()
        self.expected_len = 0

    def feed(self, chunk: bytes):
        self.data.extend(chunk)
        if self.expected_len == 0 and len(self.data) >= 4:
            self.expected_len = (self.data[2] << 8) | self.data[3]
        if self.expected_len > 0 and len(self.data) >= self.expected_len + 7:
            payload = bytes(self.data[6 : self.expected_len + 4])
            self.data.clear()
            self.expected_len = 0
            return payload
        return None

    def reset(self):
        self.data.clear()
        self.expected_len = 0

# ── Bear interface ────────────────────────────────────────────────────────────

class Bear:
    def __init__(self, client: BleakClient):
        self.client   = client
        self.ans_buf  = NotifBuffer()
        self.ack_buf  = NotifBuffer()
        self._ans_q   = asyncio.Queue()
        self._ack_q   = asyncio.Queue()
        self.log_raw  = False
        self.verbose  = False

    async def connect(self):
        await self.client.start_notify(ANS_CHAR, self._on_ans)
        await self.client.start_notify(ACK_CHAR, self._on_ack)
        for uuid, name in [
            (IND_CHAR,  "FEE5 (indicate)"),
            (FF62_CHAR, "FF62 (notify)"),
        ]:
            try:
                await self.client.start_notify(uuid, self._on_extra(name))
                print(f"  Subscribed to {name}")
            except Exception as e:
                print(f"  {name} subscribe failed: {e}")

    def _on_extra(self, name: str):
        def handler(sender, data):
            print(f"\n  [{name}] {bytes(data).hex()} "
                  f"| {bytes(data).decode(errors='replace')}")
        return handler

    def _on_ans(self, sender, data):
        if self.log_raw:
            print(f"\n  [ANS] {bytes(data).hex()} "
                  f"| {bytes(data).decode(errors='replace')}")
        payload = self.ans_buf.feed(bytes(data))
        if payload is not None:
            self._ans_q.put_nowait(payload)

    def _on_ack(self, sender, data):
        if self.log_raw:
            print(f"\n  [ACK] {bytes(data).hex()} "
                  f"| {bytes(data).decode(errors='replace')}")
        payload = self.ack_buf.feed(bytes(data))
        if payload is not None:
            self._ack_q.put_nowait(payload)

    async def _send_raw(self, json_str: str, timeout: float = 5.0) -> dict:
        while not self._ans_q.empty():
            self._ans_q.get_nowait()
        pkt = make_cmd(json_str)
        await self.client.write_gatt_char(CMD_CHAR, pkt, response=True)
        try:
            raw = await asyncio.wait_for(self._ans_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"No response to: {json_str}")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"_raw_hex": raw.hex()}

    async def clear_all_and_wait(self, timeout: float = 15.0) -> float:
        """Returns elapsed seconds."""
        while not self._ans_q.empty():
            self._ans_q.get_nowait()
        t0 = time.monotonic()
        pkt = make_cmd('{"cmd":"ClearAll","parm":"0"}')
        await self.client.write_gatt_char(CMD_CHAR, pkt, response=True)
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("ClearAll timed out")
            try:
                raw = await asyncio.wait_for(
                    self._ans_q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError("ClearAll timed out")
            try:
                resp = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            status = str(resp.get("status", ""))
            if status == STATUS_OK:
                return time.monotonic() - t0
            elif status == STATUS_PENDING:
                continue
            else:
                raise RuntimeError(f"ClearAll failed: {resp}")

    async def upload(self, audio_data: bytes, title: str) -> float:
        """Upload MP3 bytes. Returns elapsed seconds."""
        title  = strip_ext(title)
        chunks = [audio_data[i:i+AUDIO_CHUNK]
                  for i in range(0, len(audio_data), AUDIO_CHUNK)]
        total  = len(chunks)
        print(f"  Uploading {len(audio_data):,} bytes ({total} chunks)...")
        t0 = time.monotonic()
        for i, chunk in enumerate(chunks):
            packet  = make_upload_packet(chunk, title, i, total)
            subpkts = split_for_mtu(packet)
            pack_label = ("single" if total == 1 else
                          "first"  if i == 0 else
                          "last"   if i == total - 1 else "middle")
            if self.verbose:
                print(f"  Chunk {i}: {len(chunk)}B, "
                      f"{len(subpkts)} subpkts [{pack_label}]")
            retries = 0
            while True:
                while not self._ack_q.empty():
                    self._ack_q.get_nowait()
                for sp in subpkts:
                    await self.client.write_gatt_char(
                        UPL_CHAR, sp, response=False)
                try:
                    ack_raw = await asyncio.wait_for(
                        self._ack_q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"No ACK for chunk {i+1}/{total}")
                try:
                    ack = json.loads(ack_raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    ack = {"_raw": ack_raw.hex()}
                if self.verbose:
                    print(f"  Chunk {i} ACK: {ack}")
                ack_status = ack.get("status", -1)
                if ack_status == 0:
                    break
                elif ack_status == 2 and retries < MAX_RETRY:
                    retries += 1
                    print(f"  Chunk {i} NAK — retry {retries}/{MAX_RETRY}")
                else:
                    raise RuntimeError(
                        f"Upload failed at chunk {i+1}/{total}: {ack}")
        return time.monotonic() - t0

    async def play(self, file_id: str) -> dict:
        return await self._send_raw(
            f'{{"cmd":"try","parm":"{strip_ext(file_id)}"}}')

    async def delete_and_wait(self, file_id: str,
                              timeout: float = 15.0) -> float:
        """Delete a file by ID, polling until STATUS_OK. Returns elapsed seconds."""
        fid = strip_ext(file_id)
        while not self._ans_q.empty():
            self._ans_q.get_nowait()
        t0 = time.monotonic()
        pkt = make_cmd(f'{{"cmd":"deleteRing","parm":"{fid}"}}')
        await self.client.write_gatt_char(CMD_CHAR, pkt, response=True)
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("deleteRing timed out")
            try:
                raw = await asyncio.wait_for(
                    self._ans_q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError("deleteRing timed out")
            try:
                resp = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            status = str(resp.get("status", ""))
            if status == STATUS_OK:
                return time.monotonic() - t0
            elif status == STATUS_PENDING:
                continue
            else:
                raise RuntimeError(f"deleteRing failed: {resp}")

    async def stop(self) -> dict:
        return await self._send_raw('{"cmd":"playmusic","parm":"0"}')

    async def list_files(self) -> list:
        resp = await self._send_raw('{"cmd":"getAudioList","parm":"0"}')
        status = str(resp.get("status", ""))
        if status == STATUS_OK:
            return resp.get("list", [])
        if status == STATUS_EMPTY:
            return []
        raise RuntimeError(f"getAudioList failed: {resp}")

    async def info(self) -> dict:
        return {
            "version": await self._send_raw('{"cmd":"getVersion","parm":"0"}'),
            "disk":    await self._send_raw('{"cmd":"getDisk","parm":"0"}'),
            "battery": await self._send_raw('{"cmd":" BatLevel","parm":"0"}'),
            "volume":  await self._send_raw('{"cmd":"getVolume","parm":"1"}'),
            "name":    await self._send_raw('{"cmd":"getName","parm":"0"}'),
        }

    async def write_ff61(self, data: bytes) -> None:
        await self.client.write_gatt_char(FF61_CHAR, data, response=False)

# ── TTS ───────────────────────────────────────────────────────────────────────

def sanitise_for_tts(text: str) -> str:
    text = re.sub(r'\*[^*]*\*', '', text)
    text = re.sub(r'_[^_]*_', '', text)
    text = re.sub(r'\[/?INST\]', '', text)
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'^[A-Za-z\s]+:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_to_mp3(text: str,
               voice: str = DEFAULT_VOICE,
               speed: float = SPEED) -> tuple[bytes, float]:
    pipeline = _get_tts()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    pcm_float = np.concatenate(chunks)
    pcm_int16 = (np.clip(pcm_float, -1.0, 1.0) * 32767).astype(np.int16)

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "s16le", "-ar", "24000", "-ac", "1",
         "-i", "pipe:0",
         "-codec:a", "libmp3lame",
         "-b:a", MP3_BITRATE,
         "-af", f"apad=pad_dur={PAD_DURATION}",
         "-f", "mp3", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mp3, ffmpeg_err = ffmpeg.communicate(input=pcm_int16.tobytes())
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{ffmpeg_err.decode(errors='replace')}")

    duration = len(pcm_float) / 24000 + PAD_DURATION
    return mp3, duration

# ── REPL ─────────────────────────────────────────────────────────────────────

HELP = """
Commands:
  info                     Device info (version, disk, battery, volume)
  list                     List files on bear
  play <id>                Play a stored file by ID
  stop                     Stop playback
  clear                    ClearAll and wait for completion
  delete <id>              Delete a single file by ID (no .mp3 extension)
  upload <path> [<id>]     Upload an MP3 file (generates ID if omitted)
  say <text>               TTS → MP3 → upload → play (stays connected)
  say-sleep <secs> <text>  say, then sleep N seconds before returning
  wait <secs>              Sleep N seconds (keeps connection alive)
  raw <json>               Send raw JSON command, print response
  disk                     Raw getDisk response
  log on|off               Toggle raw notification logging
  verbose on|off           Toggle verbose upload messages
  ff61 <hex>               Write hex bytes to FF61 characteristic
  newid                    Generate a new firmware-compatible file ID
  voice [<name>]           Show or set TTS voice (Kokoro voice name)
  speed [<n>]              Show or set TTS speed (default 0.85, <1.0 = slower)
  chat                     Enter LLM chat mode (Ollama)
  help                     Show this help
  quit / exit / q          Disconnect and exit
"""

async def repl(bear: Bear):
    state = {
        "voice": DEFAULT_VOICE,
        "speed": SPEED,
    }
    last_id = None

    print(HELP)

    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("loca> ").strip())
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")
            continue

        cmd  = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "help":
                print(HELP)

            elif cmd == "info":
                data = await bear.info()
                for k, v in data.items():
                    print(f"  {k}: {v}")

            elif cmd == "disk":
                print(await bear._send_raw('{"cmd":"getDisk","parm":"0"}'))

            elif cmd == "list":
                files = await bear.list_files()
                if not files:
                    print("  No files on bear.")
                else:
                    for f in files:
                        print(f"  {f['id']:<35} {f['name']:<40} cn={f['cn']}")

            elif cmd == "play":
                fid = args[0] if args else last_id
                if not fid:
                    print("  Usage: play <id>")
                    continue
                resp = await bear.play(fid)
                print(f"  {resp}")
                last_id = fid

            elif cmd == "stop":
                print(await bear.stop())

            elif cmd == "clear":
                elapsed = await bear.clear_all_and_wait()
                print(f"  Cleared in {elapsed:.2f}s")

            elif cmd == "delete":
                fid = args[0] if args else last_id
                if not fid:
                    print("  Usage: delete <id>")
                    continue
                elapsed = await bear.delete_and_wait(fid)
                print(f"  Deleted in {elapsed:.2f}s")

            elif cmd == "wait":
                secs = float(args[0]) if args else 1.0
                print(f"  Waiting {secs}s...")
                await asyncio.sleep(secs)
                print("  Done.")

            elif cmd == "upload":
                if not args:
                    print("  Usage: upload <path> [<id>]")
                    continue
                path = Path(args[0])
                if not path.exists():
                    print(f"  File not found: {path}")
                    continue
                fid     = args[1] if len(args) > 1 else new_file_id()
                data    = path.read_bytes()
                elapsed = await bear.upload(data, fid)
                print(f"  Upload done in {elapsed:.2f}s  id={fid}")
                last_id = fid

            elif cmd in ("say", "say-sleep"):
                if cmd == "say-sleep":
                    if len(args) < 2:
                        print("  Usage: say-sleep <secs> <text>")
                        continue
                    extra_sleep = float(args[0])
                    text = " ".join(args[1:])
                else:
                    if not args:
                        print("  Usage: say <text>")
                        continue
                    extra_sleep = 0.0
                    text = " ".join(args)

                print(f"  TTS: {text!r}")
                t0 = time.monotonic()
                mp3, duration = tts_to_mp3(
                    text,
                    voice=state["voice"],
                    speed=state["speed"])
                print(f"  Rendered {len(mp3):,} bytes, "
                      f"{duration:.2f}s in {time.monotonic()-t0:.2f}s")

                fid     = new_file_id()
                elapsed = await bear.upload(mp3, fid)
                print(f"  Uploaded in {elapsed:.2f}s  id={fid}")

                resp = await bear.play(fid)
                print(f"  Play: {resp}")
                last_id = fid

                if extra_sleep > 0:
                    total_wait = duration + extra_sleep
                    print(f"  Sleeping {total_wait:.2f}s...")
                    await asyncio.sleep(total_wait)
                    print("  Done sleeping.")

            elif cmd == "raw":
                if not args:
                    print("  Usage: raw <json>")
                    continue
                print(await bear._send_raw(" ".join(args)))

            elif cmd == "log":
                if args:
                    bear.log_raw = args[0].lower() == "on"
                print(f"  Logging {'ON' if bear.log_raw else 'OFF'}")

            elif cmd == "verbose":
                if args:
                    bear.verbose = args[0].lower() == "on"
                print(f"  Verbose {'ON' if bear.verbose else 'OFF'}")

            elif cmd == "ff61":
                if not args:
                    print("  Usage: ff61 <hex bytes, e.g. 01 02 03>")
                    continue
                data = bytes.fromhex("".join(args))
                await bear.write_ff61(data)
                print(f"  Wrote {data.hex()} to FF61")

            elif cmd == "newid":
                print(f"  {new_file_id()}")

            elif cmd == "voice":
                if args:
                    state["voice"] = args[0]
                print(f"  Voice: {state['voice']}")

            elif cmd == "speed":
                if args:
                    state["speed"] = float(args[0])
                print(f"  Speed: {state['speed']}")

            elif cmd == "chat":
                print("  Entering chat mode. Type 'done' to return to shell.")
                history = []
                async with httpx.AsyncClient(timeout=60) as http:
                    while True:
                        try:
                            user_input = await asyncio.get_event_loop() \
                                .run_in_executor(
                                    None,
                                    lambda: input("you> ").strip())
                        except (EOFError, KeyboardInterrupt):
                            print()
                            break
                        if user_input.lower() in ("done", "exit", "quit"):
                            break
                        if not user_input:
                            continue

                        history.append({"role": "user",
                                        "content": user_input})

                        print("bear> ", end="", flush=True)
                        payload = {
                            "model":    OLLAMA_MODEL,
                            "messages": [{"role": "system",
                                          "content": SYSTEM_PROMPT}]
                                         + history,
                            "stream":   True,
                        }

                        response_text = ""
                        async with http.stream("POST", OLLAMA_URL,
                                               json=payload) as resp:
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                token = chunk.get("message", {}) \
                                             .get("content", "")
                                if token:
                                    print(token, end="", flush=True)
                                    response_text += token
                        print()

                        history.append({"role": "assistant",
                                        "content": response_text})

                        clean = sanitise_for_tts(response_text)
                        if clean:
                            t_tts = time.monotonic()
                            mp3, duration = tts_to_mp3(
                                clean,
                                voice=state["voice"],
                                speed=state["speed"])
                            print(f"  [TTS: {time.monotonic()-t_tts:.1f}s, "
                                  f"{len(mp3):,}B, {duration:.1f}s]")

                            fid = new_file_id()
                            await bear.upload(mp3, fid)
                            await bear.play(fid)
                            await asyncio.sleep(duration)
                            await bear.delete_and_wait(fid)

            else:
                print(f"  Unknown command: {cmd!r}  (type 'help')")

        except TimeoutError as e:
            print(f"  Timeout: {e}")
        except RuntimeError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Unexpected error: {type(e).__name__}: {e}")

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print(f"Scanning for bear ({BLE_MAC})...", end=" ", flush=True)
    devices = await BleakScanner.discover(timeout=10)
    device  = next(
        (d for d in devices if d.address.upper() == BLE_MAC.upper()),
        None)
    if not device:
        print("not found.")
        sys.exit(1)
    print(f"found {device.address} ({device.name or 'no name'})")

    async with BleakClient(device) as client:
        bear = Bear(client)
        await bear.connect()
        print("Connected. Type 'help' for commands.\n")
        await repl(bear)

    print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
