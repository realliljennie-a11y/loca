#!/usr/bin/env python3
"""
loca-say.py — Pipe text to device speaker via BLE upload.

Reads text from stdin, renders with Kokoro TTS, encodes to MP3,
uploads over BLE, plays it (moving the device's mouth if applicable), then cleans up.

Usage:
    echo "Hello there!" | ./loca-say.py
    ./loca-say.py --voice af_heart --verbose

Requires:
    pip install kokoro soundfile bleak pyyaml
    ffmpeg
    sudo apt install espeak-ng
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
import argparse
import base64
import json
import re
import secrets
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

# ── GATT UUIDs ────────────────────────────────────────────────────────────────

ACK_CHAR = "0000AE01-0000-1000-8000-00805f9b34fb"
UPL_CHAR = "0000AE02-0000-1000-8000-00805f9b34fb"
CMD_CHAR = "0000AE03-0000-1000-8000-00805f9b34fb"
ANS_CHAR = "0000AE04-0000-1000-8000-00805f9b34fb"

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

# ── CRC-16/CCITT ──────────────────────────────────────────────────────────────

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
        self.client  = client
        self.ans_buf = NotifBuffer()
        self.ack_buf = NotifBuffer()
        self._ans_q  = asyncio.Queue()
        self._ack_q  = asyncio.Queue()

    async def connect(self):
        await self.client.start_notify(ANS_CHAR, self._on_ans)
        await self.client.start_notify(ACK_CHAR, self._on_ack)
        # Optional extra channels — failures are non-fatal
        for uuid, name in [
            ("0000FEE5-0000-1000-8000-00805f9b34fb", "FEE5"),
            ("0000ff62-0000-1000-8000-00805f9b34fb", "FF62"),
        ]:
            try:
                await self.client.start_notify(uuid, self._on_extra)
            except Exception:
                pass

    def _on_extra(self, sender, data):
        pass   # swallow silently in loca-say

    def _on_ans(self, sender, data):
        payload = self.ans_buf.feed(bytes(data))
        if payload is not None:
            self._ans_q.put_nowait(payload)

    def _on_ack(self, sender, data):
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

    async def clear_all_and_wait(self, timeout: float = 15.0) -> None:
        while not self._ans_q.empty():
            self._ans_q.get_nowait()
        pkt = make_cmd('{"cmd":"ClearAll","parm":"0"}')
        await self.client.write_gatt_char(CMD_CHAR, pkt, response=True)
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("ClearAll did not complete within timeout")
            try:
                raw = await asyncio.wait_for(
                    self._ans_q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError("ClearAll did not complete within timeout")
            try:
                resp = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            status = str(resp.get("status", ""))
            if status == STATUS_OK:
                return
            elif status == STATUS_PENDING:
                continue
            else:
                raise RuntimeError(f"ClearAll failed: {resp}")

    async def upload(self, audio_data: bytes, title: str,
                     verbose: bool = False) -> None:
        chunks = [audio_data[i:i+AUDIO_CHUNK]
                  for i in range(0, len(audio_data), AUDIO_CHUNK)]
        total = len(chunks)
        if verbose:
            print(f"  Uploading {len(audio_data):,} bytes "
                  f"({total} chunk(s))...", file=sys.stderr)
        for i, chunk in enumerate(chunks):
            packet  = make_upload_packet(chunk, title, i, total)
            subpkts = split_for_mtu(packet)
            if verbose:
                pack_label = ("single" if total == 1 else
                              "first"  if i == 0 else
                              "last"   if i == total - 1 else "middle")
                print(f"  Chunk {i}: {len(chunk)} bytes "
                      f"[{pack_label}]", file=sys.stderr)
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
                if verbose:
                    print(f"  Chunk {i} ACK: {ack}", file=sys.stderr)
                ack_status = ack.get("status", -1)
                if ack_status == 0:
                    break
                elif ack_status == 2 and retries < MAX_RETRY:
                    retries += 1
                    print(f"  Chunk {i+1} NAK — retry {retries}/{MAX_RETRY}",
                          file=sys.stderr)
                else:
                    raise RuntimeError(
                        f"Upload failed at chunk {i+1}/{total}: {ack}")

    async def play(self, file_id: str) -> None:
        resp = await self._send_raw(
            f'{{"cmd":"try","parm":"{strip_ext(file_id)}"}}')
        if str(resp.get("status", "")) != STATUS_OK:
            raise RuntimeError(f"Play failed: {resp}")

    async def delete_and_wait(self, file_id: str,
                              timeout: float = 15.0) -> None:
        """Delete a file using deleteRing, polling until STATUS_OK."""
        fid = strip_ext(file_id)
        while not self._ans_q.empty():
            self._ans_q.get_nowait()
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
                return
            elif status == STATUS_PENDING:
                continue
            else:
                raise RuntimeError(f"deleteRing failed: {resp}")

# ── TTS ───────────────────────────────────────────────────────────────────────

def tts_to_mp3(text: str, pipeline: KPipeline,
               voice: str = DEFAULT_VOICE,
               speed: float = SPEED) -> tuple[bytes, float]:
    """Render text to MP3 via Kokoro → ffmpeg pipe.
    Returns (mp3_bytes, duration_seconds) including pad silence."""
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
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    mp3, ffmpeg_err = ffmpeg.communicate(input=pcm_int16.tobytes())
    if ffmpeg.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{ffmpeg_err.decode(errors='replace')}")

    duration = len(pcm_float) / 24000 + PAD_DURATION
    return mp3, duration

# ── Bear discovery ────────────────────────────────────────────────────────────

async def find_bear(timeout: float = 10.0):
    """Find bear by MAC address from config."""
    print(f"Scanning for bear ({BLE_MAC})...", file=sys.stderr)
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        if d.address.upper() == BLE_MAC.upper():
            print(f"Found: {d.address} ({d.name or 'no name'})",
                  file=sys.stderr)
            return d
    return None

# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(text: str, args):
    t0 = time.monotonic()
    if args.verbose:
        print(f"Voice:  {args.voice}", file=sys.stderr)
        print(f"Text:   {text!r}", file=sys.stderr)
        print("Loading TTS model...", file=sys.stderr)

    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    if args.verbose:
        print("Rendering TTS...", file=sys.stderr)

    mp3_bytes, duration = tts_to_mp3(
        text, pipeline,
        voice=args.voice,
        speed=args.speed,
    )
    tts_time = time.monotonic() - t0

    if args.verbose:
        print(f"  {tts_time:.2f}s → {len(mp3_bytes):,} bytes, "
              f"{duration:.2f}s audio", file=sys.stderr)

    if args.dump_mp3:
        with open(args.dump_mp3, "wb") as f:
            f.write(mp3_bytes)
        print(f"MP3 written to {args.dump_mp3}", file=sys.stderr)

    device = await find_bear()
    if not device:
        print("Bear not found. Is he powered on and nearby?", file=sys.stderr)
        sys.exit(1)

    file_id = new_file_id()

    async with BleakClient(device) as client:
        bear = Bear(client)
        await bear.connect()

        if args.verbose:
            print(f"Uploading as {file_id}...", file=sys.stderr)
        t2 = time.monotonic()
        await bear.upload(mp3_bytes, file_id, verbose=args.verbose)
        if args.verbose:
            print(f"  Upload: {time.monotonic() - t2:.2f}s", file=sys.stderr)

        if args.verbose:
            print(f"Playing ({duration:.2f}s + {args.tail_silence}s tail)...",
                  file=sys.stderr)
        await bear.play(file_id)
        await asyncio.sleep(duration + args.tail_silence)

        if not args.no_cleanup:
            if args.verbose:
                print("Deleting file...", file=sys.stderr)
            t3 = time.monotonic()
            await bear.delete_and_wait(file_id)
            if args.verbose:
                print(f"  Delete: {time.monotonic() - t3:.2f}s",
                      file=sys.stderr)

        if args.verbose:
            print(f"Done. Total: {time.monotonic() - t0:.2f}s", file=sys.stderr)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="loca-say",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--voice", "-V", default=DEFAULT_VOICE,
        metavar="NAME",
        help=f"Kokoro voice name (default: {DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=SPEED,
        metavar="F",
        help=f"TTS speed, <1.0 slower (default: {SPEED})")
    p.add_argument("--tail-silence", type=float, default=0.5,
        metavar="SECS",
        help="Extra seconds after estimated playback end (default: 0.5)")
    p.add_argument("--no-cleanup", action="store_true",
        help="Skip deleting the file after playback")
    p.add_argument("--dump-mp3", metavar="PATH",
        help="Write the encoded MP3 to this file for inspection")
    p.add_argument("--verbose", "-v", action="store_true",
        help="Print timing and progress information")

    args = p.parse_args()

    text = sys.stdin.read().strip()
    if not text:
        print("No input text.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(text, args))

if __name__ == "__main__":
    main()
