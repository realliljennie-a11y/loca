#!/usr/bin/env python3
"""
loca-quiz.py — Quiz mode via BLE lip-sync.

Loads a YAML quiz file, speaks questions and feedback through the device.
No microphone — entirely keyboard driven.

Usage:
    ./loca-quiz.py quiz.yaml
    ./loca-quiz.py quiz.yaml --random

Controls:
  Q          Speak the question
  A          Reveal answer on screen and speak it
  Y          Speak "correct" response and advance to next question
  N          Speak "wrong" response and hide answer (for retry)
  Space/Enter  Advance to next question without feedback
  R          Speak a random remark from the quiz file
  Ctrl-C     Quit
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
import random as _random
import re
import secrets
import struct
import subprocess
import sys
import termios
import tty

import numpy as np
import yaml
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

    async def upload(self, audio_data: bytes, title: str) -> None:
        chunks = [audio_data[i:i+AUDIO_CHUNK]
                  for i in range(0, len(audio_data), AUDIO_CHUNK)]
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            packet  = make_upload_packet(chunk, title, i, total)
            subpkts = split_for_mtu(packet)
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
                    ack = {}
                ack_status = ack.get("status", -1)
                if ack_status == 0:
                    break
                elif ack_status == 2 and retries < MAX_RETRY:
                    retries += 1
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

_pipeline: KPipeline | None = None

def get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _pipeline

def tts_to_mp3(text: str,
               voice: str = DEFAULT_VOICE,
               speed: float = SPEED) -> tuple[bytes, float]:
    pipeline = get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    pcm_float = np.concatenate(chunks)
    pcm_int16 = (np.clip(pcm_float, -1.0, 1.0) * 32767).astype(np.int16)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
         "-af", f"apad=pad_dur={PAD_DURATION}",
         "-f", "mp3", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mp3, err = ffmpeg.communicate(input=pcm_int16.tobytes())
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')}")
    return mp3, len(pcm_float) / 24000 + PAD_DURATION

async def bear_say(bear: Bear, text: str,
                   voice: str = DEFAULT_VOICE,
                   speed: float = SPEED) -> None:
    """TTS → MP3 → upload → play → wait → delete."""
    mp3, duration = tts_to_mp3(text, voice=voice, speed=speed)
    fid = new_file_id()
    await bear.upload(mp3, fid)
    await bear.play(fid)
    await asyncio.sleep(duration)
    await bear.delete_and_wait(fid)

# ── Keyboard ──────────────────────────────────────────────────────────────────

def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

async def get_key() -> str:
    return await asyncio.get_event_loop().run_in_executor(None, _read_key)

# ── Quiz display ──────────────────────────────────────────────────────────────

def render(quiz: dict, questions: list, idx: int, total: int) -> None:
    q = questions[idx]
    print("\033[2J\033[H", end="")
    print(f"=== {quiz['topic']} ===  ({idx + 1}/{total})\n")
    print(f"  Q: {q['q']}")
    print(f"  A: {q['a']}\n")
    print("  Q=question  A=answer  Y=correct  N=wrong  Space=next  Backspace=prev  R=remark  Ctrl-C=quit")
    sys.stdout.flush()

# ── Quiz loop ─────────────────────────────────────────────────────────────────

async def quiz_loop(bear: Bear, quiz: dict, args) -> None:
    questions = list(quiz["questions"])
    if args.random:
        _random.shuffle(questions)

    randoms      = quiz.get("random", [])
    total        = len(questions)
    idx          = 0
    number_said  = False
    voice        = args.voice
    speed        = args.speed

    async def speak(text: str) -> None:
        try:
            await bear_say(bear, text, voice=voice, speed=speed)
        except Exception as e:
            print(f"\n  [speak error: {e}]")

    intro = f"Welcome to the quiz! Tonight's topic is {quiz['topic']}. {quiz.get('explanation', '')}"

    # Wait for Q before speaking the introduction
    render(quiz, questions, idx, total)
    print("\n  (press Q to begin)")
    sys.stdout.flush()
    while True:
        key = (await get_key()).lower()
        if key in ("\x03", "\x04"):
            print("\nQuitting.")
            return
        if key == "q":
            await speak(intro.strip())
            break

    while True:
        render(quiz, questions, idx, total)
        key = (await get_key()).lower()

        if key in ("\x03", "\x04"):      # Ctrl-C / Ctrl-D
            print("\nQuitting.")
            break

        q = questions[idx]

        if key == "q":
            if not number_said:
                await speak(f"Number {idx + 1}. {q['q']}")
                number_said = True
            else:
                await speak(q["q"])

        elif key == "a":
            await speak(f"The correct answer is: {q['a']}.")

        elif key == "y":
            await speak(quiz["right"])

        elif key == "n":
            await speak(quiz["wrong"])

        elif key in (" ", "\r", "\n"):
            idx += 1
            if idx >= total:
                closing = quiz.get("closing", "")
                if closing:
                    await speak(closing)
                print("\n\n  Quiz complete!\n")
                break
            number_said = False
            render(quiz, questions, idx, total)

        elif key in ("\x7f", "\x08"):    # Backspace / Delete
            if idx > 0:
                idx -= 1
                number_said = False
                render(quiz, questions, idx, total)

        elif key == "r" and randoms:
            await speak(_random.choice(randoms))

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    p = argparse.ArgumentParser(
        prog="loca-quiz",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("quiz", help="Path to quiz YAML file")
    p.add_argument("--random", action="store_true",
        help="Shuffle questions")
    p.add_argument("--voice", default=DEFAULT_VOICE, metavar="NAME",
        help=f"Kokoro voice name (default: {DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=SPEED, metavar="F",
        help=f"TTS speed (default: {SPEED})")
    args = p.parse_args()

    quiz = yaml.safe_load(open(args.quiz))

    print("Loading TTS model...", end=" ", flush=True)
    get_pipeline()
    print("ready.")

    print(f"Scanning for bear ({BLE_MAC})...", end=" ", flush=True)
    devices = await BleakScanner.discover(timeout=10)
    device  = next(
        (d for d in devices if d.address.upper() == BLE_MAC.upper()), None)
    if not device:
        print("not found.")
        sys.exit(1)
    print("found.")

    async with BleakClient(device) as client:
        bear = Bear(client)
        await bear.connect()
        print("Connected.\n")
        await quiz_loop(bear, quiz, args)

    print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
