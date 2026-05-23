#!/usr/bin/env python3
"""
loca-whisper-diagnose.py — Record utterances and show Whisper confidence metrics.

Helps calibrate no_speech_threshold and logprob_threshold in config.yaml.
Loops continuously; press Ctrl-C to exit.

Usage:
    ./loca-whisper-diagnose.py
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

import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from faster_whisper import WhisperModel
from config import CFG

# ── Config — same values loca-talk.py uses ────────────────────────────────────

AUDIO_SOURCE    = CFG.get("audio", {}).get("input", {}).get("source", "")
MIC_RATE        = 16000
MIC_CHANNELS    = 1

WHISPER_MODEL   = CFG["stt"]["whisper_model"]
WHISPER_DEVICE  = CFG["stt"]["whisper_device"]
WHISPER_COMPUTE = CFG["stt"]["whisper_compute"]

NO_SPEECH_THRESHOLD = CFG["stt"].get("no_speech_threshold", 0.6)
LOGPROB_THRESHOLD   = CFG["stt"].get("logprob_threshold", -1.0)

SILENCE_SECONDS = CFG["vad"]["silence_seconds"]
MIN_SPEECH_SECS = CFG["vad"]["min_speech_secs"]
MAX_RECORD_SECS = CFG["vad"]["max_record_secs"]
NOISE_HEADROOM  = CFG["vad"]["noise_headroom"]

# ── Noise floor calibration ───────────────────────────────────────────────────

def calibrate_noise_floor(seconds: float = 1.0) -> float:
    chunk_frames = int(MIC_RATE * 0.05)
    chunk_bytes  = chunk_frames * MIC_CHANNELS * 2
    num_chunks   = int(seconds / 0.05)
    proc = subprocess.Popen(
        ["pw-record", f"--target={AUDIO_SOURCE}", f"--channels={MIC_CHANNELS}",
         f"--rate={MIC_RATE}", "--format=s16", "--latency=50", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rms_values = []
    try:
        for _ in range(num_chunks):
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk or len(chunk) < chunk_bytes:
                break
            samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
            rms_values.append(math.sqrt(sum(s*s for s in samples) / len(samples)))
    finally:
        proc.terminate()
        proc.wait()
    baseline = sum(rms_values) / len(rms_values) if rms_values else 300.0
    return baseline + NOISE_HEADROOM

# ── Recording ─────────────────────────────────────────────────────────────────

def record_utterance(threshold: float) -> bytes | None:
    chunk_secs   = 0.05
    chunk_frames = int(MIC_RATE * chunk_secs)
    chunk_bytes  = chunk_frames * MIC_CHANNELS * 2
    silence_needed = int(SILENCE_SECONDS / chunk_secs)
    max_chunks     = int(MAX_RECORD_SECS / chunk_secs)

    proc = subprocess.Popen(
        ["pw-record", f"--target={AUDIO_SOURCE}", f"--channels={MIC_CHANNELS}",
         f"--rate={MIC_RATE}", "--format=s16", "--latency=50", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames        = []
    silent_chunks = 0
    recording     = False
    total_chunks  = 0

    print("  [listening...]", end="", flush=True)
    try:
        while total_chunks < max_chunks:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk or len(chunk) < chunk_bytes:
                break
            total_chunks += 1
            samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            if rms > threshold:
                if not recording:
                    print(f"\n  [recording, RMS={rms:.0f}]", end="", flush=True)
                    recording = True
                silent_chunks = 0
                frames.append(chunk)
            elif recording:
                silent_chunks += 1
                frames.append(chunk)
                if silent_chunks >= silence_needed:
                    break
    finally:
        proc.terminate()
        proc.wait()

    print()
    if not frames:
        return None
    pcm = b"".join(frames)
    if len(pcm) / (MIC_RATE * MIC_CHANNELS * 2) < MIN_SPEECH_SECS:
        return None
    return pcm

# ── Transcribe and print metrics ──────────────────────────────────────────────

def analyse(pcm: bytes, model: WhisperModel) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(MIC_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(MIC_RATE)
            wf.writeframes(pcm)
        segments, info = model.transcribe(wav_path, language="en", vad_filter=True)
        segments = list(segments)
    finally:
        Path(wav_path).unlink(missing_ok=True)

    if not segments:
        print("  (no segments returned)")
        return

    for i, seg in enumerate(segments):
        passes = (seg.no_speech_prob <= NO_SPEECH_THRESHOLD
                  and seg.avg_logprob >= LOGPROB_THRESHOLD)
        verdict = "PASS" if passes else "DROP"
        print(f"  [{verdict}] \"{seg.text.strip()}\"")
        print(f"         no_speech_prob={seg.no_speech_prob:.3f}  "
              f"(threshold: <={NO_SPEECH_THRESHOLD})")
        print(f"         avg_logprob   ={seg.avg_logprob:.3f}  "
              f"(threshold: >={LOGPROB_THRESHOLD})")

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not AUDIO_SOURCE:
        print("No audio.input.source configured.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Whisper ({WHISPER_MODEL})...", end=" ", flush=True)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                         compute_type=WHISPER_COMPUTE)
    print("ready.")

    print("Calibrating noise floor...", end=" ", flush=True)
    threshold = calibrate_noise_floor()
    print(f"threshold={threshold:.0f}")

    print(f"\nCurrent thresholds from config:")
    print(f"  no_speech_threshold : {NO_SPEECH_THRESHOLD}")
    print(f"  logprob_threshold   : {LOGPROB_THRESHOLD}")
    print("\nMake a sound after each [listening...] prompt. Ctrl-C to exit.\n")

    while True:
        try:
            pcm = record_utterance(threshold)
            if pcm is None:
                print("  (nothing recorded)")
                continue
            analyse(pcm, model)
            print()
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()
