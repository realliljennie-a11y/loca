#!/usr/bin/env python3
"""
loca-hfp-test.py — Speak a phrase through the HFP sink at maximum volume.

Useful for testing whether HFP audio triggers the bear's mouth animation.

Usage:
    echo "Hello, this is a test." | ./loca-hfp-test.py
    ./loca-hfp-test.py              # reads from stdin; type phrase, then Ctrl-D

Safe to run alongside loca-talk — this writes to the HFP sink (speaker)
while loca-talk reads from the HFP source (mic); they don't conflict.
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

import subprocess
import sys

import numpy as np
from kokoro import KPipeline
from config import CFG

AUDIO_SINK = CFG.get("audio", {}).get("output", {}).get("sink", "")
VOICE      = CFG["tts"]["voice"]
SPEED      = CFG["tts"]["speed"]
RATE       = 24000

def main():
    phrase = sys.stdin.read().strip()
    if not phrase:
        print("No input.", file=sys.stderr)
        sys.exit(1)

    if not AUDIO_SINK:
        print("No audio.output.sink configured.", file=sys.stderr)
        sys.exit(1)

    print(f"Phrase : {phrase!r}", file=sys.stderr)
    print(f"Sink   : {AUDIO_SINK}", file=sys.stderr)

    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    chunks = [audio for _, _, audio in pipeline(phrase, voice=VOICE, speed=SPEED)]
    if not chunks:
        print("TTS produced no audio.", file=sys.stderr)
        sys.exit(1)

    pcm_float = np.concatenate(chunks)

    # Normalize to peak so output is always at maximum digital volume
    peak = np.abs(pcm_float).max()
    if peak > 0:
        pcm_float = pcm_float / peak

    pcm_int16 = (pcm_float * 32767).astype(np.int16)

    # Leading silence avoids HFP transport clipping the first syllable
    silence   = np.zeros(int(RATE * 0.15), dtype=np.int16)
    pcm_bytes = np.concatenate([silence, pcm_int16]).tobytes()

    pw = subprocess.Popen(
        ["pw-play",
         f"--target={AUDIO_SINK}",
         "--channels=1",
         f"--rate={RATE}",
         "--format=s16",
         "-"],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pw.communicate(input=pcm_bytes)
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
