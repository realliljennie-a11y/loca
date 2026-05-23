# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# First-time setup (after editing config.yaml, secret.yaml, Modelfile)
make install

# Download recommended Piper TTS voices
make voices

# (Re)build the Ollama model from Modelfile
make model

# Check system state (BT, audio, Ollama, services, voices)
make status

# Pair device over Bluetooth
make pair

# Manual testing without the device physically present
./loca-talk.py --type --greet        # type instead of speak
./loca-talk.py --verbose             # show timing info

# One-shot TTS → BLE upload → lip-sync playback
echo "Hello." | ./loca-say.py --verbose

# Interactive BLE protocol shell
./loca-shell.py

# Service logs
journalctl --user -u loca-talk -f
journalctl --user -u loca-watcher -f

# Bluetooth connection management
./loca-connect.sh connect|disconnect|status|wake
```

All Python scripts self-select their venv at startup. Priority: `$LOCA_VENV` env var → `.venv/` in the project root → `~/hf-venv` legacy fallback. The Makefile default is `.venv/` in the project root.

## Architecture

The project has two independent audio paths:

**Classic BT HFP (loca-talk.py)** — real-time voice conversation pipeline:
- `pw-record` captures mic audio from the device's HFP source → simple RMS-based VAD → `faster-whisper` STT → Ollama streaming API → Piper TTS → `pw-play` to the device's HFP sink
- `loca-watcher.sh` (systemd user service) monitors BlueZ for the device's connection, sets the HFP mSBC profile, boosts mic gain to 300%, then starts/stops `loca-talk.service`

**BLE GATT (loca-say.py, loca-shell.py)** — file upload for lip-sync:
- Reverse-engineered Jieli AC6966 chunked upload protocol with CRC-16/CCITT framing (`JL` header)
- Files must be named `U-<19 base64 chars>` to be indexed by firmware
- Upload file, play, then delete by ID with `deleteRing` (~0.66s) — avoids `ClearAll` which would wipe app-stored stories
- Still not suitable for low-latency responses due to upload time
- `loca-talk.py` uses HFP streaming instead of BLE for this reason

## Configuration

`config.py` merges two YAML files at import time into the `CFG` singleton:
- `config.yaml` — all non-secret settings (copy from `config.yaml.example`)
- `secret.yaml` — `ha_url` and `ha_token` for Home Assistant (copy from `secret.yaml.example`)

Key values to set before first run:
- `bear.mac` — Classic BT MAC (for HFP audio and `pactl`)
- `bear.ble_mac` — BLE MAC (typically Classic MAC + 1 on last octet)
- `bear.sink` / `bear.source` / `bear.card` — PipeWire device names (colons → underscores in MAC)

## BLE protocol details

GATT service UUID prefix: `0000AE30-...`

| Characteristic | Direction | Purpose |
|---|---|---|
| `0000AE01` | notify | Upload ACKs |
| `0000AE02` | write | Binary audio upload chunks |
| `0000AE03` | write | JSON commands to device |
| `0000AE04` | notify | JSON responses from device |

Packet framing: `4A 4C` (JL header) + 2-byte big-endian length + type bytes + payload + 2-byte CRC-16/CCITT + `FF` tail. MTU is split at 509 bytes. Audio chunks are 4096 bytes each.

## Home Assistant integration

`loca-talk.py:get_ha_context()` fetches two sensors before each LLM call and prepends them to the system prompt. Sensor entity IDs are hardcoded in that function. Add sensors there to expand context.
