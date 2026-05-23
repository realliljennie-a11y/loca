# LOCA — Local Online Conversational Assistant

A fully offline, voice-driven AI companion built around a Bluetooth HFP device (speaker + microphone). The device listens through its microphone, processes speech locally, generates a response via a local or cloud LLM, and speaks back through its speaker. No cloud services are required for the core pipeline after initial model downloads.

The companion's name, persona, voice, and language are fully configurable. The default configuration in `config.yaml` provides a neutral starting point; your persona lives in `hfp_secret.yaml`, which is never committed.

---

## Hardware

- **Ubuntu desktop** (tested on Intel Core i9-9900K, 32GB RAM, AMD Radeon RX 580)
- **Any Bluetooth Classic HFP device** with a microphone — a Bluetooth headset, a smart speaker with HFP support, or a novelty device with an embedded speaker and microphone
- **Home Assistant** instance on the local network (optional, for sensor context and event triggers)

### BLE lip-sync (optional)

`loca-say.py` and `loca-shell.py` implement a separate BLE upload path for devices based on the **Jieli AC6966** SoC (common in novelty talking toys). This path uploads MP3 audio to the device's internal flash and triggers lip-sync mouth movement during playback. It is not used for the main conversation pipeline. See [BLE protocol notes](#ble-protocol-notes) below.

---

## Architecture

```
Ubuntu desktop
├── Ollama            — local LLM inference (systemd service)
├── loca-watcher      — Bluetooth lifecycle manager (systemd user service)
│    └── watches for device BT connection, sets HFP profile,
│        boosts mic gain, starts/stops loca-talk
├── loca-talk.py      — main conversational pipeline
│    ├── pw-record    — mic input from device (HFP mSBC)
│    ├── faster-whisper — speech-to-text
│    ├── Ollama / Claude / OpenRouter — LLM response generation
│    ├── Kokoro TTS   — text-to-speech
│    └── pw-play      — audio output to device (HFP mSBC)
├── loca-shell.py     — interactive BLE testing shell
│    └── BLE protocol — upload MP3s to device flash,
│                       trigger playback with mouth movement
└── loca-say.py       — one-shot TTS → BLE upload → play
```

Classic Bluetooth (HFP mSBC) is used for real-time voice conversation. BLE is used separately for uploading audio files to the device's internal flash storage, which triggers lip-sync mouth movement during playback.

---

## Dependencies

### System packages
```bash
sudo apt install pipewire pipewire-pulse wireplumber \
                 bluez bluetooth \
                 ffmpeg \
                 espeak-ng espeak-ng-data \
                 python3-dev
```

`espeak-ng` is required by the Kokoro TTS engine. `ffmpeg` is required for MP3 encoding.

### Python

`make install` creates the venv and installs all required packages automatically.

The venv is created at `.venv/` in the project root by default. Override with the `LOCA_VENV` environment variable if you prefer a different location.

### Ollama
`make install` installs Ollama automatically if not already present.

### TTS voice (Kokoro)
After `make install`, run:
```bash
make voices
```
This downloads and caches the voice set in `config.yaml` (`tts.voice`) from HuggingFace into `~/.cache/huggingface/`. Requires `espeak-ng` and an internet connection on first run.

### LLM model

LOCA uses a local Ollama model by default. You can use any model Ollama supports, or a custom fine-tuned GGUF file.

**Option A — standard Ollama model** (e.g. Mistral):
```bash
ollama pull mistral
```
Then set `FROM mistral` in `Modelfile`, set `ollama.model: mistral` in `hfp_secret.yaml`, and run `make model`.

**Option B — custom GGUF** (e.g. a fine-tuned personality model):
1. Place the `.gguf` file in the project root directory
2. Set `FROM your-model.gguf` in `Modelfile`
3. Set `ollama.model: your-model-name` in `hfp_secret.yaml`
4. Run `make model`

---

## Configuration

Configuration is split across several files — copy each `.example` file and fill in your values:

| File | Purpose | Committed? |
|---|---|---|
| `config.yaml` | AI/conversation settings — LLM, TTS, STT, VAD, memory, triggers defaults | Yes |
| `audio_device.yaml` | Audio hardware — PipeWire sink/source names, BT MAC, HFP profile | No |
| `ble_device.yaml` | BLE MAC address for the Jieli upload path | No |
| `hfp_secret.yaml` | HA credentials, API keys, persona (name, pronouns, system prompts) | No |

```bash
cp config.yaml.example config.yaml
cp audio_device.yaml.example audio_device.yaml
cp ble_device.yaml.example ble_device.yaml
cp hfp_secret.yaml.example hfp_secret.yaml
```

Edit `audio_device.yaml` to set your device's PipeWire sink/source names and Bluetooth MAC. Edit `hfp_secret.yaml` to set your persona, HA credentials, and (if used) cloud API keys.

### Finding your PipeWire device names

With your device connected in HFP mode:
```bash
pactl list sinks short    # find output sink name
pactl list sources short  # find input source name
pactl list cards short    # find card name (for profile switching)
```

---

## Ollama model setup

Create or edit `Modelfile` in the project directory, then:

```bash
make model
```

**NVIDIA:** Ollama detects CUDA automatically — no configuration needed beyond having the NVIDIA drivers installed.

**AMD** (tested on RX 580): Ollama uses Vulkan for AMD cards. Enable it via the systemd override:

```bash
sudo systemctl edit ollama
```

Add:
```ini
[Service]
Environment="OLLAMA_VULKAN=1"
Environment="VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json"
```

Then restart: `sudo systemctl restart ollama`. Confirm GPU is active with `ollama ps` — the `PROCESSOR` column should show `100% GPU`. Note: `HSA_OVERRIDE_GFX_VERSION` is a ROCm variable and is not needed for Vulkan; setting it causes a warning that may interfere with GPU discovery.

---

## Bluetooth setup

### First-time pairing

```bash
make pair
```

Or manually via the GNOME Bluetooth settings, or:
```bash
bluetoothctl
scan on
pair <MAC>
trust <MAC>
connect <MAC>
quit
```

After pairing, BlueZ will reconnect automatically on power-on. `loca-watcher` handles the HFP profile switch and mic volume boost.

### Enable Bluetooth auto-connect on boot

Add to `/etc/bluetooth/main.conf`:

```ini
[Policy]
AutoEnable=true
ReconnectAttempts=7
ReconnectIntervals=1,2,4,8,16,32,64
```

---

## Installation

```bash
# 1. Copy and edit configuration files
cp config.yaml.example config.yaml
cp audio_device.yaml.example audio_device.yaml      # fill in device names and MAC
cp ble_device.yaml.example ble_device.yaml           # fill in BLE MAC (Jieli path only)
cp hfp_secret.yaml.example hfp_secret.yaml           # fill in persona, HA, API keys

# 2. Set up Modelfile (see LLM model section above)

# 3. Run the installer (creates .venv, installs packages, configures BT, installs services)
make install

# 4. Cache the TTS voice (requires espeak-ng and internet)
make voices

# 5. Pair the device
make pair
```

After pairing, the service starts automatically whenever the device is powered on.

---

## Usage

### Automatic (normal use)

1. Power on the device
2. Wait ~10 seconds for Bluetooth to connect and `loca-talk` to start
3. Speak into the device — the assistant will respond through its speaker
4. Power off the device to end the session

### Manual testing

```bash
# Connect device manually if needed
./loca-connect.sh connect

# Full voice conversation
./loca-talk.py --greet

# Type instead of speak (useful for testing without mic)
./loca-talk.py --type --greet

# Use a specific LLM backend
./loca-talk.py --llm ollama
./loca-talk.py --llm claude
./loca-talk.py --llm openrouter

# With timing information
./loca-talk.py --verbose

# Interactive BLE shell (for testing file upload / mouth movement)
./loca-shell.py

# One-shot TTS to device speaker via HFP
echo "Hello!" | ./loca-hfp-test.py

# One-shot TTS → BLE upload → play (Jieli devices only)
echo "Hello!" | ./loca-say.py
```

### Service management

```bash
# Check status
systemctl --user status loca-watcher
systemctl --user status loca-talk

# View logs
journalctl --user -u loca-talk -f
journalctl --user -u loca-watcher -f

# Restart
systemctl --user restart loca-watcher

# Manual Bluetooth management
./loca-connect.sh connect
./loca-connect.sh disconnect
./loca-connect.sh status
./loca-connect.sh wake      # re-activate HFP transport if audio stalls
```

---

## Files

| File | Description |
|---|---|
| `loca-talk.py` | Main conversational pipeline — STT → LLM → TTS |
| `loca-shell.py` | Interactive BLE shell for testing the device's GATT protocol |
| `loca-say.py` | One-shot: text → TTS → BLE upload → play (with lip sync, Jieli only) |
| `loca-hfp-test.py` | One-shot: text → TTS → HFP playback (for testing and diagnostics) |
| `loca-quiz.py` | Quiz mode via BLE lip-sync, keyboard driven |
| `loca-whisper-diagnose.py` | Diagnostic tool: record utterances and display Whisper confidence metrics |
| `loca-connect.sh` | Bluetooth connection management script |
| `loca-watcher.sh` | Systemd service script: watches for device, manages lifecycle |
| `loca-watcher.service` | Systemd user unit for the watcher |
| `loca-talk.service` | Systemd user unit for the conversational pipeline |
| `Modelfile` | Ollama model configuration |
| `jieli-notes.txt` | Reverse-engineering notes for the Jieli AC6966 BLE protocol |

---

## BLE protocol notes

The Jieli AC6966 BLE GATT service (UUID `0000AE30-...`) uses a proprietary
chunked upload protocol reverse-engineered from the decompiled Android app.
Audio files must be named with the firmware's convention (`U-<19 base64 chars>`)
to be indexed correctly. The protocol uses CRC-16/CCITT framing with a
`JL` header. See `jieli-notes.txt` for details.

Key characteristics:

| UUID | Direction | Purpose |
|---|---|---|
| `0000AE03` | write | JSON commands to device |
| `0000AE04` | notify | JSON responses from device |
| `0000AE02` | write | Binary audio upload |
| `0000AE01` | notify | Upload ACKs |

Audio is uploaded as plain MP3, encoded CBR at 64kbps with 0.3s of silence
padding at the end to prevent the firmware's decoder from clipping the final
frames.

---

## Home Assistant integration

The assistant is aware of conditions fetched from Home Assistant before each
LLM response. Configure sensor entity IDs in `hfp_secret.yaml`:

```yaml
triggers:
  illuminance_entity: sensor.your_illuminance_sensor
  sensors:
    - entity: binary_sensor.your_sensor
      watch_state: on
      event_text: "The sensor just turned on."
```

Set `ha_url` and `ha_token` in `hfp_secret.yaml`. The token is a long-lived
access token created in your Home Assistant profile settings.

### OVOS / other voice assistant coexistence

If another voice assistant (OVOS, Alexa, etc.) shares the same audio device,
set `triggers.ovos_speaking_entity` in `hfp_secret.yaml` to a binary sensor
that is `on` while that assistant is speaking. LOCA will mute its microphone
input during that window so it does not attempt to respond to the other
assistant's output.

---

## Known issues and limitations

- The HFP audio transport occasionally goes stale and requires a profile
  toggle to wake (`./loca-connect.sh wake`). This is a Linux/BlueZ/PipeWire
  interaction issue, not specific to this project.
- Background noise affects VAD sensitivity. The noise floor is calibrated
  on startup; if conditions change significantly, restart `loca-talk`.
- The Jieli BLE upload pipeline has a ~6 second storage clear delay before
  each upload, making it unsuitable for low-latency responses. `loca-talk.py`
  uses direct HFP audio streaming instead.
