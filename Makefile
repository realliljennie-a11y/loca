# Makefile — LOCA Project
#
# Usage:
#   make install      Full one-time setup (run once after cloning)
#   make uninstall    Reverse system changes (leaves voices/venv intact)
#   make pair         Interactively pair and trust device over Bluetooth
#   make unpair       Remove device from BlueZ device list
#   make status       Show current system state
#   make stop         Stop the services (loca-watcher and loca-talk)
#   make start        Start the services (loca-watcher -> loca-talk)
#   make restart      Restart the services (stop loca-talk; loca-watcher restarts)
#   make follow       Follow loca-talk's journal
#   make model        (Re)create the Ollama sunny model from Modelfile
#   make help         Show this help
#
# Before running 'make install':
#   1. Copy audio_device.yaml.example -> audio_device.yaml and fill in your device
#   2. Copy ble_device.yaml.example -> ble_device.yaml and fill in your BLE MAC
#   3. Copy hfp_secret.yaml.example -> hfp_secret.yaml and fill in ha_url and ha_token
#   4. Edit Modelfile — adjust LLM parameters if desired
#   5. Power on device and have it nearby for the pairing step

.PHONY: install uninstall pair unpair status stop start restart follow model training \
	clear_memory disable enable \
        _install-bt _uninstall-bt \
        _install-ollama _uninstall-ollama \
        _install-services _uninstall-services \
        _install-python _check-config voices help

SHELL    := /bin/bash
USER     := $(shell whoami)
PROJ_DIR := $(shell pwd)
VENV     := $(or $(LOCA_VENV),$(PROJ_DIR)/.venv)
PYTHON   := $(VENV)/bin/python3

# Read values from merged config (config.yaml + audio_device.yaml + ble_device.yaml)
CLASSIC_MAC  = $(shell $(VENV)/bin/python3 -c "import sys,os; sys.path.insert(0,'$(PROJ_DIR)'); os.chdir('$(PROJ_DIR)'); from config import CFG; a=CFG.get('audio',{}); print(a.get('output',{}).get('bt_mac') or a.get('input',{}).get('bt_mac',''))")
BLE_MAC      = $(shell $(VENV)/bin/python3 -c "import sys,os; sys.path.insert(0,'$(PROJ_DIR)'); os.chdir('$(PROJ_DIR)'); from config import CFG; print(CFG['ble']['mac'])")
BT_CARD      = $(shell $(VENV)/bin/python3 -c "import sys,os; sys.path.insert(0,'$(PROJ_DIR)'); os.chdir('$(PROJ_DIR)'); from config import CFG; print(CFG['audio']['input']['bt_card'])")
BT_PROFILE   = $(shell $(VENV)/bin/python3 -c "import sys,os; sys.path.insert(0,'$(PROJ_DIR)'); os.chdir('$(PROJ_DIR)'); from config import CFG; print(CFG['audio']['input']['bt_profile'])")

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "LOCA — setup targets"
	@echo ""
	@echo "  make install      Full one-time setup"
	@echo "  make uninstall    Reverse system changes"
	@echo "  make pair         Pair and trust device over Bluetooth"
	@echo "  make unpair       Remove device from BlueZ"
	@echo "  make status       Show current system state"
	@echo "  make stop         Stop loca-watcher and loca-talk"
	@echo "  make start        Start loca-watcher (loca-talk follows)"
	@echo "  make restart      Restart loca-talk only"
	@echo "  make follow       Follow loca-talk's journal"
	@echo "  make model        (Re)create Ollama sunny model"
	@echo "  make help         Show this help"
	@echo ""
	@echo "Before 'make install', copy and edit:"
	@echo "  audio_device.yaml  — PipeWire device names and Bluetooth MACs"
	@echo "  ble_device.yaml    — BLE MAC for lip-sync upload"
	@echo "  hfp_secret.yaml    — HA URL and token"
	@echo "  Modelfile          — LLM parameters"
	@echo ""

# ── Config validation ─────────────────────────────────────────────────────────

_check-config:
	@echo "Checking configuration..."
	@if [ ! -f "$(PROJ_DIR)/audio_device.yaml" ]; then \
		echo "ERROR: audio_device.yaml not found."; \
		echo "       Copy audio_device.yaml.example to audio_device.yaml and fill it in."; \
		exit 1; \
	fi
	@if [ ! -f "$(PROJ_DIR)/ble_device.yaml" ]; then \
		echo "ERROR: ble_device.yaml not found."; \
		echo "       Copy ble_device.yaml.example to ble_device.yaml and fill it in."; \
		exit 1; \
	fi
	@$(VENV)/bin/python3 -c "\
import yaml, sys; \
a = yaml.safe_load(open('$(PROJ_DIR)/audio_device.yaml')) or {}; \
inp = a.get('audio', {}).get('input', {}); \
errors = []; \
[errors.append(f'audio.input.{k} is a placeholder — edit audio_device.yaml') \
 for k in ('source', 'bt_mac') \
 if 'YOUR_' in str(inp.get(k, 'YOUR_'))]; \
[sys.exit(f'ERROR: {e}') for e in errors]; \
print('  audio_device.yaml OK')" 2>&1 || exit 1
	@$(VENV)/bin/python3 -c "\
import yaml, sys; \
b = yaml.safe_load(open('$(PROJ_DIR)/ble_device.yaml')) or {}; \
mac = b.get('ble', {}).get('mac', 'YOUR'); \
sys.exit('ERROR: ble.mac is a placeholder — edit ble_device.yaml') if 'YOUR' in mac else None; \
print('  ble_device.yaml OK')" 2>&1 || exit 1
	@if [ ! -f "$(PROJ_DIR)/hfp_secret.yaml" ]; then \
		echo "  WARNING: hfp_secret.yaml not found — HA integration disabled."; \
		echo "           Copy hfp_secret.yaml.example to hfp_secret.yaml to enable."; \
	else \
		$(VENV)/bin/python3 -c "\
import yaml, sys; \
s = yaml.safe_load(open('$(PROJ_DIR)/hfp_secret.yaml')) or {}; \
errors = []; \
[errors.append(f'{k} is a placeholder — edit hfp_secret.yaml') \
 for k in ('ha_url','ha_token') \
 if 'your_' in str(s.get(k,'')).lower() or not s.get(k,'')]; \
[print(f'  WARNING: {e}') for e in errors]; \
print('  hfp_secret.yaml OK')" 2>&1; \
	fi
	@if [ ! -f "$(PROJ_DIR)/Modelfile" ]; then \
		echo "ERROR: Modelfile not found."; \
		exit 1; \
	fi
	@echo "  Modelfile OK"

# ── Top-level targets ─────────────────────────────────────────────────────────

install: _install-python _check-config _install-bt _install-ollama _install-services
	@echo ""
	@echo "================================================================"
	@echo "Installation complete!"
	@echo ""
	@echo "Remaining manual steps:"
	@echo "  1. Place your GGUF model file in $(PROJ_DIR)/"
	@echo "     Edit Modelfile if the filename differs, then run: make model"
	@echo "  2. Cache the TTS voice (needs espeak-ng installed first):"
	@echo "       sudo apt install espeak-ng espeak-ng-data"
	@echo "       make voices"
	@echo "  3. Run 'make pair' to pair the device over Bluetooth"
	@echo "  4. Start the service:"
	@echo "       systemctl --user start loca-watcher"
	@echo "     Or reboot — it starts automatically on login."
	@echo "================================================================"
	@echo ""

uninstall: _uninstall-services _uninstall-ollama _uninstall-bt
	@echo ""
	@echo "================================================================"
	@echo "Uninstall complete."
	@echo ""
	@echo "The following were NOT removed:"
	@echo "  $(VENV)               Python venv"
	@echo "  ~/.cache/huggingface           Kokoro and Whisper model caches"
	@echo ""
	@echo "To also remove those:"
	@echo "  rm -rf $(VENV)"
	@echo "  rm -rf ~/.cache/huggingface"
	@echo ""
	@echo "Device remains paired in BlueZ. To unpair: make unpair"
	@echo "================================================================"
	@echo ""

# ── Python environment ────────────────────────────────────────────────────────

_install-python:
	@echo "--- Python environment ---"
	@if [ ! -d "$(VENV)" ]; then \
		echo "  Creating venv at $(VENV)..."; \
		python3 -m venv $(VENV); \
	else \
		echo "  Venv already exists at $(VENV)"; \
	fi
	@echo "  Installing/updating packages..."
	@$(VENV)/bin/python3 -m pip install --quiet --upgrade pip
	@$(VENV)/bin/python3 -m pip install --quiet \
		kokoro soundfile bleak httpx faster-whisper \
		pyyaml sounddevice numpy websockets
	@echo "  Python environment ready."

# ── Bluetooth ─────────────────────────────────────────────────────────────────

_install-bt:
	@echo "--- Bluetooth configuration ---"
	@if grep -q "AutoEnable=true" /etc/bluetooth/main.conf 2>/dev/null; then \
		echo "  Bluetooth auto-connect already configured."; \
	else \
		echo "  Adding auto-connect policy to /etc/bluetooth/main.conf..."; \
		printf '\n[Policy]\nAutoEnable=true\nReconnectAttempts=7\nReconnectIntervals=1,2,4,8,16,32,64\n' \
			| sudo tee -a /etc/bluetooth/main.conf > /dev/null; \
		sudo systemctl restart bluetooth; \
		echo "  Bluetooth configured."; \
	fi

_uninstall-bt:
	@echo "--- Removing Bluetooth configuration ---"
	@if grep -q "AutoEnable=true" /etc/bluetooth/main.conf 2>/dev/null; then \
		echo "  Removing auto-connect policy..."; \
		sudo python3 -c "\
path = '/etc/bluetooth/main.conf'; \
lines = open(path).readlines(); \
skip = {'[Policy]','AutoEnable=true','ReconnectAttempts=7','ReconnectIntervals=1,2,4,8,16,32,64'}; \
out = [l for l in lines if l.strip() not in skip]; \
open(path,'w').writelines(out)"; \
		sudo systemctl restart bluetooth; \
		echo "  Bluetooth configuration restored."; \
	else \
		echo "  Nothing to remove from /etc/bluetooth/main.conf."; \
	fi

# ── Ollama ────────────────────────────────────────────────────────────────────

_install-ollama:
	@echo "--- Ollama ---"
	@if ! command -v ollama &>/dev/null; then \
		echo "  Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	else \
		echo "  Ollama already installed."; \
	fi
	@echo "  Configuring AMD GPU (Vulkan) override..."
	@if grep -q "0x1002" /sys/class/drm/card*/device/vendor 2>/dev/null; then \
		sudo mkdir -p /etc/systemd/system/ollama.service.d; \
		printf '[Service]\nEnvironment="OLLAMA_VULKAN=1"\nEnvironment="VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json"\n' \
			| sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null; \
		sudo systemctl daemon-reload; \
		echo "  AMD GPU Vulkan override applied."; \
		if [ ! -f /etc/udev/rules.d/99-amdgpu-fan.rules ]; then \
			sudo cp $(PROJ_DIR)/99-amdgpu-fan.rules /etc/udev/rules.d/; \
			sudo udevadm control --reload-rules; \
			sudo udevadm trigger --subsystem-match=hwmon; \
			echo "  AMD GPU fan control rule installed."; \
		else \
			echo "  AMD GPU fan control rule already present."; \
		fi; \
	else \
		echo "  No AMD GPU detected — skipping GPU override."; \
	fi
	@sudo systemctl enable --now ollama
	@echo "  Creating/updating sunny model..."
	@$(MAKE) model

model:
	@echo "  Building Ollama model from Modelfile..."
	@ollama create sunny -f $(PROJ_DIR)/Modelfile
	@echo "  Model ready. Verify with: ollama show sunny"

_uninstall-ollama:
	@echo "--- Ollama ---"
	@echo "  Removing sunny model..."
	@ollama rm sunny 2>/dev/null && echo "  Removed." || echo "  Model not found."
	@if [ -f /etc/systemd/system/ollama.service.d/override.conf ]; then \
		echo "  Removing AMD GPU override..."; \
		sudo rm /etc/systemd/system/ollama.service.d/override.conf; \
		sudo rmdir /etc/systemd/system/ollama.service.d 2>/dev/null || true; \
		sudo systemctl daemon-reload; \
		echo "  Override removed."; \
	fi
	@if [ -f /etc/udev/rules.d/99-amdgpu-fan.rules ]; then \
		sudo rm /etc/udev/rules.d/99-amdgpu-fan.rules; \
		sudo udevadm control --reload-rules; \
		echo "  AMD GPU fan control rule removed."; \
	fi
	@echo "  Ollama service left installed and running."
	@echo "  To fully remove Ollama: sudo systemctl disable --now ollama"

# ── Systemd user services ─────────────────────────────────────────────────────

_install-services:
	@echo "--- Systemd user services ---"
	@mkdir -p ~/.config/systemd/user
	@sed "s|@@PROJ_DIR@@|$(PROJ_DIR)|g" $(PROJ_DIR)/loca-watcher.service > ~/.config/systemd/user/loca-watcher.service
	@sed "s|@@PROJ_DIR@@|$(PROJ_DIR)|g" $(PROJ_DIR)/loca-talk.service    > ~/.config/systemd/user/loca-talk.service
	@systemctl --user daemon-reload
	@systemctl --user enable loca-watcher.service
	@echo "  Enabling linger for $(USER) (services run without login)..."
	@sudo loginctl enable-linger $(USER)
	@echo "  Making scripts executable..."
	@chmod +x \
		$(PROJ_DIR)/loca-watcher.sh \
		$(PROJ_DIR)/loca-connect.sh \
		$(PROJ_DIR)/loca-talk.py \
		$(PROJ_DIR)/loca-say.py \
		$(PROJ_DIR)/loca-shell.py \
		$(PROJ_DIR)/loca-quiz.py \
		$(PROJ_DIR)/loca-ha-init.py
	@echo "  Services installed and enabled."
	@echo "  loca-watcher will start automatically at next login/reboot,"
	@echo "  or start now with: systemctl --user start loca-watcher"

_uninstall-services:
	@echo "--- Removing systemd user services ---"
	@systemctl --user stop loca-watcher.service loca-talk.service 2>/dev/null || true
	@systemctl --user disable loca-watcher.service 2>/dev/null || true
	@rm -f ~/.config/systemd/user/loca-watcher.service
	@rm -f ~/.config/systemd/user/loca-talk.service
	@systemctl --user daemon-reload
	@echo "  Services removed."
	@echo "  Note: loginctl linger left enabled for $(USER)."
	@echo "  To disable: sudo loginctl disable-linger $(USER)"

# ── Pairing ───────────────────────────────────────────────────────────────────

pair:
	@echo "--- Pairing device ---"
	@echo "  Classic BT MAC: $(CLASSIC_MAC)"
	@echo "  BLE MAC:        $(BLE_MAC)"
	@echo ""
	@echo "  Power on the device now (Bluetooth indicator should be blinking)."
	@echo "  Press Enter when ready..."
	@read _
	@$(PROJ_DIR)/loca-connect.sh connect

unpair:
	@echo "--- Unpairing device ---"
	@echo "  This will remove the device from BlueZ. You will need to run"
	@echo "  'make pair' again before using it."
	@read -p "  Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		bluetoothctl disconnect $(CLASSIC_MAC) 2>/dev/null || true; \
		sleep 1; \
		bluetoothctl remove $(CLASSIC_MAC) 2>/dev/null || true; \
		echo "  Device unpaired."; \
	else \
		echo "  Aborted."; \
	fi

# ── Voice download ────────────────────────────────────────────────────────────

voices:
	@echo "--- Caching Kokoro TTS voice ---"
	@$(VENV)/bin/python3 -c "\
import yaml; \
cfg = yaml.safe_load(open('$(PROJ_DIR)/config.yaml')); \
voice = cfg.get('tts', {}).get('voice', 'af_heart'); \
print(f'  Voice: {voice}'); \
print('  Downloading from HuggingFace (this may take a moment)...'); \
from kokoro import KPipeline; \
p = KPipeline(lang_code='a'); \
list(p('Ready.', voice=voice)); \
print('  Done. Voice cached at ~/.cache/huggingface/')"

# ── Status ────────────────────────────────────────────────────────────────────

status:
	@echo ""
	@echo "=== Bluetooth ==="
	@bluetoothctl info $(CLASSIC_MAC) 2>/dev/null | \
		grep -E "Name|Connected|Paired|Trusted" || \
		echo "  Device ($(CLASSIC_MAC)) not found in BlueZ"
	@echo ""
	@echo "=== Audio ==="
	@pactl list sinks short 2>/dev/null | grep "$(subst :,_,$(CLASSIC_MAC))" && true || \
		echo "  No device audio sink (is HFP connected?)"
	@pactl list sources short 2>/dev/null | grep "$(subst :,_,$(CLASSIC_MAC))" && true || \
		echo "  No device audio source"
	@echo ""
	@echo "=== Ollama ==="
	@systemctl is-active --quiet ollama && \
		(echo "  ollama: running"; ollama list 2>/dev/null | grep sunny || \
		echo "  WARNING: sunny model not found — run 'make model'") || \
		echo "  ollama: not running"
	@echo ""
	@echo "=== Services ==="
	@systemctl --user is-active --quiet loca-watcher && \
		echo "  loca-watcher: active" || echo "  loca-watcher: inactive"
	@systemctl --user is-active --quiet loca-talk && \
		echo "  loca-talk:    active" || echo "  loca-talk:    inactive"
	@echo ""
	@echo "=== Python ==="
	@if [ -d "$(VENV)" ]; then \
		echo "  venv: $(VENV)"; \
		$(VENV)/bin/pip show kokoro faster-whisper bleak 2>/dev/null | \
			grep -E "^Name|^Version" | paste - - | \
			awk '{printf "  %-20s %s\n", $$2, $$4}'; \
	else \
		echo "  venv not found at $(VENV)"; \
	fi
	@echo ""
	@echo "=== Voices ==="
	@ls ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M 2>/dev/null \
		&& echo "  Kokoro-82M: cached" \
		|| echo "  Kokoro-82M: not cached — run: make voices"
	@echo ""

training: training/model_training.jsonl

training/model_training.jsonl: training/exchanges.yaml hfp_secret.yaml
	python3 training/make_jsonl.py

clear_memory:
	echo "" > memory/long_term.txt
	echo "[]" > memory/recent_turns.json

start:
	systemctl --user start loca-watcher

stop:
	systemctl --user stop loca-watcher
	systemctl --user stop loca-talk

disable:
	systemctl --user stop loca-watcher
	systemctl --user stop loca-talk
	systemctl --user disable loca-watcher

enable:
	systemctl --user enable loca-watcher
	systemctl --user start loca-watcher

restart:
	systemctl --user stop loca-talk || true

follow:
	journalctl --user -xef -u loca-talk

