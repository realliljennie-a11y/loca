#!/bin/bash
# loca-watcher.sh — Watch for the device's Bluetooth connection and manage
# the HFP profile and loca-talk service lifecycle.
#
# Run as a systemd user service. Polls every 5 seconds.
# When device connects: sets HFP profile, boosts mic, starts loca-talk.
# When device disconnects: stops loca-talk.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CONF_DIR="$SCRIPT_DIR"
eval $(python3 -c "
import sys, os
sys.path.insert(0, '$CONF_DIR')
os.chdir('$CONF_DIR')
from config import CFG
inp = CFG.get('audio', {}).get('input', {})
out = CFG.get('audio', {}).get('output', {})
bt_mac = out.get('bt_mac') or inp.get('bt_mac', '')
print('BT_MAC=' + bt_mac)
print('BT_CARD=' + inp.get('bt_card', ''))
print('AUDIO_SOURCE=' + inp.get('source', ''))
print('AUDIO_SINK=' + out.get('sink', ''))
print('BT_PROFILE=' + inp.get('bt_profile', ''))
print('MIC_VOLUME=' + str(inp.get('mic_volume', '')))
")
LOCA_SERVICE="loca-talk.service"
POLL_INTERVAL=5
SETUP_DELAY=4   # seconds to wait after connection before setting profile

log() { echo "$(date '+%H:%M:%S') loca-watcher: $*"; }

poe_bt_connected() {
    bluetoothctl info "$BT_MAC" 2>/dev/null | grep -q "Connected: yes"
}

poe_audio_ready() {
    pactl list sinks short 2>/dev/null | grep -q "${BT_MAC//:/_}"
}

loca_talk_running() {
    systemctl --user is-active "$LOCA_SERVICE" > /dev/null 2>&1
}

setup_audio() {
    log "Setting up HFP audio profile..."
    pactl set-card-profile "$BT_CARD" off 2>/dev/null
    sleep 1
    pactl set-card-profile "$BT_CARD" "$BT_PROFILE" 2>/dev/null
    sleep 2
    pactl set-source-volume "$AUDIO_SOURCE" "$MIC_VOLUME" 2>/dev/null
    sleep 1
    # Verify volume was set
    local vol
    vol=$(pactl list sources 2>/dev/null | \
          grep -A 5 "$AUDIO_SOURCE" | \
          grep "Volume:" | \
          grep -o '[0-9]*%' | head -1)
    log "Audio ready. Mic volume: ${vol:-unknown}"
}

log "Starting. Watching for device ($BT_MAC)..."

was_connected=false

while true; do
    if poe_bt_connected; then
        if ! $was_connected; then
            log "Device connected."
            sleep $SETUP_DELAY
            setup_audio

            if poe_audio_ready; then
                if ! loca_talk_running; then
                    log "Starting loca-talk..."
                    systemctl --user start "$LOCA_SERVICE"
                fi
            else
                log "Audio not ready after setup — will retry next poll."
            fi
            was_connected=true
        else
            if poe_audio_ready && ! loca_talk_running; then
                log "loca-talk not running but device connected — restarting..."
                setup_audio
                sleep 1
                systemctl --user start "$LOCA_SERVICE"
            fi
        fi
    else
        if $was_connected; then
            log "Device disconnected."
            if loca_talk_running; then
                log "Stopping loca-talk..."
                systemctl --user stop "$LOCA_SERVICE"
            fi
            was_connected=false
        fi

        # Try to connect if device is visible but not connected
        if bluetoothctl devices | grep -q "$BT_MAC"; then
            log "Device visible but not connected — attempting connect..."
            bluetoothctl info "$BT_MAC" 2>/dev/null | grep -q "Name:" && \
                bluetoothctl connect "$BT_MAC" > /dev/null 2>&1 &
            sleep 8   # give it time to connect before next poll
        fi
    fi

    sleep $POLL_INTERVAL
done
