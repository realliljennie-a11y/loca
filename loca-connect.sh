#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
eval $(python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR')
os.chdir('$SCRIPT_DIR')
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

cmd=${1:-connect}

case "$cmd" in
    connect)
#        sudo systemctl restart bluetooth
#        sleep 2
        bluetoothctl power on
        sleep 1

        echo "Scanning (power on device now)..."
        # Start a long-running bluetoothctl just for scanning/agent
        # and leave it running in background throughout
        coproc BTC { bluetoothctl; }
        echo "scan on" >&${BTC[1]}
        sleep 12
        echo "scan off" >&${BTC[1]}
        sleep 1

        echo "Pairing..."
        echo "pair $BT_MAC" >&${BTC[1]}
        sleep 5

        echo "Trusting..."
        echo "trust $BT_MAC" >&${BTC[1]}
        sleep 1

        echo "Connecting..."
        echo "connect $BT_MAC" >&${BTC[1]}
        sleep 5

        echo "Waiting for PipeWire card..."
        for i in $(seq 1 20); do
            if pactl list cards short 2>/dev/null | grep -Fq "$BT_CARD"; then
                echo "Card found."
                break
            fi
            echo "  ($i/20)"
            sleep 1
        done

        echo "Setting profile..."
        pactl set-card-profile $BT_CARD $BT_PROFILE
        sleep 3

        echo "Waiting for audio source..."
        for i in $(seq 1 20); do
            if pactl list sources short 2>/dev/null | grep -Fq "$AUDIO_SOURCE"; then
                echo "Source found."
                pactl set-source-volume $AUDIO_SOURCE $MIC_VOLUME
                break
            fi
            echo "  ($i/20)"
            sleep 1
        done

        echo ""
        echo "=== Status ==="
        bluetoothctl info $BT_MAC 2>/dev/null | \
            grep -E "Name|Connected|Paired|Trusted"
        pactl list cards 2>/dev/null | \
            grep -A 35 "$BT_CARD" | grep "Active Profile"
        pactl list sinks   short 2>/dev/null | \
            grep "${BT_MAC//:/_}" || echo "  No sink."
        pactl list sources short 2>/dev/null | \
            grep "${BT_MAC//:/_}" || echo "  No source."

        # Leave the coproc bluetoothctl running as agent
        # Kill it cleanly on script exit
        echo "quit" >&${BTC[1]}
        wait ${BTC_PID} 2>/dev/null
        ;;

    disconnect)
        bluetoothctl disconnect $BT_MAC 2>/dev/null || true
        sleep 1
        bluetoothctl remove $BT_MAC 2>/dev/null || true
        echo "Disconnected and removed."
        ;;

    status)
        bluetoothctl info $BT_MAC 2>/dev/null | \
            grep -E "Name|Connected|Paired|Trusted"
        pactl list cards 2>/dev/null | \
            grep -A 35 "$BT_CARD" | grep "Active Profile"
        pactl list sinks   short 2>/dev/null | \
            grep "${BT_MAC//:/_}" || echo "No sink."
        pactl list sources short 2>/dev/null | \
            grep "${BT_MAC//:/_}" || echo "No source."
        ;;

    profile)
        pactl set-card-profile $BT_CARD ${2:-$BT_PROFILE}
        ;;

    volume)
        pactl set-source-volume $AUDIO_SOURCE ${2:-$MIC_VOLUME}
        ;;

    wake)
        echo "Toggling profile to wake audio transport..."
        pactl set-card-profile $BT_CARD off
        sleep 1
        pactl set-card-profile $BT_CARD $BT_PROFILE
        sleep 2
        pactl set-source-volume $AUDIO_SOURCE $MIC_VOLUME
        echo "Done."
        pactl list sinks short | grep "${BT_MAC//:/_}" || echo "No sink."
        pactl list sources short | grep "${BT_MAC//:/_}" || echo "No source."
        ;;

    *)
        echo "Usage: $0 [connect|disconnect|status|wake|profile <name>|volume <pct>]"
        ;;
esac
