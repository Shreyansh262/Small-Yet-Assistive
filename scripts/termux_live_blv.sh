#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# Smol-VL-BLV: Live Camera Assistant for Blind & Low-Vision Users
# Captures photos → generates spatial scene descriptions → reads aloud via TTS
#
# Prerequisites:
#   - Run setup_termux.sh first
#   - GGUF models in ~/models/
#   - termux-api installed (for camera + TTS)
#
# Usage:
#   bash termux_live_blv.sh                  # default settings
#   bash termux_live_blv.sh --cooldown 15    # 15s between captures
#   bash termux_live_blv.sh --camera 1       # use front camera
# ============================================================================

set -euo pipefail

# ── Configuration ──
MODEL="${HOME}/models/blv_final_iq4nl_best.gguf"
MMPROJ="${HOME}/models/sft_patch_mmproj.gguf"
LLAMA_BIN="${HOME}/llama.cpp/build/bin/llama-mtmd-cli"
THREADS=4
CTX=1538
MAX_TOKENS=90
TEMP=0.2
COOLDOWN=10
CAMERA_ID=0
PROMPT="Describe this scene to a blind person in max 4 lines"

# ── Parse optional arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --cooldown) COOLDOWN="$2"; shift 2 ;;
        --camera)   CAMERA_ID="$2"; shift 2 ;;
        --threads)  THREADS="$2"; shift 2 ;;
        --tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --prompt)   PROMPT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate prerequisites ──
for f in "$MODEL" "$MMPROJ" "$LLAMA_BIN"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing: $f"
        echo "   Run setup_termux.sh first, then push GGUF models."
        exit 1
    fi
done

for cmd in termux-camera-photo termux-tts-speak; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ Missing: $cmd"
        echo "   Install termux-api: pkg install termux-api"
        echo "   Also install the Termux:API app from F-Droid."
        exit 1
    fi
done

# ── Temp files ──
FRAME=$(mktemp /tmp/blv_frame_XXXXXX.jpg)
DESC=$(mktemp /tmp/blv_desc_XXXXXX.txt)
trap 'rm -f "$FRAME" "$DESC"' EXIT

echo "🟢 Smol-VL-BLV Live Assistant"
echo "   Model : $(basename "$MODEL")"
echo "   Camera: $CAMERA_ID | Threads: $THREADS"
echo "   Cooldown: ${COOLDOWN}s between captures"
echo "   Press Ctrl+C to stop"
echo ""

COUNT=0

while true; do
    COUNT=$((COUNT + 1))
    echo "══════════════════════════════════════════"
    echo "📸 [$COUNT] Capturing frame..."

    # Capture photo
    if ! termux-camera-photo -c "$CAMERA_ID" "$FRAME" 2>/dev/null; then
        echo "❌ Camera error. Retrying in 5s..."
        sleep 5
        continue
    fi

    # Verify frame is non-empty
    if [ ! -s "$FRAME" ]; then
        echo "❌ Empty frame captured. Retrying in 5s..."
        sleep 5
        continue
    fi

    FILESIZE=$(wc -c < "$FRAME")
    echo "   Frame size: ${FILESIZE} bytes"
    echo "🧠 Analyzing scene..."

    # Run inference
    START_TIME=$(date +%s)
    if "$LLAMA_BIN" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --image "$FRAME" \
        --no-warmup \
        -t "$THREADS" \
        -c "$CTX" \
        -n "$MAX_TOKENS" \
        --temp "$TEMP" \
        -p "$PROMPT" \
        > "$DESC" 2>/dev/null; then

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        echo ""
        echo "🗣️  Description (${ELAPSED}s):"
        echo "────────────────────────────────────────"
        cat "$DESC"
        echo ""
        echo "────────────────────────────────────────"

        # Read aloud via TTS (runs in background so we don't block)
        DESCRIPTION=$(cat "$DESC")
        if [ -n "$DESCRIPTION" ]; then
            termux-tts-speak "$DESCRIPTION" &
        fi
    else
        echo "❌ Inference failed. Check model files and retry."
    fi

    rm -f "$FRAME"
    echo "⏳ Next capture in ${COOLDOWN}s..."
    sleep "$COOLDOWN"
done
