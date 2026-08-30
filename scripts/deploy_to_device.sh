#!/usr/bin/env bash
# ============================================================================
# Smol-VL-BLV: Deploy to Android Device via ADB
# Run this from your PC with the phone connected via USB.
#
# Usage:
#   bash scripts/deploy_to_device.sh                     # deploy models + scripts
#   bash scripts/deploy_to_device.sh --models-only       # deploy models only
#   bash scripts/deploy_to_device.sh --scripts-only      # deploy scripts only
#   bash scripts/deploy_to_device.sh --gguf-dir path/    # custom GGUF location
# ============================================================================

set -euo pipefail

TERMUX_HOME="/data/data/com.termux/files/home"
GGUF_DIR="models/gguf"
DEPLOY_MODELS=true
DEPLOY_SCRIPTS=true

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --models-only)  DEPLOY_SCRIPTS=false; shift ;;
        --scripts-only) DEPLOY_MODELS=false; shift ;;
        --gguf-dir)     GGUF_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "📱 Smol-VL-BLV: Deploying to Android Device"
echo "═══════════════════════════════════════════"

# ── Check ADB connection ──
if ! command -v adb &>/dev/null; then
    echo "❌ ADB not found. Install Android SDK Platform Tools:"
    echo "   https://developer.android.com/tools/releases/platform-tools"
    exit 1
fi

DEVICE_COUNT=$(adb devices | grep -c 'device$' || true)
if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "❌ No Android device connected."
    echo "   1. Enable USB Debugging on your phone"
    echo "   2. Connect via USB cable"
    echo "   3. Accept the 'Allow USB debugging' prompt on your phone"
    exit 1
fi
echo "✅ Device connected"
echo ""

# ── Create directories on device ──
adb shell "mkdir -p ${TERMUX_HOME}/models ${TERMUX_HOME}/scripts" 2>/dev/null || true

# ── Deploy GGUF models ──
if [ "$DEPLOY_MODELS" = true ]; then
    echo "📦 Pushing GGUF models (~442 MB total)..."
    echo "   This may take a few minutes."
    echo ""

    LM_GGUF="${GGUF_DIR}/blv_final_iq4nl_best.gguf"
    MMPROJ_GGUF="${GGUF_DIR}/sft_patch_mmproj.gguf"

    if [ ! -f "$LM_GGUF" ]; then
        echo "❌ LM GGUF not found: $LM_GGUF"
        echo "   Run the quantization pipeline first, or set --gguf-dir"
        exit 1
    fi
    if [ ! -f "$MMPROJ_GGUF" ]; then
        echo "❌ mmproj GGUF not found: $MMPROJ_GGUF"
        exit 1
    fi

    echo "   → $(basename $LM_GGUF) ($(du -h "$LM_GGUF" | cut -f1))..."
    adb push "$LM_GGUF" "${TERMUX_HOME}/models/blv_final_iq4nl_best.gguf"

    echo "   → $(basename $MMPROJ_GGUF) ($(du -h "$MMPROJ_GGUF" | cut -f1))..."
    adb push "$MMPROJ_GGUF" "${TERMUX_HOME}/models/sft_patch_mmproj.gguf"

    echo "   ✅ Models deployed"
    echo ""
fi

# ── Deploy scripts ──
if [ "$DEPLOY_SCRIPTS" = true ]; then
    echo "📦 Pushing scripts..."

    adb push scripts/setup_termux.sh "${TERMUX_HOME}/scripts/setup_termux.sh"
    adb push scripts/termux_live_blv.sh "${TERMUX_HOME}/scripts/termux_live_blv.sh"

    # Set executable permissions
    adb shell "chmod +x ${TERMUX_HOME}/scripts/*.sh"

    echo "   ✅ Scripts deployed"
    echo ""
fi

echo "═══════════════════════════════════════════"
echo "✅ Deployment complete!"
echo ""
echo "On your phone, open Termux and run:"
echo "  bash ~/scripts/setup_termux.sh     # one-time setup (~30 min)"
echo "  bash ~/scripts/termux_live_blv.sh   # start live assistant"
