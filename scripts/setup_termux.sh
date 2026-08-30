#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# Smol-VL-BLV: One-Time Termux Setup
# Run this ONCE after installing Termux + Termux:API from F-Droid.
# Estimated time: ~30-40 minutes (mostly llama.cpp compilation)
# ============================================================================

set -euo pipefail

echo "========================================="
echo "  Smol-VL-BLV: Termux Setup"
echo "========================================="
echo ""

# ── Step 1: Update system packages ──
echo "[1/4] Updating system packages..."
pkg update -y && pkg upgrade -y

# ── Step 2: Install build tools and dependencies ──
echo "[2/4] Installing dependencies..."
pkg install -y git cmake clang ninja python make termux-api

# ── Step 3: Build llama.cpp from source ──
echo "[3/4] Building llama.cpp (this takes ~30 minutes)..."
if [ -f "$HOME/llama.cpp/build/bin/llama-mtmd-cli" ]; then
    echo "  ✅ llama.cpp already built, skipping."
else
    cd "$HOME"
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp
    fi
    cd llama.cpp
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    echo "  ✅ llama.cpp built successfully!"
fi

# Verify the binary exists
if [ ! -f "$HOME/llama.cpp/build/bin/llama-mtmd-cli" ]; then
    echo "  ❌ ERROR: llama-mtmd-cli not found after build."
    echo "  The binary name may have changed. Check: ls $HOME/llama.cpp/build/bin/"
    exit 1
fi

# ── Step 4: Create directory structure ──
echo "[4/4] Creating directories..."
mkdir -p "$HOME/models"
mkdir -p "$HOME/scripts"

echo ""
echo "========================================="
echo "  ✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Push GGUF models from your PC (run on your PC):"
echo "     adb push blv_final_iq4nl_best.gguf /data/data/com.termux/files/home/models/"
echo "     adb push sft_patch_mmproj.gguf /data/data/com.termux/files/home/models/"
echo ""
echo "  2. Grant camera permission (run in Termux):"
echo "     termux-camera-photo -c 0 ~/test.jpg"
echo ""
echo "  3. Test TTS (run in Termux):"
echo "     termux-tts-speak 'Smol VL BLV setup complete'"
echo ""
echo "  4. Run the live assistant:"
echo "     bash ~/scripts/termux_live_blv.sh"
