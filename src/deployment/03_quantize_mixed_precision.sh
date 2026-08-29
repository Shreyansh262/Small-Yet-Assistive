#!/usr/bin/env bash

set -e
cd "$(dirname "$0")/../.."   # repo root

LLAMA_DIR="tools/llama_cpp"          # adjust to wherever you built llama.cpp
GGUF_DIR="models/gguf"
F16_GGUF="$GGUF_DIR/blv_final_f16.gguf"          # merged SFT+GRPO+SFT-patch model, converted to F16 GGUF
IMATRIX_CALIB="data/generated/blv_calibration_500.txt"   # 500 BLV captions used for calibration (paper 3.4)
IMATRIX_FILE="$GGUF_DIR/blv_imatrix.dat"
OUT_GGUF="$GGUF_DIR/blv_final_iq4nl_best.gguf"

# ── Step 1: Generate the importance matrix from BLV calibration captions ────
# ~30 min one-time cost (paper Appendix A.2.1, Step 3)
"$LLAMA_DIR/build/bin/llama-imatrix" \
    -m "$F16_GGUF" \
    -f "$IMATRIX_CALIB" \
    -o "$IMATRIX_FILE" \
    2>&1 | tee logs/deployment/imatrix.log

# ── Step 2: Quantize to IQ4_NL with Q5_K override on attention projections ──
# q_proj/k_proj/v_proj kept at higher precision because they govern
# directional/spatial focus (paper 3.4, "Mixed-Precision Quantization Strategy")
"$LLAMA_DIR/build/bin/llama-quantize" \
    --imatrix "$IMATRIX_FILE" \
    --tensor-type attn_q=q5_K \
    --tensor-type attn_k=q5_K \
    --tensor-type attn_v=q5_K \
    "$F16_GGUF" \
    "$OUT_GGUF" \
    IQ4_NL \
    2>&1 | tee logs/deployment/quant_iq4nl.log

echo ""
echo "Done. Expect ~251 MB for the language-model backbone (paper Table 4)."
ls -lh "$OUT_GGUF"