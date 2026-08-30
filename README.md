# Smol-VL-BLV: Spatially-Aware Post-Training for Low Vision

<p align="center">
  <a href="https://smol-vl-blv.github.io/Smol-VL-BLV-website/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"></a>
  <a href="TODO_PAPER_LINK"><img src="https://img.shields.io/badge/Paper-EMNLP%202026-red" alt="Paper"></a>
  <a href="https://huggingface.co/YOUR_ORG/smol-vl-blv"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/YOUR_ORG/smol-vl-blv-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-orange" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

<!-- TODO: Replace YOUR_ORG with your actual HuggingFace organization/username -->
<!-- TODO: Replace TODO_PAPER_LINK with the ACL Anthology / ArXiv link once available -->

> **Accepted at EMNLP 2026**

We fine-tune **SmolVLM2-500M-Video-Instruct** (~507M params) into a compact,
on-device vision-language model that generates **audio-description-style,
spatially grounded scene narrations** for blind and low-vision (BLV) users —
distances, directions, obstacles, hazards — instead of the generic captions
produced by off-the-shelf VLMs. The pipeline has three post-training stages:

1. **Teacher-student distillation (SFT, phase-switched)** — captions from a
   Gemma-4-31B-IT teacher prompted with professional audio-description guidelines.
2. **GRPO** (Group Relative Policy Optimization) — composite, rule-based BLV
   reward (directional language, metric distances, hazard/CAUTION flagging,
   structure, hallucination penalty, environment identification).
3. **SFT-patch recovery** — a short LoRA patch after GRPO to recover general
   descriptive quality without hurting BLV spatial grounding.

The final model is exported to GGUF and deployed **fully offline on a mid-range
Android phone** (Samsung Galaxy A55) via mixed-precision quantization.

## Headline Results

| Metric | Baseline (SmolVLM2-500M) | Ours (Condition D) | Relative Gain |
|---|---|---|---|
| Spatial score (LLM judge, 1-10) | 3.21 | 3.83 | **+19.3%** |
| Social score (LLM judge, 1-10) | 3.30 | 3.79 | **+14.8%** |
| OCR-Bench ANLS | 32.5% | 65.5% | **+101.5%** |
| TextVQA accuracy | 37.8% | 54.5% | **+44.2%** |
| On-device size (2 GGUF files) | — | 442 MB | — |
| On-device speed (Samsung A55, CPU) | 13.6 tok/s | 39.3 tok/s | **+189%** |

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Live Demo & Inference](#2-live-demo--inference)
3. [Evaluation (Automated)](#3-evaluation)
4. [Dataset Details & Synthetic Data Generation](#4-dataset-details--synthetic-data-generation)
5. [Training](#5-training)
6. [Quantization](#6-quantization-gguf-export--mixed-precision)
7. [Mobile Deployment (Termux on Android)](#7-mobile-deployment-termux-on-android)
8. [Repository Layout](#8-repository-layout)
9. [Results Reference](#9-results-reference)
10. [Limitations](#10-limitations)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Environment Setup

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training/inference) or CPU-only for inference
- ~4 GB disk space for the base model + checkpoints
- For mobile deployment: Android phone + Termux (see [Section 7](#7-mobile-deployment-termux-on-android))

### Installation

```bash
git clone https://github.com/YOUR_ORG/smol-vl-blv.git
cd smol-vl-blv

# Create virtual environment
conda create -n smolvlblv python=3.10 -y && conda activate smolvlblv
# or: python -m venv .venv && source .venv/bin/activate  (Linux/Mac)
# or: python -m venv .venv && .venv\Scripts\activate     (Windows)

# Install dependencies
pip install -r requirements.txt
```

<!-- TODO: Replace YOUR_ORG with your actual GitHub organization/username -->

### Model Weights

Download the fine-tuned model from HuggingFace:

```bash
# The model will be auto-downloaded when you run demo/evaluation scripts.
# Or download manually:
pip install huggingface_hub
huggingface-cli download YOUR_ORG/smol-vl-blv --local-dir models/student/final
```

<!-- TODO: Replace YOUR_ORG/smol-vl-blv with your actual HuggingFace model repo -->

### llama.cpp (for GGUF quantization & mobile deployment)

If you plan to export GGUF models or deploy on mobile, you also need
[llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
git clone https://github.com/ggerganov/llama.cpp tools/llama_cpp
cd tools/llama_cpp
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
cd ../..
```

---

## 2. Live Demo & Inference

### Quick Start — Single Image (PyTorch)

```bash
# Run on a single image
python demo/run_demo_inference.py --image path/to/photo.jpg

# Run on a directory of images
python demo/run_demo_inference.py --image_dir path/to/frames/

# Specify model (HuggingFace ID or local path)
python demo/run_demo_inference.py --image photo.jpg --model YOUR_ORG/smol-vl-blv

# CPU-only
python demo/run_demo_inference.py --image photo.jpg --device cpu

# Save results to JSON
python demo/run_demo_inference.py --image_dir ./frames/ --output results.json
```

<!-- TODO: Replace YOUR_ORG/smol-vl-blv with your actual HuggingFace model repo -->

### Quick Start — GGUF (llama.cpp CLI)

If you have the quantized GGUF files (see [Section 6](#6-quantization-gguf-export--mixed-precision)):

```bash
./tools/llama_cpp/build/bin/llama-mtmd-cli \
    -m models/gguf/blv_final_iq4nl_best.gguf \
    --mmproj models/gguf/sft_patch_mmproj.gguf \
    --image path/to/photo.jpg \
    -p "Describe this scene for a blind user." \
    -n 90 --temp 0.2 -t 4
```

### Example Output

> **Input:** A photo of a kitchen scene.
>
> **Output:** *"Kitchen environment, no immediate hazards. A wooden dining table
> is approximately 1 meter ahead with two chairs on the left side. A person in
> a blue shirt stands near the counter, approximately 2 meters to the right,
> facing away. The floor is tiled and clear of obstacles."*

---

## 3. Evaluation

### Automated One-Click Evaluation

We provide a single script that automatically downloads the model and benchmarks
from HuggingFace and runs the complete evaluation pipeline:

```bash
# Run all benchmarks (OCR-Bench, TextVQA, BLV eval, LLM judge)
python scripts/evaluate.py --model YOUR_ORG/smol-vl-blv --benchmarks all --gpu 0

# Run specific benchmarks
python scripts/evaluate.py --model YOUR_ORG/smol-vl-blv --benchmarks ocr_bench textvqa

# BLV evaluation only (keyword coverage + NLP metrics)
python scripts/evaluate.py --model YOUR_ORG/smol-vl-blv --benchmarks blv_eval

# Skip LLM judge (if you don't have a local 32B model)
python scripts/evaluate.py --model YOUR_ORG/smol-vl-blv --benchmarks all --skip-judge

# Use Gemini API instead of local Ollama for LLM judge
export GEMINI_API_KEY="your-api-key"
python scripts/evaluate.py --model YOUR_ORG/smol-vl-blv --benchmarks llm_judge \
    --judge-backend gemini_api
```

<!-- TODO: Replace YOUR_ORG/smol-vl-blv with your actual HuggingFace model repo -->

**What the script does:**

1. Downloads the model from HuggingFace (if not already cached locally)
2. Downloads benchmark datasets (OCR-Bench, TextVQA) from HuggingFace
3. Downloads BLV evaluation data from the project's data card
4. Runs selected benchmarks and computes all metrics
5. Prints a consolidated results table and saves to `results/evaluation_output/`

**Benchmarks available:**

| Benchmark | Metrics | Notes |
|---|---|---|
| `ocr_bench` | ANLS (%) | OCR text reading and visual reasoning |
| `textvqa` | Accuracy (%) | Visual question answering on scene text |
| `blv_eval` | BLV keyword coverage, NLP metrics (BLEU/ROUGE/METEOR/CIDEr) | Domain-specific BLV evaluation |
| `llm_judge` | MCF score (spatial/social/action/ambience), NAF score (descriptiveness/objectivity/accuracy/clarity) | Requires local Ollama `qwen2.5:32b` or Gemini API |

### Individual Evaluation Scripts (Advanced)

For more control, you can run each evaluation script individually:

```bash
# Generate captions for all conditions (A=base, B=SFT, C=SFT+GRPO, D=SFT+GRPO+patch)
python src/evaluation/run_inference_all_conditions.py --conditions A,B,C,D --gpu 0

# LLM-as-judge scoring
python src/evaluation/run_llm_judge.py --conditions A B C D

# Standard NLP metrics
python src/evaluation/run_nlp_metrics_ABCD.py

# BLV keyword coverage (Table 7 in the paper)
python src/evaluation/blv_keyword_coverage.py --condition all --gpu 0

# Full consolidated evaluation (all 4 phases)
python src/evaluation/eval_final.py

# OCR-Bench
python src/evaluation/ocr_vqa/ocr_bench/run_ocr_bench.py

# TextVQA
python src/evaluation/ocr_vqa/textvqa/run_textvqa.py
```

---

## 4. Dataset Details & Synthetic Data Generation

### Overview

We train on keyframes + generated captions from two public video datasets:

| Dataset | Videos Used | Source |
|---|---|---|
| **Charades** (Sigurdsson et al., 2016) | 7,985 | [charades.allenai.org](https://prior.allenai.org/projects/charades) |
| **AVCaps** (Sudarsanam et al., 2025) | 1,661 | [AVCaps](https://github.com/AVCaps) |
| **Total** | **9,646** | |

### Option A: Download Pre-Generated Teacher Captions

We provide our pre-generated teacher captions and evaluation data on HuggingFace:

```bash
# Download all pre-generated data
python scripts/download_data.py

# Download evaluation data only
python scripts/download_data.py --split eval

# Download training captions only
python scripts/download_data.py --split train
```

<!-- TODO: Replace YOUR_ORG/smol-vl-blv-data in scripts/download_data.py with your actual HuggingFace dataset repo -->

### Option B: Generate Your Own Teacher Captions

If you want to reproduce the full data pipeline from scratch:

**Step 1: Download the source videos** from Charades and AVCaps (see their
respective project pages). Point the scripts at your local copies via
`configs/paths_config.example.yaml`.

**Step 2: Run the data pipeline:**

```bash
# 1. Extract keyframes (LUV color-space differencing, n=4 per video)
python src/data_pipeline/01_extract_keyframes.py

# 2. Generate teacher captions (Gemma-4-31B-IT with AD-guideline prompt)
python src/data_pipeline/02_generate_teacher_captions.py

# 3. Filter captions by BLV keyword coverage
python src/data_pipeline/03_filter_captions.py

# 4. Build preference pairs for GRPO/DPO training
python src/data_pipeline/04_build_preference_pairs.py

# 5. Analyze teacher coverage (generates Figure 3 data)
python src/data_pipeline/05_analyze_teacher_coverage.py

# 6. Plot teacher coverage chart (Figure 3)
python src/data_pipeline/06_plot_teacher_coverage.py
```

### Keyframe Extraction Algorithm

We extract 4 keyframes per video using LUV perceptual color-space differencing
(zero model overhead, pure algebraic operation):

1. Subsample video into ≤64 evenly spaced frames
2. Convert to CIE 1976 L\*u\*v\* color space
3. Compute mean absolute inter-frame difference
4. Select frame f₁ plus the 3 frames with highest difference, sorted chronologically

---

## 5. Training

All commands assume you are in the repo root with the environment active.
The three stages correspond to Conditions B, C, D in the paper.

### Stage 1 — SFT with Phase-Switching (Condition B)

```bash
python src/training/01_sft_phase_switching.py
```

LoRA fine-tuning of SmolVLM2-500M in a single run: first 25% of steps train
only the SigLIP encoder + connector (LoRA frozen), remaining 75% unfreeze all
LoRA adapters with the vision-encoder LR halved. Uses `DifferentialLRTrainer`
with three independent AdamW parameter groups.

### Stage 2 — GRPO (Condition C)

```bash
python src/training/02_prepare_grpo_data.py
python src/training/03_grpo_train.py
```

Group size G=4, KL coefficient β=0.1, 2 epochs (~9,076 optimization steps).
Reward function: `src/training/grpo_reward.py::compute_blv_reward` — six
deterministic components (CAUTION match, directional-word density, distance
mentions, sentence-count structure, hallucination-phrase penalty, environment
ID in first sentence). See the paper's Table 1 for exact weights.

### Stage 3 — SFT-Patch Recovery (Condition D — deployed model)

```bash
python src/training/04_sft_patch.py
```

3 epochs on 50 curated Gemma captions, fresh LoRA (r=64, α=128, lr=5e-5,
batch size 2) applied on top of the GRPO checkpoint. Recovers general
descriptive quality lost to task-specific over-optimization.

### Baseline Comparisons (Table 9 in the paper)

```bash
python src/training/05_dpo_train.py         # offline DPO on SFT v2
python src/training/06_rlaif_dpo_train.py   # RLAIF-V-style DPO
```

These *underperform* GRPO for BLV-specific navigational content — included
for the sake of a complete, honest reproduction of Table 9.

---

## 6. Quantization (GGUF Export + Mixed-Precision)

### Step 1: Merge LoRA adapters into base weights

```bash
python src/deployment/01_merge_lora.py
```

Folds all three LoRA adapters (SFT + GRPO + SFT-patch) into the base model
weights sequentially. Output: a single merged HuggingFace checkpoint.

### Step 2: Convert to GGUF

```bash
bash src/deployment/02_convert_to_gguf.sh
```

Converts the merged model to GGUF format (F16) and produces:
- **Language model backbone:** `sft_v2_f16.gguf` → quantized to `sft_v2_q4km.gguf` (Q4_K_M)
- **Vision projector:** `sft_v2_mmproj.gguf` (kept at F16, 191 MB)

> **Note:** The conversion uses `src/deployment/gguf_convert_wrapper.py` to work
> around a known torchvision circular-import bug.

### Step 3: Mixed-precision quantization (paper's final model)

```bash
bash src/deployment/03_quantize_mixed_precision.sh
```

This produces the **final deployed model** (442 MB total):

1. **Importance matrix calibration** — runs `llama-imatrix` on 500 BLV
   navigation captions to record channel activation frequencies (~30 min).
2. **IQ4_NL quantization with Q5_K attention override** — the backbone is
   quantized to importance-weighted non-linear 4-bit (IQ4_NL, ~251 MB), but
   attention projections (`q_proj`, `k_proj`, `v_proj`) are kept at 5-bit
   (Q5_K) because uniform 4-bit degrades directional reasoning.

| Quantization | LM Size | Speed (A55) | Notes |
|---|---|---|---|
| Q4_K_M | 290 MB | 17-19 tok/s | Baseline |
| **IQ4_NL + Q5_K attn** | **251 MB** | **39.3 tok/s** | ✅ Paper's final model |
| Q5_K_M | ~340 MB | — | RAM exceeds budget |
| IQ4_XS | ~230 MB | — | Spatial coherence degradation |

### Optional: Ollama benchmark

```bash
bash src/deployment/04_ollama_benchmark_reference.sh
```

---

## 7. Mobile Deployment (Termux on Android)

Deploy the model to run **fully offline** on an Android phone. Tested on
Samsung Galaxy A55 (Exynos 1480, 8 GB RAM, Android 16).

### Prerequisites

- Android phone with ≥4 GB RAM
- [Termux](https://f-droid.org/packages/com.termux/) installed from **F-Droid** (not Play Store)
- [Termux:API](https://f-droid.org/packages/com.termux.api/) installed from F-Droid (for camera & TTS)
- USB cable + ADB on your PC (for file transfer)

### Step 1: Setup Termux on the Phone

Open Termux on your Android device and run:

```bash
# Download and run the setup script (builds llama.cpp, ~30-40 min)
pkg install -y wget
wget https://raw.githubusercontent.com/YOUR_ORG/smol-vl-blv/main/scripts/setup_termux.sh
bash setup_termux.sh
```

<!-- TODO: Replace YOUR_ORG with your actual GitHub organization/username -->

Or copy the setup script manually and run it:

```bash
bash ~/scripts/setup_termux.sh
```

**What this does:**
1. Updates Termux packages
2. Installs build dependencies (git, cmake, clang, ninja, termux-api)
3. Clones and builds `llama.cpp` from source (~30 min)
4. Creates `~/models/` directory

### Step 2: Push Model Files from Your PC

On your PC (with the phone connected via USB):

```bash
# Automated deployment (pushes models + scripts)
bash scripts/deploy_to_device.sh

# Or push manually:
adb push models/gguf/blv_final_iq4nl_best.gguf /data/data/com.termux/files/home/models/
adb push models/gguf/sft_patch_mmproj.gguf /data/data/com.termux/files/home/models/
adb push scripts/termux_live_blv.sh /data/data/com.termux/files/home/scripts/
```

### Step 3: Grant Permissions

In Termux, test camera and TTS:

```bash
# Test camera (will ask for permission on first run)
termux-camera-photo -c 0 ~/test.jpg

# Test text-to-speech
termux-tts-speak "Smol VL BLV is ready"
```

### Step 4: Run the Live Camera Assistant

```bash
bash ~/scripts/termux_live_blv.sh
```

The script will:
1. 📸 Capture a photo from the rear camera
2. 🧠 Run the VLM to generate a spatially-grounded scene description
3. 🗣️ Read the description aloud via text-to-speech
4. ⏳ Wait 10 seconds, then repeat

**Options:**

```bash
# Adjust cooldown between captures
bash ~/scripts/termux_live_blv.sh --cooldown 15

# Use front camera
bash ~/scripts/termux_live_blv.sh --camera 1

# Custom prompt
bash ~/scripts/termux_live_blv.sh --prompt "What obstacles are near me?"
```

### Expected Performance (Samsung Galaxy A55)

| Metric | Value |
|---|---|
| Model load time | ~291 ms |
| Time to first token | ~25.1 s |
| Generation speed | **39.3 tok/s** |
| Total per-frame latency | ~27.1 s |
| Peak RAM | ~780 MB |
| Total model size | 442 MB |

> **Note on GPU acceleration:** On the Samsung A55, we use CPU-only inference
> because the Exynos Mali OpenCL/Vulkan drivers are inaccessible from Termux.
> On hardware with proper GPU support (e.g., Apple Silicon via Metal, NVIDIA
> via CUDA), generation completes in **under 2.5 seconds**.

---

## 8. Repository Layout

```
smol-vl-blv/
├── src/
│   ├── data_pipeline/     # keyframe extraction, teacher captioning, filtering
│   ├── training/          # SFT (phase-switched), GRPO, SFT-patch, DPO/RLAIF-V
│   ├── deployment/        # LoRA merge → GGUF export → mixed-precision quantization
│   └── evaluation/        # LLM-judge, NLP metrics, BLV keyword coverage, OCR/VQA
├── scripts/
│   ├── evaluate.py        # one-click automated evaluation
│   ├── download_data.py   # download pre-generated data from HuggingFace
│   ├── setup_termux.sh    # one-time Termux environment setup
│   ├── termux_live_blv.sh # live camera inference loop for Android
│   └── deploy_to_device.sh # push models to Android via ADB
├── demo/                  # quick inference on your own images
├── configs/               # example path config
├── results/               # evaluation artifacts (scores, figures)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## 9. Results Reference

- `results/final_eval.png` — consolidated table across all 4 conditions
- `results/gemma_coverage_chart.png` — Figure 3 (teacher caption BLV attribute coverage)
- `results/grpo_reward_curve.png` — GRPO training reward curve
- `results/scores/`, `results/analysis/` — per-condition JSON scores

Full tables (NLP metrics, BLV keyword coverage, LLM-judge sub-dimensions,
DPO/RLAIF-V comparison, cross-hardware latency) are in the paper.

---

## 10. Limitations

This model targets **low-vision assistance** primarily; it does not yet meet
the reliability bar for safe independent navigation by fully blind users.

- Coverage remains weak for step-changes/curbs (5%) and only partial for
  moving hazards (66%), largely due to under-representation in Charades/AVCaps.
- Reported distances are the teacher model's visual estimates, not sensor
  measurements.
- On-device latency (~27s per frame on the Samsung A55) is usable for
  periodic scene checks but not real-time continuous navigation.

---

## 11. Citation

```bibtex
@inproceedings{choudhary2026small,
  title     = {Small yet Assistive: Spatially-Aware Post-Training for Low Vision},
  author    = {Choudhary, Rishabh and Raj, Shreyansh and Goyal, Umesh and Kasyap, Subh and Kumar, Shrestha and Jena, Sushovan and Kumar, Komal and Cholakkal, Hisham and Nigam, Aditya},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

This work builds on the MCF/NAF evaluation frameworks from:

```bibtex
@inproceedings{baghel2025towards,
  title     = {Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals},
  author    = {Baghel, Shruti Singh and Rathore, Yash Pratap Singh and Pradhan, Anurag and Jena, Sushovan and Bhavsar, Arnav and Shukla, Amit and Goyal, Pawan},
  booktitle = {Proceedings of the 1st Workshop on Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo 2025)},
  pages     = {86--94},
  year      = {2025}
}
```

---

## 12. License

This codebase is released under the [MIT License](LICENSE).

> **Note:** The datasets (Charades, AVCaps), base model (SmolVLM2), and teacher
> model (Gemma-4-31B-IT) each have their own licenses. Users must comply with
> those licenses separately when using the respective resources.

---

## Acknowledgments

We thank the authors of [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[Gemma](https://ai.google.dev/gemma), [llama.cpp](https://github.com/ggml-org/llama.cpp),
and the [TRL](https://github.com/huggingface/trl) library for making this
research possible.
