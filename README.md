# Smol-VL-BLV: Spatially-Aware Post-Training for Low Vision

Code and evaluation artifacts for **"Small yet Assistive: Spatially-Aware
Post-Training for Low Vision"**.

> `Accepted at EMNLP 2026 `

We fine-tune **SmolVLM2-500M-Video-Instruct** (~507M params) into a compact,
on-device vision-language model that generates audio-description-style,
spatially grounded scene narrations for blind and low-vision (BLV) users --
distances, directions, obstacles, hazards -- instead of the generic captions
produced by off-the-shelf VLMs. The pipeline has three post-training stages:

1. **Teacher-student distillation (SFT, phase-switched)** on captions from a
   Gemma-4-31B-IT teacher prompted with professional audio-description
   guidelines.
2. **GRPO** (Group Relative Policy Optimization) with a composite, rule-based
   BLV reward (directional language, metric distances, hazard/CAUTION
   flagging, structure, hallucination penalty, environment identification).
3. **SFT-patch recovery**, a short LoRA patch after GRPO to recover general
   descriptive quality lost to task-specific over-optimization, without hurting
   BLV spatial grounding.

The final model is exported to GGUF and deployed fully offline on a
mid-range Android phone (Samsung Galaxy A55) via mixed-precision
quantization (IQ4_NL + importance matrix, with attention projections kept
at Q5_K).

## Headline results

| | Baseline (SmolVLM2-500M) | Ours (Condition D) | Relative gain |
|---|---|---|---|
| Spatial score (1-10, LLM judge) | 3.21 | 3.83 | +19.3% |
| Social score (1-10, LLM judge) | 3.30 | 3.79 | +14.8% |
| OCR-Bench ANLS | 32.5% | 65.5% | +101.5% |
| TextVQA accuracy | 37.8% | 54.5% | +44.2% |
| On-device size (2 GGUF files) | -- | 442 MB | -- |
| On-device generation speed (Samsung A55, CPU) | 13.6 tok/s (paper baseline) | 39.3 tok/s | -- |

Full tables (NLP metrics, BLV keyword coverage, LLM-judge sub-dimensions,
DPO/RLAIF-V comparison, cross-hardware latency) are reproduced in
[`results/`](results/) and in the paper.

---

## Repository layout

```
smol-vl-blv/
├── src/
│   ├── data_pipeline/     # keyframe extraction, teacher captioning, filtering, preference pairs
│   ├── training/          # SFT (phase-switched), GRPO, SFT-patch, DPO / RLAIF-V (ablation baselines)
│   ├── deployment/        # LoRA merge -> GGUF export -> mixed-precision quantization -> on-device bench
│   └── evaluation/        # LLM-judge (MCF/NAF), NLP metrics, BLV keyword coverage, OCR/VQA, cross-model comparison
├── demo/                  # minimal script to run the final model on your own video/keyframes
├── configs/                # example path config
├── results/                # small evaluation artifacts (scores, final tables, figures) checked into git
├── requirements.txt
```

Every stage is a numbered, standalone script meant to be run **from the repo
root**. Filenames match the order they run in; `grpo_reward.py` is imported
by `src/training/03_grpo_train.py` (`from src.training.grpo_reward import
compute_blv_reward`), so keep `src/training/` as a folder if you reorganize
further.

---

## Setup

```bash
git clone <your-repo-url>.git
cd smol-vl-blv
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

You will also need a working [llama.cpp](https://github.com/ggml-org/llama.cpp)
build for the deployment stage (`src/deployment/`) -- this is a separate C++
build, not a pip package. See `src/deployment/02_convert_to_gguf.sh` and
`03_quantize_mixed_precision.sh` for the exact steps.

### Data

We train on keyframes + generated captions from two public video datasets:

- **Charades** (Sigurdsson et al., 2016) -- 7,985 videos used
- **AVCaps** (Sudarsanam et al., 2025) -- 1,661 videos used

Neither dataset's raw videos are redistributed in this repo. Download them
from their original sources and point `src/data_pipeline/01_extract_keyframes.py`
at your local copies (see `configs/paths_config.example.yaml`).

### Model weights

> **TODO:** upload the final checkpoints (SFT v2, GRPO, SFT-patch / Condition
> D, and the quantized GGUF pair) to a model host -- e.g. the Hugging Face
> Hub -- and link them here. They are intentionally not committed to this
> git repo (each PyTorch checkpoint is hundreds of MB to several GB; the
> final GGUF pair alone is 442 MB). Suggested repo name: `<org>/smol-vl-blv`.

---

## Reproducing the pipeline

All commands assume you are in the repo root with the venv/conda env active.

### 1. Data preparation

```bash
python src/data_pipeline/01_extract_keyframes.py      # LUV keyframe extraction (n=4/video)
python src/data_pipeline/02_generate_teacher_captions.py   # Gemma-4-31B-IT teacher captions, AD-guideline system prompt
python src/data_pipeline/03_filter_captions.py         # filter by BLV keyword coverage
python src/data_pipeline/04_build_preference_pairs.py  # build GRPO/DPO preference pairs
python src/data_pipeline/05_analyze_teacher_coverage.py  # -> results/gemma_coverage.json (Figure 3 data)
python src/data_pipeline/06_plot_teacher_coverage.py      # -> results/gemma_coverage_chart.png (Figure 3)
```

### 2. Stage 1 -- SFT with phase-switching

```bash
python src/training/01_sft_phase_switching.py
```
LoRA fine-tuning of SmolVLM2-500M in a single run: first 25% of steps train
only the SigLIP encoder + connector (LoRA frozen), remaining 75% unfreeze all
LoRA adapters with the vision-encoder LR halved (`DifferentialLRTrainer`,
three independent AdamW parameter groups). This is **Condition B** in the
paper.

### 3. Stage 2 -- GRPO

```bash
python src/training/02_prepare_grpo_data.py
python src/training/03_grpo_train.py
```
Group size G=4, KL coefficient β=0.1, 2 epochs (~9,076 optimization steps),
reward = `src/training/grpo_reward.py::compute_blv_reward` (six deterministic
components -- CAUTION match, directional-word density, distance mentions,
sentence-count structure, hallucination-phrase penalty, environment ID in the
first sentence; see the paper's Table 1 for exact weights). This is
**Condition C**.

### 4. Stage 3 -- SFT-patch recovery

```bash
python src/training/04_sft_patch.py
```
3 epochs on 50 curated Gemma captions, fresh LoRA (r=64, α=128, lr=5e-5,
batch size 2) applied on top of the GRPO checkpoint. This is **Condition D
(the paper's best/deployed model)**.

### 5. Baseline comparisons used in the paper (Table 9)

```bash
python src/training/05_dpo_train.py         # offline DPO on SFT v2
python src/training/06_rlaif_dpo_train.py   # RLAIF-V-style DPO
```
Reported to *underperform* GRPO for BLV-specific navigational content --
included here for the sake of a complete, honest reproduction of Table 9, not
because they're the recommended training path.

### 6. Deployment (GGUF export + mobile quantization)

```bash
python src/deployment/01_merge_lora.py            # fold SFT+GRPO+SFT-patch LoRA into base weights
bash src/deployment/02_convert_to_gguf.sh          # HF -> GGUF (F16) + Q4_K_M quantization
bash src/deployment/03_quantize_mixed_precision.sh # IQ4_NL + imatrix + Q5_K attention override (final deployed model, ⚠️ see script header)
```
`gguf_convert_wrapper.py` works around a torchvision circular-import bug
triggered by `convert_hf_to_gguf.py` under recent `transformers` versions.

### 7. Evaluation

```bash
# Generate captions for all 4 conditions (A=base, B=SFT, C=SFT+GRPO, D=SFT+GRPO+patch)
python src/evaluation/run_inference_all_conditions.py

# LLM-as-judge scoring (MCF: spatial/social/action/ambience; NAF: descriptiveness/objectivity/accuracy/clarity)
python src/evaluation/run_llm_judge.py

# Standard NLP metrics (BLEU/ROUGE/METEOR/CIDEr)
python src/evaluation/run_nlp_metrics_ABCD.py

# BLV / navigational keyword coverage (Table 7)
python src/evaluation/blv_keyword_coverage.py

# Aggregate everything into the final results table
python src/evaluation/eval_final.py
```

General VLM benchmarks (Table 2, OCR-Bench + TextVQA) and the cross-model
comparison against PaliGemma2, moondream2, Qwen2/2.5-VL, etc. (Table 3) live
in `src/evaluation/ocr_vqa/` and `src/evaluation/cross_model_benchmark/`
respectively. These download their own benchmark data on first run --
nothing large is checked into git.

### 8. Demo

```bash
python demo/run_demo_inference.py --model_dir <path-to-condition-D-checkpoint> --frames_dir <your-keyframes-dir>
```
Runs the final model on a handful of your own videos/keyframes and prints
BLV-style descriptions, similar to Figure 1/4 in the paper.

---

## Results reference

- `results/final_eval.png` -- presentation-ready table across all 4 conditions
  (NLP metrics, BLV coverage, LLM judge).
- `results/gemma_coverage_chart.png` -- Figure 3 (teacher caption BLV
  attribute coverage).
- `results/grpo_reward_curve.png` -- GRPO training reward curve.
- `results/scores/`, `results/analysis/final/` -- the underlying per-condition
  JSON scores and generated captions behind the paper's tables.

## Limitations (from the paper)

This model targets **low-vision** assistance primarily; it does not yet meet
the reliability bar for safe independent navigation by fully blind users.
Coverage remains weak for step-changes/curbs (5%) and only partial for moving
hazards (66%), largely due to under-representation of these events in
Charades/AVCaps. Reported distances are the teacher model's visual estimates,
not sensor measurements.

## Related work this builds on

This project extends the evaluation frameworks (Multi-Context Framework /
MCF and Navigational Assistance Framework / NAF) introduced in Baghel et al.,
2025, *"Towards Blind and Low-Vision Accessibility of Lightweight VLMs and
Custom LLM-Evals"* (ACL MMLoSo 2025) -- see the paper's References for the
full citation.

## Citation

```bibtex
% TODO: replace with the official camera-ready BibTeX from the ACL Anthology
% once available.
@inproceedings{smolvlblv2026,
  title     = {Small yet Assistive: Spatially-Aware Post-Training for Low Vision},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

## License

TODO -- see [`LICENSE`](LICENSE) (currently a placeholder MIT license; confirm
this is what your institution/venue wants for the code release, and check the
license terms of Charades, AVCaps, and the SmolVLM2 base model separately --
those are **not** covered by this repo's license).
