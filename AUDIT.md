# Cleanup audit log

This documents exactly what changed when this public repo was built from the
original working repo (`p16_blv-main`), so nothing here is a mystery later.
Original repo had 491 files / ~196 MB (excluding data/models/tools already
git-ignored); this repo has ~85 files / ~3.6 MB.

## What did NOT change

- No file was renamed or moved in a way that breaks a Python import.
  The only cross-file import in the entire original codebase was
  `src/training/09_grpo_train.py` → `from src.training.grpo_reward import
  compute_blv_reward`. Both files were kept together under `src/training/`
  in the new repo (`03_grpo_train.py` + `grpo_reward.py`), and the import
  still resolves the same way (`sys.path.insert(0, os.getcwd())`, so run
  scripts from the repo root).
- Every copied `.py` file was byte-checked with `python3 -m py_compile`
  after path sanitization -- all compile cleanly (syntax only; none of this
  was executed, since the original models/GPU/data aren't available here).

## What was renamed (safe -- no other file referenced these by name)

| Original | New | Why |
|---|---|---|
| `src/data_pipeline/03_generate_gemma_captions.py` | `src/data_pipeline/02_generate_teacher_captions.py` | This is the teacher used in the paper (Gemma-4-31B-IT); renamed for a clean numbered sequence. |
| `src/data_pipeline/filter_captions.py` | `03_filter_captions.py` | numbered into pipeline order |
| `src/data_pipeline/04_build_dpo_pairs.py` | `04_build_preference_pairs.py` | clarity (used for both DPO and GRPO pair prep) |
| `analyze_gemma_coverage.py`, `plot_gemma_coverage.py` (repo root) | `src/data_pipeline/05_...`, `06_...` | these produce Figure 3 and belonged in the pipeline, not the repo root |
| `src/training/10_sft_v2.py` | `src/training/01_sft_phase_switching.py` | this is the paper's Stage 1 (Section 3.2); verified line-by-line against the paper (25% phase-1 transition step, `DifferentialLRTrainer`, three AdamW param groups, LoRA r/α, label masking) |
| `src/training/09_grpo_train.py` | `src/training/03_grpo_train.py` | Stage 2 |
| `src/training/sft_patch_grpo_v2.py` | `src/training/04_sft_patch.py` | Stage 3; verified hyperparameters (r=64, α=128, lr=5e-5, batch=2, 3 epochs, 50 captions) match paper Section 3.3 exactly |
| `src/training/09_dpo_training.py` | `src/training/05_dpo_train.py` | this is the DPO run reported in Table 9 (the "v2"; the original `07_dpo_train.py` v1 was dropped, see below) |
| `src/training/12_rlaif_dpo.py` | `src/training/06_rlaif_dpo_train.py` | RLAIF-V DPO row in Table 9 |
| `src/deployment/11_merge_lora.py` → `01_merge_lora.py`, `12_gguf_pipeline.sh` → `02_convert_to_gguf.sh`, `13_ollama_bench.sh` → `04_ollama_benchmark_reference.sh` | renumbered into pipeline order | |
| `src/evaluation/blv_score.py` (was `src/evaluation/` in original, separate from `src/eval/`) | `src/evaluation/blv_keyword_coverage.py` | clarified name; this is Table 7 |
| `ten_video_inference.py` (repo root) | `demo/run_demo_inference.py` | this is the closest thing to a "run the model on a video" demo script |

## What was cut (excluded from the public repo)

**Superseded / pre-Gemma experiments (confusing if left in — see "Naming
issue" below):**
- `src/data_pipeline/03_generate_teacher_captions.py`, `src/training/06_sft_train.py`,
  `src/training/06_sft_stage1_projector.py`, `src/inference/compare_inference.py`,
  `src/inference/raw_captions_comparison.py`, `config/paths_config.yaml`,
  `qwen_captions.py` -- these are from an **earlier version of the pipeline
  that used Qwen2-VL-7B-Instruct as the teacher model**, before the team
  switched to Gemma-4-31B-IT (the teacher actually described in the paper).
  Keeping both versions in one repo with similar filenames is exactly the
  kind of confusion you flagged -- a contributor could easily fine-tune
  against the wrong/old teacher captions. Only the Gemma-teacher path was kept.
  (Files where "Qwen" legitimately appears in the *kept* repo, e.g.
  `src/evaluation/ocr_vqa/baselines/run_qwen.py`, are fine -- there Qwen2-VL
  is evaluated as a **baseline model for comparison in Table 2/3**, not as
  the teacher. Different role, not a mistake.)
- `src/training/08_sft_single_stage.py`, `11_sft_v3.py`, `sft_val3.py`,
  `sft_val4.py`, `sft_validation.py`, `sft_patch_grpo.py` (v1) -- earlier/
  alternate training runs superseded by the two scripts actually reported in
  the paper (`01_sft_phase_switching.py`, `04_sft_patch.py`).
- `src/training/07_dpo_train.py` -- DPO v1, superseded by v2
  (`09_dpo_training.py`, kept as `05_dpo_train.py`).
- RAG-related code (`src/data_pipeline/02_build_rag_db.py`) -- a "Mobile RAG"
  direction is mentioned in the internal project docs as a mentor suggestion
  that was accepted early on, but **the published paper's Methodology
  (Section 3) has no RAG component** — SFT + GRPO + SFT-patch only. Including
  RAG code would misrepresent the paper's method. Dropped.

**Debug / one-off comparison scripts** (not part of reproducing any
paper table): everything in the original `src/debug/`, and most of
`src/inference/` (`compare_inference.py`, `compare_sft_v2_v3.py`,
`grpo_hybrid.py`, `grpo_template.py`, `grpo_zeroshot.py`, `infer_compare.py`,
`infer_compare_all.py`, `infer_sft_adprompt.py`, `sft_patch_fixed.py`,
`sft_patch_inference.py`, `threeway_compare.py`), plus root-level
`compare_captions.py`, `compare_merged.py`, `qwen_captions.py`,
`ten_video_chatml.py`, `ten_video_simpleprompt.py`, `get_free_gpu.sh`.
`plot_grpo_reward.py` was **kept as `results/grpo_reward_curve.png`** (the
already-generated chart) without keeping the plotting script itself, since
it read from a training-server-only trainer-state file path.

**Large / regenerable data, not code** (excluded from git; scripts that
*produce* this output were kept):
- `eval/ocr_bench/ocr_bench_test/` (82 MB Hugging Face dataset cache --
  regenerate with `download_ocr_bench.py`)
- All `*_preds.json` raw model-output dumps across `eval/baselines/`,
  `eval/ablation/`, `eval/textvqa/` (~50 MB total) -- these are
  regenerable by re-running each `run_*.py`; only the small `*_score.txt`
  summaries were considered for inclusion, and only representative small
  JSON result files were kept under `results/`
- `outputs/keyframes_seed77*`, `outputs/*.zip` -- sample extracted frames,
  regenerate via `01_extract_keyframes.py`
- `logs/` (22 MB of raw training/eval console logs) -- historical, not
  needed to *run* the pipeline; kept only as evidence for the
  quantization-command reconstruction (see below)
- `results/archive/` -- explicitly documented in the original
  `results/README.md` as "superseded early-run outputs... kept for
  reference only"

**Internal project-management docs** (not excluded for being wrong, but
because they contain information that shouldn't go on a public repo, and
duplicate/contradict the published paper in places -- e.g. `docs/09_DEPLOYMENT.md`
describes only the Q4_K_M path and predates the paper's final IQ4_NL
result): all of `docs/`, including `docs/00_AI_READ_THIS_FIRST.md` and
`docs/15_MENTOR_UPDATES.md`. These contain:
- an SSH login string with a real username and server IP address
  (`ssh -i ~/.ssh/p16_key cs671_user2@10.8.1.106`)
- your mentor's name and course/team info (CS671, "Team: 9 members",
  "Mentor: ...")
- a decision log written for an AI coding assistant mid-project, not for
  external readers

**If you want to keep any of this history**, put it in a *separate, private*
repo (or a private wiki), not the public release -- and scrub the SSH
string first regardless.

## Naming / consistency issues found and fixed

1. **Teacher model mismatch (Qwen2-VL-7B vs Gemma-4-31B-IT).** The original
   repo's config (`config/paths_config.yaml`: `teacher_id: Qwen/Qwen2-VL-7B-Instruct`)
   and several early scripts describe Qwen2-VL-7B as the teacher. The paper
   (Section 3.1) uses **Gemma-4-31B-IT**. This is a leftover from an earlier
   iteration of the project. Resolved by excluding the Qwen-teacher scripts
   and that config file entirely (see above) — the kept pipeline is
   Gemma-teacher only, matching the paper.
2. **Hardcoded server paths.** Every kept script originally hardcoded
   `/usershome/cs671_user2/p16_blv/...` or `~/p16_blv` or used
   `Path.home() / "p16_blv"`. These were rewritten to relative paths (assumes
   scripts are run from the repo root, same convention the original project
   already followed -- their own docs said "Always cd to project root").
   **This was a mechanical find-and-replace across all files, not a manual
   review of every line** -- re-check each script's path constants before a
   real run, especially ones with several path variables.
3. **`README.md`'s own "Directory Structure" section** (original repo)
   described `models/teacher/` as containing a Qwen2-VL model -- consistent
   with issue #1, and further evidence the Qwen references were stale.
4. **No `requirements.txt` existed anywhere in the original repo.** The one
   in this repo was reconstructed by grepping every `import`/`from` line
   across the kept files -- it has **not** been installed and tested end to
   end. Treat it as a strong first draft, not a verified lockfile (see the
   note at the bottom of `requirements.txt`).
5. **The paper's final mobile-deployment quantization command (IQ4_NL +
   imatrix + Q5_K attention override) was never saved as a script** in the
   original repo -- only `logs/imatrix.log` and `logs/quant_iq4nl.log` (raw
   llama.cpp stdout) exist as evidence it was run. I reconstructed
   `src/deployment/03_quantize_mixed_precision.sh` from the paper text plus
   the log's "applying manual override: iq4_nl -> q5_K" line, but **flagged
   it clearly in its own header as needing your verification** -- this is
   the single highest-value thing to double check before release, since it's
   the paper's headline deployment result (442 MB, 39.3 tok/s on-device).
6. **`docs/09_DEPLOYMENT.md` is stale relative to the paper.** It documents
   the earlier Q4_K_M-only deployment (dated 2026-05-08) and doesn't mention
   IQ4_NL/imatrix/Q5_K at all, even though the paper's Table 4 shows IQ4_NL
   is the final, better result (dated May 24 in Appendix A.4.1). This is
   another reason `docs/` wasn't carried over wholesale -- it would read as
   contradicting the paper.

## Things you still need to do

- [ ] Verify `src/deployment/03_quantize_mixed_precision.sh` against your own
      shell history / llama.cpp version (see its header).
- [ ] Fill in `configs/paths_config.example.yaml` → copy to
      `paths_config.yaml` for your own environment if scripts don't find
      their inputs via relative paths.
- [ ] Host model checkpoints externally (Hugging Face Hub recommended) and
      link them in `README.md` -- nothing in `models/` was ever in this repo.
- [ ] Confirm/replace the placeholder MIT `LICENSE`.
- [ ] Replace the `TODO` BibTeX in `README.md` with the camera-ready citation.
- [ ] Decide whether to keep hardcoded `CUDA_VISIBLE_DEVICES` GPU-index lines
      in various scripts (harmless, but assumes a specific multi-GPU shared
      server layout that outside users won't have) -- left as-is since
      editing dozens of files by regex risked silently changing behavior;
      not a secret, just worth a one-line note for external users.
- [ ] Spot check 2-3 scripts per stage by actually running them in your
      environment -- I verified these are syntactically valid Python and
      cross-checked their logic/hyperparameters against the paper text, but
      I could not execute them here (no GPU, no data, no model weights
      available in this environment).
