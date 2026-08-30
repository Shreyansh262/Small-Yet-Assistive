#!/usr/bin/env python3
"""
Smol-VL-BLV: Download Pre-Generated Data from HuggingFace

Downloads teacher captions, evaluation splits, and calibration data
from the project's HuggingFace dataset card.

Usage:
    python scripts/download_data.py                    # download everything
    python scripts/download_data.py --split eval       # evaluation data only
    python scripts/download_data.py --split train      # training captions only
    python scripts/download_data.py --split calibration # quantization calibration data
"""

import os, sys, argparse, json
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# TODO: Replace with your actual HuggingFace dataset repository ID
HF_DATASET_REPO = "YOUR_ORG/smol-vl-blv-data"   # <-- PLACEHOLDER: UPDATE THIS
# ──────────────────────────────────────────────────────────────────────────────


def download_from_hf(repo_id, filename, local_dir):
    """Download a single file from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )
    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Download Smol-VL-BLV data from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["all", "eval", "train", "calibration"],
        help="Which data split to download (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generated",
        help="Local directory to save downloaded files (default: data/generated)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define available files on the HF data card
    files = {
        "train": [
            "all_captions_gemma.json",          # teacher captions (Gemma-4-31B-IT)
            "grpo_pairs.json",                  # GRPO preference pairs
        ],
        "eval": [
            "balanced_eval.json",               # BLV evaluation split
        ],
        "calibration": [
            "blv_calibration_500.txt",           # 500 captions for imatrix quantization
        ],
    }

    if args.split == "all":
        splits_to_download = list(files.keys())
    else:
        splits_to_download = [args.split]

    print(f"Downloading from: {HF_DATASET_REPO}")
    print(f"Output directory: {output_dir}")
    print()

    total_files = sum(len(files[s]) for s in splits_to_download)
    downloaded = 0

    for split in splits_to_download:
        print(f"── {split.upper()} ──")
        for filename in files[split]:
            local_path = output_dir / filename
            if local_path.exists():
                print(f"  ✅ {filename} (already exists, skipping)")
                downloaded += 1
                continue

            try:
                print(f"  ⬇️  Downloading {filename}...")
                download_from_hf(HF_DATASET_REPO, filename, str(output_dir))
                print(f"  ✅ {filename}")
                downloaded += 1
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                print(f"     Make sure '{HF_DATASET_REPO}' is accessible.")
        print()

    print(f"Downloaded {downloaded}/{total_files} files to {output_dir}")


if __name__ == "__main__":
    main()
