#!/usr/bin/env python3
"""
Smol-VL-BLV: Quick Inference Demo

Run the fine-tuned BLV model on your own images to generate
spatially-grounded scene descriptions for blind & low-vision users.

Usage:
    # Single image
    python demo/run_demo_inference.py --image photo.jpg

    # Directory of images
    python demo/run_demo_inference.py --image_dir ./my_frames/

    # With a specific model (local path or HuggingFace ID)
    python demo/run_demo_inference.py --image photo.jpg --model YOUR_ORG/smol-vl-blv

    # CPU-only inference
    python demo/run_demo_inference.py --image photo.jpg --device cpu
"""

import os, sys, json, argparse, glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smol-VL-BLV: Generate BLV scene descriptions from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default="YOUR_ORG/smol-vl-blv",  # <-- PLACEHOLDER
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for inference (default: auto-detect)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index if using CUDA")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Max tokens to generate (default: 200)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    return parser.parse_args()


BLV_PROMPT = (
    "You are a professional audio describer for blind and low-vision (BLV) audiences, "
    "following ITC and Netflix Audio Description standards. "
    "STRICT RULES: "
    "1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. "
    "2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. "
    "3. People: describe by clothing color, position, and movement direction only. "
    "4. Hazards: flag steps, ramps, wet floors, hot surfaces, obstacles, moving people/vehicles. "
    "5. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. "
    "Describe this video scene for a blind user."
)


def load_model(model_path, device, gpu_index):
    """Load model from local path or HuggingFace Hub."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_map = {"":  f"cuda:{gpu_index}"}
        dtype = torch.bfloat16
    else:
        device_map = {"":  "cpu"}
        dtype = torch.float32

    # Check if it's a local path or HF ID
    local_path = Path(model_path)
    if local_path.exists():
        print(f"Loading model from local path: {model_path}")
        load_path = str(local_path)
    else:
        print(f"Downloading model from HuggingFace: {model_path}")
        from huggingface_hub import snapshot_download
        load_path = snapshot_download(model_path, cache_dir="models/.hf_cache")

    processor = AutoProcessor.from_pretrained(load_path)
    model = AutoModelForImageTextToText.from_pretrained(
        load_path, torch_dtype=dtype, device_map=device_map
    ).eval()

    print(f"Model loaded on {device}")
    return model, processor


def generate_description(model, processor, image_path):
    """Generate a BLV scene description for a single image."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": BLV_PROMPT}
    ]}]
    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_in, images=image, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
        )

    description = processor.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()
    return description


def main():
    args = parse_args()

    # Collect image paths
    image_paths = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        image_paths.append(p)
    elif args.image_dir:
        d = Path(args.image_dir)
        if not d.is_dir():
            print(f"❌ Directory not found: {args.image_dir}")
            sys.exit(1)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(sorted(d.glob(ext)))
            image_paths.extend(sorted(d.glob(ext.upper())))
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in image_paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        image_paths = unique
    else:
        print("❌ Provide --image or --image_dir")
        print("   Example: python demo/run_demo_inference.py --image photo.jpg")
        sys.exit(1)

    if not image_paths:
        print("❌ No images found")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print()

    # Load model
    model, processor = load_model(args.model, args.device, args.gpu)
    print()

    # Run inference
    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        print("─" * 50)

        description = generate_description(model, processor, img_path)
        print(description)
        print()

        results.append({
            "image": str(img_path),
            "description": description
        })

    # Optionally save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
