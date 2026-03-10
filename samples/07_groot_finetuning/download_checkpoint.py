#!/usr/bin/env python3
"""
Sample 07 — Download GR00T N1.6 Checkpoint

Downloads the GR00T N1.6 base model from HuggingFace Hub,
verifies the files, and inspects the model architecture.

Requirements:
    pip install huggingface_hub torch
    huggingface-cli login  # Authenticate first

Usage:
    python samples/07_groot_finetuning/download_checkpoint.py
    python samples/07_groot_finetuning/download_checkpoint.py --model nvidia/GR00T-N1-2B
    python samples/07_groot_finetuning/download_checkpoint.py --cache-dir /data/models
"""

import argparse
import os
import sys
from pathlib import Path


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download GR00T N1.6 from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier (e.g. "nvidia/GR00T-N1-2B").
        cache_dir: Optional custom cache directory.

    Returns:
        Path to the downloaded model snapshot.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"📥 Downloading {model_id}...")
    print("   This may take several minutes on first download (~6GB).")
    print()

    kwargs = {"repo_id": model_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    model_path = snapshot_download(**kwargs)
    print(f"✅ Model downloaded to: {model_path}")
    return Path(model_path)


def inspect_model_files(model_path: Path) -> None:
    """List and categorize all files in the checkpoint."""
    print("\n📂 Model Files:")
    print("=" * 60)

    total_size = 0
    file_categories = {
        "weights": [],   # .safetensors, .bin
        "config": [],    # .json, .yaml
        "tokenizer": [], # tokenizer*, spm*
        "other": [],
    }

    for f in sorted(model_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            rel = f.relative_to(model_path)

            if f.suffix in (".safetensors", ".bin", ".pt"):
                file_categories["weights"].append((rel, size_mb))
            elif f.suffix in (".json", ".yaml", ".yml"):
                file_categories["config"].append((rel, size_mb))
            elif "tokenizer" in f.name.lower() or f.suffix in (".model",):
                file_categories["tokenizer"].append((rel, size_mb))
            else:
                file_categories["other"].append((rel, size_mb))

    for category, files in file_categories.items():
        if files:
            print(f"\n  {category.upper()}:")
            for rel, size_mb in files:
                if size_mb > 1:
                    print(f"    {rel:<50} {size_mb:>8.1f} MB")
                else:
                    print(f"    {rel:<50} {size_mb * 1024:>8.1f} KB")

    print(f"\n  {'TOTAL:':<50} {total_size:>8.1f} MB")
    print(f"  {'TOTAL:':<50} {total_size / 1024:>8.2f} GB")


def inspect_architecture(model_path: Path) -> None:
    """Inspect the GR00T model architecture from config files.

    Reads config.json and prints key architectural details without
    loading the full model (no GPU needed).
    """
    import json

    print("\n🧠 Model Architecture:")
    print("=" * 60)

    config_file = model_path / "config.json"
    if not config_file.exists():
        print("  ⚠️  No config.json found — architecture details unavailable")
        print("  (This is common for N1.5 checkpoints)")
        return

    with open(config_file) as f:
        config = json.load(f)

    # Print top-level architecture type
    model_type = config.get("model_type", config.get("architectures", ["Unknown"])[0])
    print(f"  Model type:      {model_type}")
    print(f"  Architectures:   {config.get('architectures', ['N/A'])}")

    # Vision encoder details
    vision_cfg = config.get("vision_config", {})
    if vision_cfg:
        print("\n  🔭 Vision Encoder (Eagle-Block2A-2B-v2 / SigLIP2):")
        print(f"     Hidden size:    {vision_cfg.get('hidden_size', 'N/A')}")
        print(f"     Intermediate:   {vision_cfg.get('intermediate_size', 'N/A')}")
        print(f"     Num layers:     {vision_cfg.get('num_hidden_layers', 'N/A')}")
        print(f"     Attention heads:{vision_cfg.get('num_attention_heads', 'N/A')}")
        print(f"     Image size:     {vision_cfg.get('image_size', 'N/A')}")
        print(f"     Patch size:     {vision_cfg.get('patch_size', 'N/A')}")

    # LLM backbone details
    text_cfg = config.get("text_config", config)
    num_layers = text_cfg.get("num_hidden_layers", config.get("num_hidden_layers"))
    if num_layers:
        print("\n  📝 Language Model (Qwen3):")
        print(f"     Hidden size:    {text_cfg.get('hidden_size', config.get('hidden_size', 'N/A'))}")
        print(f"     Intermediate:   {text_cfg.get('intermediate_size', config.get('intermediate_size', 'N/A'))}")
        print(f"     Num layers:     {num_layers}")
        print(f"     Attention heads:{text_cfg.get('num_attention_heads', config.get('num_attention_heads', 'N/A'))}")
        print(f"     Vocab size:     {text_cfg.get('vocab_size', config.get('vocab_size', 'N/A'))}")
        print(f"     QK norm:        {text_cfg.get('qk_norm', config.get('qk_norm', 'N/A'))}")

    # Diffusion head details
    diffusion_cfg = config.get("diffusion_config", {})
    if diffusion_cfg:
        print("\n  🎲 Diffusion Head:")
        print(f"     Denoise steps:  {diffusion_cfg.get('num_denoising_steps', 4)}")
        print(f"     Action horizon: {diffusion_cfg.get('action_horizon', 16)}")

    # Parameter count estimate
    _estimate_parameters(config)


def _estimate_parameters(config: dict) -> None:
    """Estimate total parameter count from config dimensions."""
    hidden = config.get("hidden_size", 0)
    _intermediate = config.get("intermediate_size", 0)  # noqa: F841
    num_layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)

    if hidden and num_layers:
        # Rough estimate: each transformer layer ~= 12 * hidden^2
        # Plus embeddings: vocab * hidden
        layer_params = 12 * hidden * hidden * num_layers
        embed_params = vocab * hidden if vocab else 0
        total_est = layer_params + embed_params

        print("\n  📊 Estimated Parameters:")
        print(f"     LLM layers:     ~{layer_params / 1e9:.2f}B")
        if embed_params:
            print(f"     Embeddings:     ~{embed_params / 1e9:.2f}B")
        print(f"     Estimated total:~{total_est / 1e9:.2f}B (excludes vision + diffusion)")


def estimate_vram(model_path: Path) -> None:
    """Estimate VRAM requirements for different use cases."""
    # Calculate total weight size
    total_bytes = sum(
        f.stat().st_size
        for f in model_path.rglob("*")
        if f.suffix in (".safetensors", ".bin", ".pt")
    )
    total_gb = total_bytes / (1024**3)

    print("\n💾 VRAM Requirements (estimated):")
    print("=" * 60)
    print(f"  Model weights:           {total_gb:.1f} GB")
    print(f"  Inference (fp16):        ~{total_gb * 1.2:.1f} GB  (weights + activations)")
    print(f"  Inference (fp32):        ~{total_gb * 2.4:.1f} GB")
    print(f"  Fine-tune (projector):   ~{total_gb * 2.0:.1f} GB  (weights + gradients)")
    print(f"  Fine-tune (full):        ~{total_gb * 4.0:.1f} GB  (weights + gradients + optimizer)")
    print()
    print("  Recommended GPUs:")
    print("    Inference:    L4 (24GB), RTX 4090 (24GB)")
    print("    Fine-tune:    L40S (46GB), A100 (80GB), H100 (80GB)")
    print("    Full train:   Thor (132GB) or multi-GPU")


def main():
    parser = argparse.ArgumentParser(
        description="Download and inspect GR00T N1.6 model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="nvidia/GR00T-N1-2B",
        help="HuggingFace model ID (default: nvidia/GR00T-N1-2B)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Custom cache directory for model download",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, inspect existing cache only",
    )
    parser.add_argument(
        "--local-path", default=None,
        help="Path to already-downloaded checkpoint (skips download)",
    )
    args = parser.parse_args()

    print("🧠 GR00T N1.6 — Model Inspector")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print()

    # Get model path
    if args.local_path:
        model_path = Path(args.local_path)
        if not model_path.exists():
            print(f"❌ Path not found: {model_path}")
            sys.exit(1)
    elif args.skip_download:
        # Try to find in HF cache
        cache_dir = args.cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        model_dir_name = "models--" + args.model.replace("/", "--")
        model_cache = Path(cache_dir) / model_dir_name
        if model_cache.exists():
            # Find the latest snapshot
            snapshots = model_cache / "snapshots"
            if snapshots.exists():
                latest = max(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
                model_path = latest
            else:
                model_path = model_cache
        else:
            print("❌ Model not found in cache. Run without --skip-download first.")
            sys.exit(1)
    else:
        model_path = download_model(args.model, args.cache_dir)

    # Inspect
    inspect_model_files(model_path)
    inspect_architecture(model_path)
    estimate_vram(model_path)

    print("\n✅ Done! Next steps:")
    print(f"   python samples/07_groot_finetuning/inference_demo.py --model {model_path}")


if __name__ == "__main__":
    main()
