"""
Medical LLM Quantization Research - Model Download Script
Downloads models from HuggingFace for quantization benchmarking
"""

import os
import subprocess
import sys
from pathlib import Path

# Models to download
MODELS = {
    "medgemma-4b": {
        "repo": "google/medgemma-1.5-4b-it",
        "description": "Google MedGemma 1.5 4B Instruct",
        "params": "4B",
        "type": "medical"
    },
    "openbiollm-8b": {
        "repo": "aaditya/Llama3-OpenBioLLM-8B",
        "description": "Llama3 OpenBioLLM 8B",
        "params": "8B",
        "type": "medical"
    },
    "biomistral-7b": {
        "repo": "BioMistral/BioMistral-7B",
        "description": "BioMistral 7B",
        "params": "7B",
        "type": "medical"
    },
    "gemma-2b": {
        "repo": "google/gemma-2-2b-it",
        "description": "Gemma 2 2B Instruct (baseline)",
        "params": "2B",
        "type": "general"
    },
    "tinyllama-1b": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B Chat",
        "params": "1.1B",
        "type": "general"
    }
}

# Directory structure
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "research_models"
HF_MODELS_DIR = MODELS_DIR / "huggingface"
GGUF_MODELS_DIR = MODELS_DIR / "gguf"


def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")
    packages = [
        "transformers",
        "torch",
        "huggingface_hub",
        "accelerate",
        "sentencepiece"
    ]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✓ Dependencies installed")


def download_model(model_key: str):
    """Download a model from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        return None
    
    model_info = MODELS[model_key]
    repo = model_info["repo"]
    output_dir = HF_MODELS_DIR / model_key
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_info['description']}")
    print(f"Repo: {repo}")
    print(f"Parameters: {model_info['params']}")
    print(f"{'='*60}\n")
    
    try:
        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download
        snapshot_download(
            repo_id=repo,
            local_dir=str(output_dir),
            ignore_patterns=["*.md", "*.txt"],  # Skip docs
        )
        
        print(f"\n✓ Downloaded to: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Error downloading {model_key}: {e}")
        return None


def download_all():
    """Download all models."""
    print("\n" + "="*60)
    print("Medical LLM Quantization Research - Model Downloader")
    print("="*60 + "\n")
    
    # Create directories
    HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GGUF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Models directory: {MODELS_DIR}")
    print(f"Models to download: {len(MODELS)}\n")
    
    for key, info in MODELS.items():
        print(f"  • {info['description']} ({info['params']})")
    
    print("\n" + "-"*60)
    
    results = {}
    for key in MODELS:
        path = download_model(key)
        results[key] = path
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for key, path in results.items():
        status = "✓" if path else "❌"
        print(f"  {status} {MODELS[key]['description']}")
    
    return results


def list_models():
    """List available models."""
    print("\nAvailable Models for Download:\n")
    print(f"{'Key':<20} {'Params':<8} {'Type':<10} {'Description'}")
    print("-" * 70)
    for key, info in MODELS.items():
        print(f"{key:<20} {info['params']:<8} {info['type']:<10} {info['description']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download medical LLMs for quantization research")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Download specific model (key)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies first")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    if args.list:
        list_models()
    elif args.model:
        download_model(args.model)
    elif args.all:
        download_all()
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python download_models.py --list")
        print("  python download_models.py --model medgemma-4b")
        print("  python download_models.py --all --install-deps")
