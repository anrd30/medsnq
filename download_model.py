"""
MedQuant Model Downloader
Downloads quantized GGUF models from HuggingFace.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

MODELS_DIR = Path(__file__).parent / "models"

# Available models (small ones for quick demo)
AVAILABLE_MODELS = {
    "tinyllama-1.1b-q4": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B (Q4_K_M) - ~670MB, fast inference",
        "size_mb": 670
    },
    "tinyllama-1.1b-q2": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "description": "TinyLlama 1.1B (Q2_K) - ~400MB, smallest/fastest",
        "size_mb": 400
    },
    "phi3-mini-q4": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Phi-3 Mini 3.8B (Q4) - ~2.3GB, better quality",
        "size_mb": 2300
    },
    "gemma-2b-q4": {
        "repo": "TheBloke/gemma-2b-it-GGUF",
        "filename": "gemma-2b-it.Q4_K_M.gguf",
        "description": "Gemma 2B (Q4_K_M) - ~1.5GB, good balance",
        "size_mb": 1500
    }
}


def download_model(model_key: str) -> str:
    """Download a model from HuggingFace."""
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
        return None
    
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    model = AVAILABLE_MODELS[model_key]
    MODELS_DIR.mkdir(exist_ok=True)
    
    output_path = MODELS_DIR / model["filename"]
    if output_path.exists():
        print(f"✓ Model already exists: {output_path}")
        return str(output_path)
    
    print(f"Downloading {model['description']}...")
    print(f"This may take a few minutes depending on your connection...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["filename"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def list_models():
    """List available models for download."""
    print("\n=== Available Models for Download ===\n")
    for key, model in AVAILABLE_MODELS.items():
        print(f"  [{key}]")
        print(f"    {model['description']}")
        print(f"    Size: ~{model['size_mb']} MB")
        print()


def list_downloaded():
    """List already downloaded models."""
    MODELS_DIR.mkdir(exist_ok=True)
    models = list(MODELS_DIR.glob("*.gguf"))
    
    if not models:
        print("\n❌ No models downloaded yet.")
        return []
    
    print("\n=== Downloaded Models ===\n")
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  • {m.name} ({size_mb:.1f} MB)")
    return models


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key == "list":
            list_models()
            list_downloaded()
        else:
            download_model(model_key)
    else:
        print("MedQuant Model Downloader")
        print("=" * 40)
        list_models()
        
        downloaded = list_downloaded()
        
        if not downloaded:
            print("\n💡 Quick start: Run one of these commands:")
            print("   python download_model.py tinyllama-1.1b-q4   # Recommended for demo")
            print("   python download_model.py tinyllama-1.1b-q2   # Smallest/fastest")
