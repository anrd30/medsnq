"""
Medical LLM Quantization Script
Converts HuggingFace models to GGUF and quantizes to different bit levels
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "research_models"
HF_MODELS_DIR = MODELS_DIR / "huggingface"
GGUF_MODELS_DIR = MODELS_DIR / "gguf"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"

# Quantization levels to create
QUANT_LEVELS = ["Q4_K_M", "Q2_K"]


def setup_llama_cpp():
    """Clone and build llama.cpp if not present."""
    if LLAMA_CPP_DIR.exists():
        print(f"✓ llama.cpp already exists at {LLAMA_CPP_DIR}")
        return True
    
    print("Cloning llama.cpp...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to clone llama.cpp: {result.stderr}")
        return False
    
    # Build (Windows with CMake)
    print("Building llama.cpp...")
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    
    # CMake configure
    subprocess.run(
        ["cmake", ".."],
        cwd=str(build_dir),
        capture_output=True
    )
    
    # CMake build
    subprocess.run(
        ["cmake", "--build", ".", "--config", "Release"],
        cwd=str(build_dir),
        capture_output=True
    )
    
    print("✓ llama.cpp built successfully")
    return True


def convert_to_gguf(model_name: str) -> Path:
    """Convert a HuggingFace model to GGUF format."""
    hf_path = HF_MODELS_DIR / model_name
    gguf_path = GGUF_MODELS_DIR / f"{model_name}-f16.gguf"
    
    if not hf_path.exists():
        print(f"❌ Model not found: {hf_path}")
        return None
    
    if gguf_path.exists():
        print(f"✓ GGUF already exists: {gguf_path}")
        return gguf_path
    
    print(f"\nConverting {model_name} to GGUF...")
    
    # Use llama.cpp's convert script
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        print(f"❌ Convert script not found: {convert_script}")
        return None
    
    GGUF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [
            sys.executable, 
            str(convert_script),
            str(hf_path),
            "--outfile", str(gguf_path),
            "--outtype", "f16"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        return None
    
    print(f"✓ Converted to: {gguf_path}")
    return gguf_path


def quantize_model(model_name: str, quant_level: str) -> Path:
    """Quantize a GGUF model to specified bit level."""
    input_path = GGUF_MODELS_DIR / f"{model_name}-f16.gguf"
    output_path = GGUF_MODELS_DIR / f"{model_name}-{quant_level}.gguf"
    
    if not input_path.exists():
        print(f"❌ FP16 GGUF not found: {input_path}")
        return None
    
    if output_path.exists():
        print(f"✓ Already quantized: {output_path}")
        return output_path
    
    print(f"\nQuantizing {model_name} to {quant_level}...")
    
    # Find quantize executable
    quantize_exe = None
    for path in [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "llama-quantize.exe",
        LLAMA_CPP_DIR / "quantize.exe",
    ]:
        if path.exists():
            quantize_exe = path
            break
    
    if not quantize_exe:
        print("❌ llama-quantize not found. Build llama.cpp first.")
        return None
    
    result = subprocess.run(
        [str(quantize_exe), str(input_path), str(output_path), quant_level],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Quantization failed: {result.stderr}")
        return None
    
    # Get file sizes for comparison
    input_size = input_path.stat().st_size / (1024 * 1024 * 1024)  # GB
    output_size = output_path.stat().st_size / (1024 * 1024 * 1024)  # GB
    reduction = (1 - output_size / input_size) * 100
    
    print(f"✓ Quantized to: {output_path}")
    print(f"  Size: {input_size:.2f} GB → {output_size:.2f} GB ({reduction:.1f}% reduction)")
    
    return output_path


def process_model(model_name: str):
    """Full pipeline: convert to GGUF, then quantize to all levels."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Step 1: Convert to GGUF
    gguf_path = convert_to_gguf(model_name)
    if not gguf_path:
        return None
    
    # Step 2: Quantize to each level
    results = {"f16": gguf_path}
    
    for level in QUANT_LEVELS:
        quant_path = quantize_model(model_name, level)
        if quant_path:
            results[level] = quant_path
    
    return results


def process_all():
    """Process all downloaded models."""
    print("\n" + "="*60)
    print("Medical LLM Quantization Pipeline")
    print("="*60 + "\n")
    
    # Setup llama.cpp
    if not setup_llama_cpp():
        print("❌ Failed to setup llama.cpp")
        return
    
    # Find downloaded models
    if not HF_MODELS_DIR.exists():
        print(f"❌ No models found. Run download_models.py first.")
        return
    
    models = [d.name for d in HF_MODELS_DIR.iterdir() if d.is_dir()]
    
    if not models:
        print("❌ No models found in HuggingFace directory")
        return
    
    print(f"Found {len(models)} models to process:")
    for m in models:
        print(f"  • {m}")
    
    # Process each
    all_results = {}
    for model in models:
        results = process_model(model)
        if results:
            all_results[model] = results
    
    # Summary
    print("\n" + "="*60)
    print("Quantization Summary")
    print("="*60)
    
    print(f"\n{'Model':<25} {'FP16':<12} {'Q4_K_M':<12} {'Q2_K':<12}")
    print("-" * 60)
    
    for model, results in all_results.items():
        sizes = {}
        for level, path in results.items():
            if path and path.exists():
                sizes[level] = f"{path.stat().st_size / (1024**3):.2f} GB"
            else:
                sizes[level] = "N/A"
        
        print(f"{model:<25} {sizes.get('f16', 'N/A'):<12} {sizes.get('Q4_K_M', 'N/A'):<12} {sizes.get('Q2_K', 'N/A'):<12}")
    
    print(f"\nQuantized models saved to: {GGUF_MODELS_DIR}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize medical LLMs")
    parser.add_argument("--setup", action="store_true", help="Setup llama.cpp")
    parser.add_argument("--model", type=str, help="Process specific model")
    parser.add_argument("--all", action="store_true", help="Process all downloaded models")
    parser.add_argument("--convert-only", action="store_true", help="Only convert to GGUF, skip quantization")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_llama_cpp()
    elif args.model:
        process_model(args.model)
    elif args.all:
        process_all()
    else:
        parser.print_help()
        print("\n\nWorkflow:")
        print("  1. python download_models.py --all")
        print("  2. python quantize_models.py --setup")
        print("  3. python quantize_models.py --all")
