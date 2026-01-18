# MedNSQ: Medical Neuron Saliency Quantization

**Novel neuron-level mixed-precision quantization for medical LLMs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MedNSQ extends SALF (Semantic-Aware Layer Freezing) from training-time layer selection to **inference-time neuron-level quantization**. By identifying medically-salient neurons via gradient attribution, we achieve superior compression compared to uniform quantization.



## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/anrd30/medsnq.git
cd medsnq

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch transformers tqdm numpy
```

### 2. Run MedNSQ Analysis

```bash
# Step 1: Compute neuron saliency on medical prompts
python mednsq.py

# Step 2: Evaluate on MedQA benchmark
python evaluate_medqa.py
```

**Expected output:**
- `mednsq_precision_map.json` - Neuron-to-precision mapping
- `medqa_evaluation_results.json` - Benchmark results

---

## Full Validation Guide (For Peer Reviewers)

### Prerequisites

- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, 4050, or better)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher
- **CUDA**: 11.7+ (for GPU support)

### Step-by-Step Validation

#### Phase 1: Environment Setup

```bash
# 1. Clone & enter directory
git clone https://github.com/anrd30/medsnq.git
cd medsnq

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tqdm numpy accelerate

# 5. Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True
```

#### Phase 2: Run MedNSQ Pipeline

```bash
# Step 1: Compute neuron saliency (takes ~5 minutes on GPU)
python mednsq.py

# Expected output:
# - "Computing neuron saliency on 10 medical prompts"
# - "✓ Computed saliency for 201 parameter groups"
# - "Compression ratio: 4.98x"
# - Creates: mednsq_precision_map.json
```

#### Phase 3: Evaluate on MedQA

```bash
# Run evaluation with 20 sample questions
python evaluate_medqa.py

# Expected results:
# Baseline (FP16):  25.0% accuracy
# Uniform Q4:       15.0% accuracy  
# MedNSQ:           30.0% accuracy ✅
```

#### Phase 4: Full MedQA Validation (Optional)

To validate on the **complete MedQA dataset**:

```bash
# 1. Download full MedQA dataset
# Visit: https://github.com/jind11/MedQA
# Or use: huggingface datasets

# 2. Modify evaluate_medqa.py to load full dataset
# Replace MEDQA_QUESTIONS list with:
from datasets import load_dataset
medqa = load_dataset("medqa", "en")
# Process medqa['test'] samples

# 3. Run full evaluation (may take 1-2 hours)
python evaluate_medqa.py --full-dataset
```

**Expected on full MedQA:**
- Baseline: ~35-40% accuracy
- Uniform Q4: ~25-30% accuracy
- **MedNSQ: ~38-42% accuracy** (hypothesis)

---

## Project Structure

```
medsnq/
├── mednsq.py                    # Core MedNSQ implementation
├── evaluate_medqa.py            # Evaluation script
├── mednsq_precision_map.json   # Generated precision map
├── medqa_evaluation_results.json # Benchmark results
├── salf/                        # Original SALF code (reference)
├── .venv/                       # Virtual environment
└── README.md                    # This file
```

---

## Methodology

### 1. Neuron Saliency Computation

```python
for medical_prompt in prompts:
    loss = model(prompt).loss
    loss.backward()
    
    # Accumulate gradient magnitudes per neuron
    for param in model.parameters():
        saliency[param] += param.grad.abs()
```

### 2. Mixed Precision Assignment

```python
# Rank neurons by medical saliency
rankings = sorted(neurons, key=lambda x: x.saliency, reverse=True)

# Assign precision based on importance
Top 10%:    8-bit  # Medical knowledge neurons
Middle 60%: 4-bit  # Standard quantization
Bottom 30%: 2-bit  # Aggressive compression
```

### 3. Evaluation

Simulate quantization via noise injection proportional to precision:
```python
noise_factor = (16 - assigned_bits) / 16
param += torch.randn_like(param) * param.std() * noise_factor
```

---

## Extending to Larger Models

To test on larger models (e.g., BioMistral-7B, Phi-3):

```python
# In mednsq.py, modify main():
mednsq = MedNSQ(
    model_name="BioMistral/BioMistral-7B",  # or "microsoft/Phi-3-mini-4k-instruct"
    load_in_4bit=True  # Enable 4-bit loading for 6GB VRAM
)
```

**Note:** Larger models require more storage (~7-14GB download).

---

## Troubleshooting

### "CUDA not available"
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Out of memory"
```bash
# Use smaller batch size or reduce num_prompts in mednsq.py
mednsq.compute_neuron_saliency(num_prompts=5)  # Reduce from 10
```

### "Model download interrupted"
```bash
# Models are cached in ~/.cache/huggingface/
# Resume will pick up from there automatically
```

---

## Citation

If you use MedNSQ in your research, please cite:

```bibtex
@article{mednsq2026,
  title={MedNSQ: Medical Neuron Saliency Quantization for Edge Deployment},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Limitations

- **Simulated quantization**: Uses noise injection, not actual GGUF quantization
- **Small sample**: Validated on 20 MedQA questions (full dataset recommended)
- **Model size**: Tested on TinyLlama-1.1B (larger models need validation)

---

## Future Work

- Actual GGUF/GPTQ quantization implementation
- Attention head-level analysis
- Module-level (Attention vs FFN) importance
- Real edge device benchmarks (Raspberry Pi, Android)

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or collaboration:
- **GitHub**: [anrd30](https://github.com/anrd30)
- **Issues**: [GitHub Issues](https://github.com/anrd30/medsnq/issues)

---

## Acknowledgments

- SALF paper: [Semantic-Aware Layer Freezing](https://arxiv.org/abs/2406.11753)
- TinyLlama: [TinyLlama Project](https://github.com/jzhang38/TinyLlama)
- MedQA dataset: [Jin et al., 2021](https://github.com/jind11/MedQA)
