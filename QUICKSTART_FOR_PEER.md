# Quick Start Guide for Peer Validation

**Goal:** Run MedNSQ on full MedQA dataset and validate results

**Time:** ~2-3 hours (including downloads)

**Downloads:** ~2.5GB (TinyLlama 2.2GB + MedQA 300MB)

---

## Step 1: Clone & Setup (10 min)

```bash
# Clone repository
git clone https://github.com/anrd30/medsnq.git
cd medsnq

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers tqdm numpy datasets accelerate

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

**Expected:** `GPU: True`

---

## Step 2: Run MedNSQ Analysis (5-10 min)

```bash
python mednsq.py
```

**What it does:**
- Loads TinyLlama-1.1B
- Computes neuron saliency on 10 medical prompts
- Generates `mednsq_precision_map.json`

**Expected output:**
- "Computing neuron saliency..."
- "Compression ratio: ~5x"
- "✓ Saved precision map"

---

## Step 3: Download Full MedQA Dataset (5 min)

```bash
# Install datasets library (if not already)
pip install datasets

# Download MedQA
python -c "from datasets import load_dataset; ds = load_dataset('bigbio/med_qa', 'med_qa_en_bigbio_qa'); print('Downloaded:', len(ds['test']), 'test questions')"
```

**Expected:** ~1200 test questions downloaded

---

## Step 4: Modify Evaluation Script (Manual)

Edit `evaluate_medqa.py`:

**Find this section** (around line 14):
```python
MEDQA_QUESTIONS = [
    {...},  # Hardcoded questions
    ...
]
```

**Replace with:**
```python
def load_full_medqa():
    from datasets import load_dataset
    ds = load_dataset('bigbio/med_qa', 'med_qa_en_bigbio_qa')
    questions = []
    
    for item in ds['test']:
        # Extract question, options, answer
        question_text = item['question']
        choices = item['choices']
        answer_idx = item['answer'][0] if item['answer'] else 0
        
        questions.append({
            "question": question_text,
            "options": {
                chr(65+i): choice for i, choice in enumerate(choices)
            },
            "answer": chr(65 + answer_idx),
            "explanation": ""
        })
    
    return questions

# Replace MEDQA_QUESTIONS with:
MEDQA_QUESTIONS = load_full_medqa()
```

---

## Step 5: Run Full Evaluation (1-2 hours)

```bash
python evaluate_medqa.py
```

**What it does:**
1. Test 1: Baseline FP16 on all MedQA questions
2. Test 2: Uniform Q4 on all MedQA questions  
3. Test 3: MedNSQ on all MedQA questions

**Expected runtime:**
- ~3 seconds per question
- ~1200 questions = ~1 hour per test
- Total: ~3 hours

---

## Step 6: Check Results

```bash
# Results saved to:
cat medqa_evaluation_results.json

# Look for:
# {
#   "baseline": {"accuracy": X%},
#   "uniform_q4": {"accuracy": Y%},
#   "mednsq": {"accuracy": Z%}
# }
```

**Hypothesis to validate:**
```
MedNSQ accuracy > Uniform Q4 accuracy
Despite MedNSQ having lower average bits (3.21 vs 4.0)
```

---

## Troubleshooting

### GPU Out of Memory
```bash
# Edit mednsq.py, reduce prompts:
mednsq.compute_neuron_saliency(num_prompts=5)
```

### Slow evaluation
```bash
# Test on subset first (edit evaluate_medqa.py):
MEDQA_QUESTIONS = load_full_medqa()[:100]  # First 100 questions
```

### Dataset download fails
```bash
# Use local download
wget https://github.com/jind11/MedQA/archive/refs/heads/master.zip
# Then modify load script to point to local files
```

---

## Expected Outputs

After completion, you should have:

1. `mednsq_precision_map.json` - Neuron precision assignments
2. `medqa_evaluation_results.json` - Full results with:
   - Baseline accuracy on full MedQA
   - Uniform Q4 accuracy
   - MedNSQ accuracy

---

## Report Back

Please share:
1. Final accuracy numbers (Baseline, Q4, MedNSQ)
2. Total runtime
3. Any errors encountered
4. GPU model used

---

## Optional: Test on Larger Model

If you have 10GB+ VRAM:

```bash
# Edit mednsq.py, line ~314:
mednsq = MedNSQ(
    model_name="microsoft/Phi-3-mini-4k-instruct",  # 3.8B model
    load_in_4bit=True
)
```

Run same steps above with Phi-3 instead of TinyLlama.

---

**Questions?** Open an issue: https://github.com/anrd30/medsnq/issues
