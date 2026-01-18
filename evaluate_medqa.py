"""
MedQA Evaluation: Compare TinyLlama with different quantization strategies

This script tests:
1. Baseline FP16 (no quantization)
2. Uniform Q4 (simple baseline)
3. MedNSQ (our approach)

Note: For proof-of-concept, we simulate quantization effects rather than 
actually quantizing (which requires GGUF conversion).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from typing import List, Dict
import numpy as np

def load_full_medqa():
    """Load full MedQA dataset from bigbio."""
    import json
    import os
    from pathlib import Path
    
    # Direct path to cached US test file (we found this earlier)
    cache_base = Path.home() / ".cache" / "huggingface" / "datasets" / "downloads" / "extracted"
    
    # Find the MedQA extraction directory
    test_file = None
    if cache_base.exists():
        for extracted_dir in cache_base.iterdir():
            if extracted_dir.is_dir():
                us_test = extracted_dir / "data_clean" / "questions" / "US" / "test.jsonl"
                if us_test.exists():
                    test_file = us_test
                    break
    
    if test_file and test_file.exists():
        print(f"[INFO] Loading questions from cached file: {test_file}")
        questions = []
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        # Extract question, options, answer
                        question_text = item.get('question', '')
                        if not question_text:
                            continue
                        
                        # Options is already a dict with 'A', 'B', 'C', 'D', etc.
                        options_dict = item.get('options', {})
                        if not options_dict or len(options_dict) < 4:
                            continue
                        
                        # Take first 4 options (A, B, C, D)
                        options = {k: options_dict[k] for k in ['A', 'B', 'C', 'D'] if k in options_dict}
                        if len(options) < 4:
                            continue
                        
                        # Get answer - answer_idx is a string like 'C'
                        answer_letter = item.get('answer_idx', '')
                        if not answer_letter or answer_letter not in ['A', 'B', 'C', 'D']:
                            # Try to get from 'answer' field
                            answer_data = item.get('answer', '')
                            if isinstance(answer_data, str) and len(answer_data) == 1:
                                answer_letter = answer_data.upper()
                            elif isinstance(answer_data, list) and len(answer_data) > 0:
                                answer_letter = chr(65 + int(answer_data[0])) if isinstance(answer_data[0], int) else str(answer_data[0]).upper()
                            else:
                                continue
                        
                        if answer_letter not in ['A', 'B', 'C', 'D']:
                            continue
                        
                        questions.append({
                            "question": question_text,
                            "options": options,
                            "answer": answer_letter,
                            "explanation": item.get('explanation', '')
                        })
                    except Exception as e:
                        # Skip malformed items
                        continue
            
            print(f"[OK] Loaded {len(questions)} valid questions from cache")
            return questions
        except Exception as e:
            print(f"[ERROR] Failed to read cached file: {e}")
    
    # Fallback: try loading via datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset('bigbio/med_qa', 'med_qa_en_bigbio_qa')
        
        questions = []
        test_split = ds.get('test', None)
        if test_split is None:
            for key in ds.keys():
                if len(ds[key]) > 0:
                    test_split = ds[key]
                    break
        
        if test_split is None:
            print("[ERROR] No test split found in dataset")
            return []
        
        print(f"[INFO] Loading {len(test_split)} questions from MedQA dataset...")
        
        for item in test_split:
            try:
                question_text = item.get('question', '')
                if not question_text:
                    continue
                    
                choices = item.get('choices', [])
                if not choices or len(choices) < 4:
                    continue
                
                answer_data = item.get('answer', [])
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_idx = answer_data[0] if isinstance(answer_data[0], int) else 0
                elif isinstance(answer_data, str):
                    answer_idx = ord(answer_data.upper()) - ord('A') if len(answer_data) == 1 else 0
                else:
                    answer_idx = 0
                
                if answer_idx < 0 or answer_idx >= len(choices):
                    answer_idx = 0
                
                questions.append({
                    "question": question_text,
                    "options": {
                        chr(65+i): choice for i, choice in enumerate(choices[:4])
                    },
                    "answer": chr(65 + answer_idx),
                    "explanation": item.get('explanation', '')
                })
            except Exception:
                continue
        
        print(f"[OK] Loaded {len(questions)} valid questions")
        return questions
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return []

# Load full MedQA dataset
MEDQA_QUESTIONS = load_full_medqa()

# Fallback to sample questions if dataset loading failed
if not MEDQA_QUESTIONS:
    print("[WARNING] Using sample questions as fallback")
    MEDQA_QUESTIONS = [
    {
        "question": "A 55-year-old woman with type 2 diabetes presents with a painless foot ulcer. Physical examination shows decreased sensation in both feet. What is the most likely underlying cause?",
        "options": {
            "A": "Venous insufficiency",
            "B": "Arterial insufficiency", 
            "C": "Peripheral neuropathy",
            "D": "Trauma"
        },
        "answer": "C",
        "explanation": "Diabetic peripheral neuropathy leads to loss of protective sensation"
    },
    {
        "question": "A 28-year-old man presents with burning on urination and purulent urethral discharge. Gram stain shows gram-negative intracellular diplococci. What is the most likely causative organism?",
        "options": {
            "A": "Chlamydia trachomatis",
            "B": "Neisseria gonorrhoeae",
            "C": "Treponema pallidum",
            "D": "Escherichia coli"
        },
        "answer": "B",
        "explanation": "Gram-negative diplococci = Gonorrhea"
    },
    {
        "question": "A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. Physical examination reveals dry skin and bradycardia. Which of the following is the most likely diagnosis?",
        "options": {
            "A": "Hyperthyroidism",
            "B": "Hypothyroidism",
            "C": "Diabetes mellitus",
            "D": "Cushing syndrome"
        },
        "answer": "B",
        "explanation": "Classic hypothyroidism symptoms"
    },
    {
        "question": "A 60-year-old man with a history of smoking presents with hemoptysis and weight loss. Chest X-ray shows a mass in the right lung. Which type of lung cancer is most commonly associated with smoking?",
        "options": {
            "A": "Adenocarcinoma",
            "B": "Squamous cell carcinoma",
            "C": "Small cell carcinoma",
            "D": "Large cell carcinoma"
        },
        "answer": "B",
        "explanation": "Squamous cell strongly linked to smoking"
    },
    {
        "question": "A 25-year-old woman presents with palpitations, weight loss, and heat intolerance. Physical examination reveals exophthalmos and diffusely enlarged thyroid. What is the most likely diagnosis?",
        "options": {
            "A": "Hashimoto thyroiditis",
            "B": "Graves disease",
            "C": "Toxic multinodular goiter",
            "D": "Thyroid carcinoma"
        },
        "answer": "B",
        "explanation": "Exophthalmos pathognomonic for Graves"
    },
    {
        "question": "A 70-year-old man presents with progressive memory loss and difficulty performing daily tasks. MRI shows generalized cortical atrophy. Which neurotransmitter system is most affected?",
        "options": {
            "A": "Dopaminergic",
            "B": "Serotonergic",
            "C": "Cholinergic",
            "D": "GABAergic"
        },
        "answer": "C",
        "explanation": "Alzheimer's affects cholinergic system"
    },
    {
        "question": "A 35-year-old woman presents with joint pain, malar rash, and photosensitivity. Lab shows positive ANA and anti-dsDNA antibodies. What is the most likely diagnosis?",
        "options": {
            "A": "Rheumatoid arthritis",
            "B": "Systemic lupus erythematosus",
            "C": "Scleroderma",
            "D": "Dermatomyositis"
        },
        "answer": "B",
        "explanation": "Classic SLE presentation"
    },
    {
        "question": "A newborn presents with projectile vomiting after feeding. Physical exam shows palpable olive-shaped mass in right upper quadrant. What is the diagnosis?",
        "options": {
            "A": "Intussusception",
            "B": "Pyloric stenosis",
            "C": "Hirschsprung disease",
            "D": "Duodenal atresia"
        },
        "answer": "B",
        "explanation": "Olive mass = hypertrophied pylorus"
    },
    {
        "question": "A 50-year-old alcoholic presents with confusion, ataxia, and ophthalmoplegia. Which vitamin deficiency is most likely?",
        "options": {
            "A": "Vitamin B12",
            "B": "Vitamin B1 (Thiamine)",
            "C": "Vitamin B6",
            "D": "Folate"
        },
        "answer": "B",
        "explanation": "Wernicke encephalopathy from thiamine deficiency"
    },
    {
        "question": "A 40-year-old woman presents with sudden 'worst headache of my life'. What is the most important initial diagnostic test?",
        "options": {
            "A": "MRI brain",
            "B": "CT head without contrast",
            "C": "Lumbar puncture",
            "D": "Cerebral angiography"
        },
        "answer": "B",
        "explanation": "Rule out subarachnoid hemorrhage with CT first"
    },
    {
        "question": "A 5-year-old child presents with honey-crusted lesions on face. What is the most likely organism?",
        "options": {
            "A": "Streptococcus pyogenes",
            "B": "Staphylococcus epidermidis",
            "C": "Candida albicans",
            "D": "Pseudomonas aeruginosa"
        },
        "answer": "A",
        "explanation": "Impetigo caused by Strep pyogenes"
    },
    {
        "question": "A patient with chronic kidney disease presents with bone pain and elevated PTH. What is the most likely diagnosis?",
        "options": {
            "A": "Primary hyperparathyroidism",
            "B": "Secondary hyperparathyroidism",
            "C": "Osteoporosis",
            "D": "Multiple myeloma"
        },
        "answer": "B",
        "explanation": "CKD leads to secondary hyperPTH"
    },
    {
        "question": "A 30-year-old presents with recurrent kidney stones. Lab shows hypercalcemia and low PTH. What is the most likely cause?",
        "options": {
            "A": "Primary hyperparathyroidism",
            "B": "Vitamin D intoxication",
            "C": "Sarcoidosis",
            "D": "Cancer"
        },
        "answer": "D",
        "explanation": "Hypercalcemia with low PTH suggests malignancy"
    },
    {
        "question": "A patient presents with microcytic anemia and koilonychia (spoon nails). What is the deficiency?",
        "options": {
            "A": "Iron",
            "B": "Vitamin B12",
            "C": "Folate",
       

     "D": "Vitamin C"
        },
        "answer": "A",
        "explanation": "Iron deficiency causes microcytic anemia + koilonychia"
    },
    {
        "question": "A patient with HIV presents with multiple purple skin lesions. Biopsy shows spindle cells. What is the diagnosis?",
        "options": {
            "A": "Melanoma",
            "B": "Kaposi sarcoma",
            "C": "Bacillary angiomatosis",
            "D": "Lymphoma"
        },
        "answer": "B",
        "explanation": "Kaposi sarcoma in AIDS patient"
    },
    {
        "question": "A patient presents with painful swollen big toe and uric acid crystals on joint aspiration. What is the treatment?",
        "options": {
            "A": "Allopurinol",
            "B": "NSAIDs or colchicine",
            "C": "Antibiotics",
            "D": "Corticosteroids only"
        },
        "answer": "B",
        "explanation": "Acute gout treated with NSAIDs/colchicine first"
    },
    {
        "question": "A pregnant woman at 20 weeks has elevated AFP and polyhydramnios. Ultrasound shows absence of cranial vault. What is the diagnosis?",
        "options": {
            "A": "Spina bifida",
            "B": "Anencephaly",
            "C": "Down syndrome",
            "D": "Edwards syndrome"
        },
        "answer": "B",
        "explanation": "Anencephaly causes high AFP + polyhydramnios"
    },
    {
        "question": "A patient presents with painful vesicular rash in a dermatomal distribution. What is the treatment?",
        "options": {
            "A": "Acyclovir",
            "B": "Antibiotics",
            "C": "Corticosteroids",
            "D": "Antifungals"
        },
        "answer": "A",
        "explanation": "Shingles (VZV reactivation) treated with acyclovir"
    },
    {
        "question": "A patient with COPD presents with worsening shortness of breath and purulent sputum. What is the most appropriate treatment?",
        "options": {
            "A": "Increase oxygen",
            "B": "Antibiotics and bronchodilators",
            "C": "Steroids only",
            "D": "Intubation"
        },
        "answer": "B",
        "explanation": "COPD exacerbation with infection needs antibiotics + bronchodilators"
    },
    {
        "question": "A child presents with barking cough and inspiratory stridor. What is the most likely diagnosis?",
        "options": {
            "A": "Epiglottitis",
            "B": "Croup",
            "C": "Bronchiolitis",
            "D": "Pneumonia"
        },
        "answer": "B",
        "explanation": "Barking cough = croup (laryngotracheobronchitis)"
    }
]


def evaluate_model(model, tokenizer, questions: List[Dict], device: str = "cuda"):
    """
    Evaluate model on MedQA questions.
    Returns accuracy and per-question results.
    """
    correct = 0
    results = []
    
    model.eval()
    
    for q in tqdm(questions, desc="Evaluating"):
        # Format prompt
        prompt = f"""You are a medical expert. Answer the following multiple choice question.

Question: {q['question']}

Options:
A) {q['options']['A']}
B) {q['options']['B']}
C) {q['options']['C']}
D) {q['options']['D']}

Answer with ONLY the letter (A, B, C, or D):"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip().upper()
        
        # Extract answer
        predicted = None
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                predicted = char
                break
        
        # Check correctness
        is_correct = (predicted == q['answer'])
        if is_correct:
            correct += 1
        
        results.append({
            "question": q['question'][:50] + "...",
            "correct_answer": q['answer'],
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:100]
        })
    
    accuracy = correct / len(questions)
    return accuracy, results


def simulate_quantization_impact(model, precision_map: Dict[str, int], noise_scale: float = 0.01):
    """
    Simulate quantization by adding noise to parameters.
    Lower precision = more noise.
    
    This is a PROXY for actual quantization (which requires GGUF conversion).
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in precision_map:
                bits = precision_map[name]
                
                # Noise increases as precision decreases
                # 16-bit: 0 noise, 8-bit: small, 4-bit: medium, 2-bit: large
                noise_factor = (16 - bits) / 16
                noise = torch.randn_like(param) * param.std() * noise_scale * noise_factor
                
                param.add_(noise)
    
    return model


def main():
    print("\n" + "="*60)
    print("MedQA Evaluation: Baseline vs MedNSQ")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading TinyLlama-1.1B...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Load MedNSQ precision map
    print("Loading MedNSQ precision map...")
    try:
        with open("mednsq_precision_map.json", "r") as f:
            mednsq_map = json.load(f)
        print(f"[OK] Loaded precision map with {len(mednsq_map)} parameters\n")
    except FileNotFoundError:
        print("[ERROR] mednsq_precision_map.json not found. Run mednsq.py first.")
        return
    
    # Test 1: Baseline (FP16, no quantization)
    print("\n" + "="*60)
    print("Test 1: Baseline FP16 (No Quantization)")
    print("="*60)
    accuracy_baseline, results_baseline = evaluate_model(
        model, tokenizer, MEDQA_QUESTIONS, device
    )
    print(f"\n[OK] Baseline Accuracy: {accuracy_baseline*100:.1f}%")
    
    # Test 2: Uniform Q4 (simulated)
    print("\n" + "="*60)
    print("Test 2: Uniform 4-bit Quantization (Simulated)")
    print("="*60)
    
    # Create uniform Q4 map
    uniform_q4_map = {name: 4 for name in mednsq_map.keys()}
    
    # Reload model (to reset from baseline)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    model = simulate_quantization_impact(model, uniform_q4_map, noise_scale=0.02)
    accuracy_q4, results_q4 = evaluate_model(
        model, tokenizer, MEDQA_QUESTIONS, device
    )
    print(f"\n[OK] Uniform Q4 Accuracy: {accuracy_q4*100:.1f}%")
    
    # Test 3: MedNSQ (our approach)
    print("\n" + "="*60)
    print("Test 3: MedNSQ Mixed Precision (Simulated)")
    print("="*60)
    
    # Reload model again
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    model = simulate_quantization_impact(model, mednsq_map, noise_scale=0.02)
    accuracy_mednsq, results_mednsq = evaluate_model(
        model, tokenizer, MEDQA_QUESTIONS, device
    )
    print(f"\n[OK] MedNSQ Accuracy: {accuracy_mednsq*100:.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Accuracy':<15} {'Avg Precision':<15}")
    print("-" * 50)
    print(f"{'Baseline (FP16)':<20} {accuracy_baseline*100:<15.1f} {'16-bit':<15}")
    print(f"{'Uniform Q4':<20} {accuracy_q4*100:<15.1f} {'4-bit':<15}")
    print(f"{'MedNSQ':<20} {accuracy_mednsq*100:<15.1f} {'3.21-bit':<15}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if accuracy_mednsq > accuracy_q4:
        improvement = (accuracy_mednsq - accuracy_q4) * 100
        print(f"\n[SUCCESS] MedNSQ OUTPERFORMS Uniform Q4 by {improvement:.1f} percentage points!")
        print(f"   Despite being {3.21/4*100:.1f}% of the size")
        print(f"   This validates our medical-aware approach!")
    elif accuracy_mednsq == accuracy_q4:
        print(f"\n[WARNING] MedNSQ equals Uniform Q4 (both {accuracy_mednsq*100:.1f}%)")
        print(f"   Need more questions or larger model to see difference")
    else:
        decline = (accuracy_q4 - accuracy_mednsq) * 100
        print(f"\n[FAIL] MedNSQ underperforms by {decline:.1f} percentage points")
        print(f"   Medical-aware approach may need refinement")
    
    # Save results
    results_summary = {
        "baseline": {"accuracy": accuracy_baseline, "results": results_baseline},
        "uniform_q4": {"accuracy": accuracy_q4, "results": results_q4},
        "mednsq": {"accuracy": accuracy_mednsq, "results": results_mednsq}
    }
    
    with open("medqa_evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n[OK] Detailed results saved to medqa_evaluation_results.json")


if __name__ == "__main__":
    main()
