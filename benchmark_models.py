"""
Medical LLM Benchmark Script
Evaluates quantized models on MedQA and measures performance metrics
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import csv
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "research_models" / "gguf"
RESULTS_DIR = BASE_DIR / "benchmark_results"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str
    quantization: str
    dataset: str
    accuracy: float
    total_questions: int
    correct_answers: int
    avg_latency_ms: float
    tokens_per_second: float
    memory_mb: float
    model_size_mb: float
    timestamp: str


# Sample MedQA questions for benchmarking
MEDQA_SAMPLES = [
    {
        "question": "A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. Physical examination reveals dry skin and bradycardia. Which of the following is the most likely diagnosis?",
        "options": ["A) Hyperthyroidism", "B) Hypothyroidism", "C) Diabetes mellitus", "D) Cushing syndrome"],
        "answer": "B"
    },
    {
        "question": "A 60-year-old man with a history of smoking presents with hemoptysis and weight loss. Chest X-ray shows a mass in the right lung. Which type of lung cancer is most commonly associated with smoking?",
        "options": ["A) Adenocarcinoma", "B) Squamous cell carcinoma", "C) Small cell carcinoma", "D) Large cell carcinoma"],
        "answer": "B"
    },
    {
        "question": "A 25-year-old woman presents with palpitations, weight loss, and heat intolerance. Physical examination reveals exophthalmos and a diffusely enlarged thyroid. What is the most likely diagnosis?",
        "options": ["A) Hashimoto thyroiditis", "B) Graves disease", "C) Toxic multinodular goiter", "D) Thyroid carcinoma"],
        "answer": "B"
    },
    {
        "question": "A 70-year-old man presents with progressive memory loss and personality changes. MRI shows generalized cortical atrophy. Which neurotransmitter system is most likely affected?",
        "options": ["A) Dopaminergic", "B) Serotonergic", "C) Cholinergic", "D) GABAergic"],
        "answer": "C"
    },
    {
        "question": "A 35-year-old woman presents with joint pain, malar rash, and photosensitivity. Laboratory tests show positive ANA and anti-dsDNA antibodies. What is the most likely diagnosis?",
        "options": ["A) Rheumatoid arthritis", "B) Systemic lupus erythematosus", "C) Scleroderma", "D) Dermatomyositis"],
        "answer": "B"
    },
    {
        "question": "A newborn presents with projectile vomiting after feeding. Physical examination reveals a palpable olive-shaped mass in the right upper quadrant. What is the most likely diagnosis?",
        "options": ["A) Intussusception", "B) Pyloric stenosis", "C) Hirschsprung disease", "D) Duodenal atresia"],
        "answer": "B"
    },
    {
        "question": "A 50-year-old man with a history of alcohol abuse presents with confusion and ataxia. Physical examination reveals ophthalmoplegia. Which vitamin deficiency is most likely responsible?",
        "options": ["A) Vitamin B12", "B) Vitamin B1 (Thiamine)", "C) Vitamin B6", "D) Folate"],
        "answer": "B"
    },
    {
        "question": "A 40-year-old woman presents with sudden onset of severe headache described as 'the worst headache of my life.' What is the most important initial diagnostic test?",
        "options": ["A) MRI brain", "B) CT head without contrast", "C) Lumbar puncture", "D) Cerebral angiography"],
        "answer": "B"
    },
    {
        "question": "A 28-year-old man presents with burning on urination and purulent urethral discharge. Gram stain shows gram-negative intracellular diplococci. What is the most likely causative organism?",
        "options": ["A) Chlamydia trachomatis", "B) Neisseria gonorrhoeae", "C) Treponema pallidum", "D) Escherichia coli"],
        "answer": "B"
    },
    {
        "question": "A 55-year-old woman with type 2 diabetes presents with a painless foot ulcer. What is the most likely underlying cause?",
        "options": ["A) Venous insufficiency", "B) Arterial insufficiency", "C) Peripheral neuropathy", "D) Trauma"],
        "answer": "C"
    }
]


def run_llama_inference(model_path: str, prompt: str, max_tokens: int = 100) -> tuple:
    """
    Run inference using llama.cpp CLI.
    Returns (response_text, latency_ms, tokens_generated)
    """
    # Find llama-cli executable
    llama_exe = None
    for path in [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-cli.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-cli.exe",
        LLAMA_CPP_DIR / "llama-cli.exe",
        LLAMA_CPP_DIR / "main.exe",
    ]:
        if path.exists():
            llama_exe = path
            break
    
    if not llama_exe:
        return None, 0, 0
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(
            [
                str(llama_exe),
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.1",  # Low temp for deterministic answers
                "-ngl", "99",  # Use GPU if available
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        response = result.stdout
        tokens = len(response.split())
        
        return response, latency_ms, tokens
        
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 120000, 0
    except Exception as e:
        return f"ERROR: {e}", 0, 0


def evaluate_answer(response: str, correct_answer: str) -> bool:
    """Check if the model's response contains the correct answer."""
    response_upper = response.upper()
    
    # Check for explicit answer patterns
    patterns = [
        f"THE ANSWER IS {correct_answer}",
        f"ANSWER: {correct_answer}",
        f"CORRECT ANSWER: {correct_answer}",
        f"({correct_answer})",
        f"{correct_answer})",
        f"{correct_answer}.",
    ]
    
    for pattern in patterns:
        if pattern in response_upper:
            return True
    
    # Check if the answer letter appears prominently
    # This is a simple heuristic
    lines = response_upper.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if line.strip().startswith(correct_answer):
            return True
    
    return False


def benchmark_model(model_path: Path, questions: List[Dict]) -> BenchmarkResult:
    """Benchmark a single model on the question set."""
    model_name = model_path.stem.rsplit('-', 1)[0]  # Remove quant suffix
    quant_level = model_path.stem.split('-')[-1]  # Get quant level
    
    correct = 0
    total_latency = 0
    total_tokens = 0
    
    print(f"\n  Testing {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        # Format prompt
        prompt = f"""You are a medical expert. Answer the following multiple choice question by selecting the correct option letter (A, B, C, or D).

Question: {q['question']}

Options:
{chr(10).join(q['options'])}

Answer with just the letter of the correct option:"""
        
        response, latency, tokens = run_llama_inference(str(model_path), prompt)
        
        if response and evaluate_answer(response, q['answer']):
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        total_latency += latency
        total_tokens += tokens
        
        print(f"    Q{i+1}: {status} (latency: {latency:.0f}ms)")
    
    # Calculate metrics
    accuracy = (correct / len(questions)) * 100
    avg_latency = total_latency / len(questions)
    tps = total_tokens / (total_latency / 1000) if total_latency > 0 else 0
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    
    return BenchmarkResult(
        model_name=model_name,
        quantization=quant_level,
        dataset="MedQA-Sample",
        accuracy=accuracy,
        total_questions=len(questions),
        correct_answers=correct,
        avg_latency_ms=avg_latency,
        tokens_per_second=tps,
        memory_mb=0,  # Would need profiling
        model_size_mb=model_size_mb,
        timestamp=datetime.now().isoformat()
    )


def run_all_benchmarks():
    """Run benchmarks on all quantized models."""
    print("\n" + "="*60)
    print("Medical LLM Benchmark Suite")
    print("="*60 + "\n")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all GGUF models
    if not MODELS_DIR.exists():
        print(f"❌ Models directory not found: {MODELS_DIR}")
        print("Run quantize_models.py first.")
        return
    
    models = list(MODELS_DIR.glob("*.gguf"))
    
    if not models:
        print("❌ No GGUF models found")
        return
    
    print(f"Found {len(models)} models to benchmark:")
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  • {m.name} ({size_mb:.1f} MB)")
    
    # Run benchmarks
    results = []
    
    for model_path in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_path.name}")
        print(f"{'='*60}")
        
        result = benchmark_model(model_path, MEDQA_SAMPLES)
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Accuracy: {result.accuracy:.1f}%")
        print(f"    Avg Latency: {result.avg_latency_ms:.0f} ms")
        print(f"    Model Size: {result.model_size_mb:.1f} MB")
    
    # Save results
    results_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    print(f"\n{'Model':<30} {'Quant':<10} {'Accuracy':<10} {'Size (MB)':<12} {'Latency (ms)':<12}")
    print("-" * 74)
    
    for r in sorted(results, key=lambda x: (x.model_name, x.quantization)):
        print(f"{r.model_name:<30} {r.quantization:<10} {r.accuracy:<10.1f} {r.model_size_mb:<12.1f} {r.avg_latency_ms:<12.0f}")


def export_to_csv():
    """Export all benchmark results to CSV for paper."""
    results_files = list(RESULTS_DIR.glob("benchmark_*.json"))
    
    if not results_files:
        print("No benchmark results found")
        return
    
    # Combine all results
    all_results = []
    for f in results_files:
        with open(f) as file:
            all_results.extend(json.load(file))
    
    # Export to CSV
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"✓ Exported to: {csv_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark medical LLMs")
    parser.add_argument("--run", action="store_true", help="Run all benchmarks")
    parser.add_argument("--model", type=str, help="Benchmark specific model")
    parser.add_argument("--export", action="store_true", help="Export results to CSV")
    
    args = parser.parse_args()
    
    if args.run:
        run_all_benchmarks()
    elif args.export:
        export_to_csv()
    else:
        parser.print_help()
        print("\n\nComplete workflow:")
        print("  1. python download_models.py --all --install-deps")
        print("  2. python quantize_models.py --all")
        print("  3. python benchmark_models.py --run")
        print("  4. python benchmark_models.py --export")
