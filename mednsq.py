"""
MedNSQ: Medical Neuron Saliency Quantization
Extends SALF (Semantic-Aware Layer Freezing) for neuron-level mixed-precision quantization.

Novel contribution: Identify medically-salient neurons and preserve them at higher precision
while aggressively quantizing less important neurons.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json


@dataclass
class NeuronSaliency:
    """Stores saliency information for a neuron/parameter."""
    name: str
    layer_idx: int
    saliency_score: float
    shape: tuple
    precision: int = 4  # default 4-bit


class MedNSQ:
    """
    Medical Neuron Saliency Quantization
    
    Workflow:
    1. Load model and medical prompts
    2. Compute neuron saliency via gradient attribution
    3. Rank neurons by medical importance
    4. Assign mixed precision based on saliency
    """
    
    def __init__(self, model_name: str = "BioMistral/BioMistral-7B", device: str = "cuda", load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_name} on {self.device}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load in 4-bit for large models to fit in 6GB VRAM
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            print("Loading in 4-bit quantization for memory efficiency...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.neuron_saliency: Dict[str, torch.Tensor] = {}
        self.precision_map: Dict[str, int] = {}
        
    def get_medical_prompts(self) -> List[Dict]:
        """Sample medical QA prompts for saliency computation."""
        return [
            {"prompt": "A patient presents with fever, headache, and neck stiffness. What is the most likely diagnosis?", "label": "Meningitis"},
            {"prompt": "A 45-year-old diabetic patient has a painless foot ulcer. What is the underlying cause?", "label": "Peripheral neuropathy"},
            {"prompt": "Patient has persistent cough, night sweats, and weight loss. What should be tested?", "label": "Tuberculosis"},
            {"prompt": "Young woman with joint pain, butterfly rash on face, and photosensitivity. Diagnosis?", "label": "Systemic lupus erythematosus"},
            {"prompt": "Child with barking cough and inspiratory stridor. What is this condition?", "label": "Croup"},
            {"prompt": "Patient with sudden severe headache described as worst of life. What to rule out?", "label": "Subarachnoid hemorrhage"},
            {"prompt": "High fever, abdominal pain, rose spots on trunk. What organism?", "label": "Salmonella typhi"},
            {"prompt": "Productive cough with rusty sputum and consolidation on X-ray. Diagnosis?", "label": "Pneumonia"},
            {"prompt": "Patient with palpitations, weight loss, and bulging eyes. What hormonal disorder?", "label": "Hyperthyroidism"},
            {"prompt": "Severe abdominal pain radiating to back after heavy meal. What is it?", "label": "Acute pancreatitis"},
        ]
    
    def compute_neuron_saliency(self, prompts: List[Dict] = None, num_prompts: int = 10) -> Dict[str, torch.Tensor]:
        """
        Compute saliency scores for each neuron using gradient attribution.
        
        Saliency = average |gradient| across medical prompts
        Higher saliency = more important for medical tasks
        """
        if prompts is None:
            prompts = self.get_medical_prompts()[:num_prompts]
        
        print(f"\n{'='*60}")
        print(f"Computing neuron saliency on {len(prompts)} medical prompts")
        print(f"{'='*60}\n")
        
        # Initialize saliency accumulator
        saliency_sum = defaultdict(lambda: None)
        
        self.model.train()  # Enable gradients
        
        for prompt_data in tqdm(prompts, desc="Processing prompts"):
            prompt = prompt_data["prompt"]
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_magnitude = param.grad.abs().detach()
                    
                    if saliency_sum[name] is None:
                        saliency_sum[name] = grad_magnitude.clone()
                    else:
                        saliency_sum[name] += grad_magnitude
        
        # Average across prompts
        self.neuron_saliency = {
            name: (sal / len(prompts)).cpu() 
            for name, sal in saliency_sum.items() 
            if sal is not None
        }
        
        self.model.eval()
        print(f"\n[OK] Computed saliency for {len(self.neuron_saliency)} parameter groups")
        
        return self.neuron_saliency
    
    def rank_neurons(self) -> List[NeuronSaliency]:
        """
        Rank all neurons by their saliency score.
        Returns sorted list from most to least important.
        """
        if not self.neuron_saliency:
            raise ValueError("Run compute_neuron_saliency first!")
        
        rankings = []
        
        for name, saliency_tensor in self.neuron_saliency.items():
            # Get layer index from name
            layer_idx = -1
            for part in name.split('.'):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            
            # Mean saliency for this parameter
            mean_saliency = saliency_tensor.mean().item()
            
            rankings.append(NeuronSaliency(
                name=name,
                layer_idx=layer_idx,
                saliency_score=mean_saliency,
                shape=tuple(saliency_tensor.shape)
            ))
        
        # Sort by saliency (highest first)
        rankings.sort(key=lambda x: x.saliency_score, reverse=True)
        
        return rankings
    
    def assign_mixed_precision(
        self, 
        high_precision_ratio: float = 0.10,
        medium_precision_ratio: float = 0.60,
        high_bits: int = 8,
        medium_bits: int = 4,
        low_bits: int = 2
    ) -> Dict[str, int]:
        """
        Assign precision levels based on saliency ranking.
        
        - Top 10% → 8-bit (preserve medical knowledge)
        - Middle 60% → 4-bit (standard)
        - Bottom 30% → 2-bit (aggressive compression)
        """
        rankings = self.rank_neurons()
        n = len(rankings)
        
        high_cutoff = int(n * high_precision_ratio)
        medium_cutoff = int(n * (high_precision_ratio + medium_precision_ratio))
        
        precision_map = {}
        
        for i, neuron in enumerate(rankings):
            if i < high_cutoff:
                neuron.precision = high_bits
            elif i < medium_cutoff:
                neuron.precision = medium_bits
            else:
                neuron.precision = low_bits
            
            precision_map[neuron.name] = neuron.precision
        
        self.precision_map = precision_map
        
        # Print statistics
        high_count = sum(1 for p in precision_map.values() if p == high_bits)
        med_count = sum(1 for p in precision_map.values() if p == medium_bits)
        low_count = sum(1 for p in precision_map.values() if p == low_bits)
        
        print(f"\n{'='*60}")
        print(f"Mixed Precision Assignment")
        print(f"{'='*60}")
        print(f"  {high_bits}-bit (high precision): {high_count} params ({100*high_count/n:.1f}%)")
        print(f"  {medium_bits}-bit (medium):        {med_count} params ({100*med_count/n:.1f}%)")
        print(f"  {low_bits}-bit (aggressive):     {low_count} params ({100*low_count/n:.1f}%)")
        
        return precision_map
    
    def analyze_medical_layers(self) -> Dict:
        """
        Analyze which layers contain most medical knowledge.
        Returns layer-wise saliency statistics.
        """
        rankings = self.rank_neurons()
        
        layer_stats = defaultdict(list)
        for neuron in rankings:
            if neuron.layer_idx >= 0:
                layer_stats[neuron.layer_idx].append(neuron.saliency_score)
        
        analysis = {}
        for layer_idx, scores in sorted(layer_stats.items()):
            analysis[layer_idx] = {
                "mean_saliency": np.mean(scores),
                "max_saliency": max(scores),
                "num_params": len(scores)
            }
        
        # Find top medical layers
        sorted_layers = sorted(
            analysis.items(), 
            key=lambda x: x[1]["mean_saliency"], 
            reverse=True
        )
        
        print(f"\n{'='*60}")
        print(f"Layer-wise Medical Saliency Analysis")
        print(f"{'='*60}")
        print(f"{'Layer':<10} {'Mean Saliency':<15} {'Max Saliency':<15} {'Params':<10}")
        print("-" * 50)
        
        for layer_idx, stats in sorted_layers[:10]:
            print(f"{layer_idx:<10} {stats['mean_saliency']:<15.6f} {stats['max_saliency']:<15.6f} {stats['num_params']:<10}")
        
        return analysis
    
    def save_precision_map(self, output_path: str):
        """Save precision map to JSON for later use."""
        with open(output_path, 'w') as f:
            json.dump(self.precision_map, f, indent=2)
        print(f"[OK] Saved precision map to {output_path}")
    
    def get_compression_stats(self) -> Dict:
        """Calculate expected compression from mixed precision."""
        if not self.precision_map:
            return {}
        
        original_bits = 16  # FP16
        
        total_params = 0
        weighted_bits = 0
        
        for name, precision in self.precision_map.items():
            if name in self.neuron_saliency:
                num_params = self.neuron_saliency[name].numel()
                total_params += num_params
                weighted_bits += num_params * precision
        
        avg_bits = weighted_bits / total_params if total_params > 0 else 0
        compression_ratio = original_bits / avg_bits if avg_bits > 0 else 1
        
        stats = {
            "total_params": total_params,
            "original_bits": original_bits,
            "average_bits": avg_bits,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1 - 1/compression_ratio) * 100
        }
        
        print(f"\n{'='*60}")
        print(f"Compression Statistics")
        print(f"{'='*60}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Original: {original_bits}-bit")
        print(f"  MedNSQ average: {avg_bits:.2f}-bit")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Size reduction: {stats['size_reduction_percent']:.1f}%")
        
        return stats


def main():
    """Run MedNSQ analysis on MedGemma."""
    print("\n" + "="*60)
    print("MedNSQ: Medical Neuron Saliency Quantization")
    print("="*60 + "\n")
    
    # Initialize with MEDFIT-LLM-3B (smaller medical model, fine-tuned from Llama-3.2)
    mednsq = MedNSQ(
        model_name="adityak74/medfit-llm-3B",
        load_in_4bit=False
    )
    
    # Step 1: Compute neuron saliency
    mednsq.compute_neuron_saliency(num_prompts=10)
    
    # Step 2: Analyze medical layers
    mednsq.analyze_medical_layers()
    
    # Step 3: Assign mixed precision
    mednsq.assign_mixed_precision(
        high_precision_ratio=0.10,
        medium_precision_ratio=0.60
    )
    
    # Step 4: Get compression stats
    mednsq.get_compression_stats()
    
    # Step 5: Save precision map (expected by evaluate_medqa.py)
    mednsq.save_precision_map("mednsq_precision_map.json")
    
    print("\n[OK] Med NSQ analysis complete!")


if __name__ == "__main__":
    main()
