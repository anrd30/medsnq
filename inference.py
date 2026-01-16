"""
MedQuant Inference Engine
Handles LLM loading and medical symptom assessment.
Supports: Ollama (primary), llama-cpp-python (fallback), Demo mode
"""

import time
import requests
import json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

# Try to import llama-cpp-python as fallback
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

MODELS_DIR = Path(__file__).parent / "models"
OLLAMA_API = "http://localhost:11434"


@dataclass
class InferenceResult:
    """Result from model inference."""
    response: str
    tokens_generated: int
    inference_time_ms: float
    tokens_per_second: float


def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        resp = requests.get(f"{OLLAMA_API}/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False


def get_ollama_models() -> List[Dict]:
    """Get list of available Ollama models."""
    try:
        resp = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [{"name": m["name"], "size_mb": round(m.get("size", 0) / (1024*1024), 1)} 
                    for m in data.get("models", [])]
    except:
        pass
    return []


class OllamaEngine:
    """Inference engine using Ollama API."""
    
    def __init__(self, model_name: str = "tinyllama:latest"):
        self.model_name = model_name
        self.model_path = f"ollama:{model_name}"
    
    def assess_symptoms(self, symptoms: str, max_tokens: int = 512) -> InferenceResult:
        """Perform medical symptom assessment using Ollama."""
        
        prompt = f"""You are a medical assistant AI. Based on the symptoms described, provide:
1. Possible conditions (list 2-3 most likely)
2. Urgency level (Low/Medium/High)
3. Recommended actions

IMPORTANT: This is for informational purposes only. Always recommend consulting a healthcare professional.

Patient symptoms: {symptoms}

Assessment:"""
        
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{OLLAMA_API}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=120
            )
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                
                # Calculate tokens (approximate)
                tokens = data.get("eval_count", len(text.split()))
                tps = tokens / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
                
                return InferenceResult(
                    response=text,
                    tokens_generated=tokens,
                    inference_time_ms=inference_time_ms,
                    tokens_per_second=tps
                )
            else:
                return InferenceResult(
                    response=f"❌ Ollama error: {response.status_code}",
                    tokens_generated=0,
                    inference_time_ms=0,
                    tokens_per_second=0
                )
                
        except requests.exceptions.Timeout:
            return InferenceResult(
                response="❌ Request timed out. Model may be loading...",
                tokens_generated=0,
                inference_time_ms=0,
                tokens_per_second=0
            )
        except Exception as e:
            return InferenceResult(
                response=f"❌ Error: {str(e)}",
                tokens_generated=0,
                inference_time_ms=0,
                tokens_per_second=0
            )
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "path": self.model_path,
            "size_mb": 0,  # Ollama manages this
            "context_length": 2048,
            "backend": "Ollama"
        }


def find_available_models() -> list:
    """Find all GGUF models in the models directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    models = list(MODELS_DIR.glob("*.gguf"))
    return [{"name": m.name, "path": str(m), "size_mb": round(m.stat().st_size / (1024*1024), 2)} 
            for m in models]


# Demo mode for when no model is available
class DemoEngine:
    """Demo engine that simulates responses without a real model."""
    
    def __init__(self):
        self.model_path = "demo_mode"
    
    def assess_symptoms(self, symptoms: str, max_tokens: int = 256) -> InferenceResult:
        """Generate a demo response."""
        import random
        
        # Simulate processing time
        time.sleep(random.uniform(0.3, 0.8))
        
        # Simple keyword matching for demo
        symptoms_lower = symptoms.lower()
        
        if "fever" in symptoms_lower and "headache" in symptoms_lower:
            response = """**Possible Conditions:**
1. Viral Infection (Common Cold/Flu) - Most likely given fever and headache combination
2. Sinusitis - If accompanied by facial pressure
3. Migraine with fever - Less common

**Urgency Level:** Medium

**Recommended Actions:**
• Rest and stay hydrated
• Take over-the-counter fever reducers (acetaminophen/ibuprofen)
• Monitor temperature - seek care if fever exceeds 103°F (39.4°C)
• Consult a healthcare provider if symptoms persist beyond 3 days

⚠️ *This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.*"""
        elif "chest" in symptoms_lower or "heart" in symptoms_lower:
            response = """**Possible Conditions:**
1. Anxiety/Panic - Common cause of chest discomfort
2. Muscle strain - If related to physical activity
3. Cardiac concerns - Requires evaluation

**Urgency Level:** High

**Recommended Actions:**
• If experiencing severe chest pain, shortness of breath, or pain radiating to arm/jaw - CALL EMERGENCY SERVICES IMMEDIATELY
• For mild symptoms, schedule an urgent appointment with your doctor
• Avoid strenuous activity until evaluated

⚠️ *Chest symptoms should always be evaluated by a healthcare professional promptly.*"""
        else:
            response = f"""**Possible Conditions:**
1. General assessment required - Symptoms described: {symptoms[:50]}...
2. Multiple conditions could match these symptoms
3. Further evaluation recommended

**Urgency Level:** Low to Medium

**Recommended Actions:**
• Document when symptoms started and any patterns
• Note any medications currently being taken
• Schedule an appointment with your primary care provider
• Rest and maintain good hydration

⚠️ *This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.*"""
        
        tokens = len(response.split())
        
        return InferenceResult(
            response=response,
            tokens_generated=tokens,
            inference_time_ms=random.uniform(300, 800),
            tokens_per_second=random.uniform(15, 35)
        )
    
    def get_model_info(self) -> dict:
        return {
            "name": "Demo Mode (No GPU Model)",
            "path": "N/A",
            "size_mb": 0,
            "context_length": 2048,
            "gpu_layers": 0,
            "note": "Running in demo mode - download a real model for actual inference"
        }


if __name__ == "__main__":
    # Test the engine
    print("=== MedQuant Inference Engine Test ===\n")
    
    models = find_available_models()
    if models:
        print(f"Found {len(models)} models:")
        for m in models:
            print(f"  • {m['name']} ({m['size_mb']} MB)")
    else:
        print("No models found. Using demo mode...")
        engine = DemoEngine()
        result = engine.assess_symptoms("I have a fever and headache for 2 days")
        print(f"\nDemo Response:\n{result.response}")
        print(f"\n[Tokens: {result.tokens_generated} | Time: {result.inference_time_ms:.0f}ms | TPS: {result.tokens_per_second:.1f}]")
