"""
MedQuant - Streamlit Demo Application
Quantized LLMs for Edge Medical Symptom Assessment
"""

import streamlit as st
import uuid
from pathlib import Path

# Import our modules
from database import (
    init_database, seed_medical_conditions, search_symptoms,
    log_interaction, get_interaction_stats, get_registered_models
)
from inference import (
    OllamaEngine, DemoEngine, find_available_models, 
    check_ollama_running, get_ollama_models, LLAMA_AVAILABLE
)

# Page configuration
st.set_page_config(
    page_title="MedQuant - Medical Symptom Checker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Response card */
    .response-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #2d5a87;
        color: #1a1a1a !important;
    }
    
    .response-card * {
        color: #1a1a1a !important;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* KB results */
    .kb-result {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Quantization comparison table */
    .quant-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .quant-table th, .quant-table td {
        padding: 0.5rem;
        text-align: center;
        border-bottom: 1px solid #dee2e6;
    }
    
    .quant-table tr:hover {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None
    if 'use_demo' not in st.session_state:
        st.session_state.use_demo = True


def load_engine(model_name: str = None):
    """Load the inference engine."""
    if model_name and model_name != "Demo Mode":
        engine = OllamaEngine(model_name)
        st.session_state.engine = engine
        st.session_state.use_demo = False
    else:
        st.session_state.engine = DemoEngine()
        st.session_state.use_demo = True


def main():
    # Initialize
    init_session_state()
    init_database()
    seed_medical_conditions()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedQuant</h1>
        <p>Quantized LLMs for Edge Medical Symptom Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check for Ollama models
        ollama_running = check_ollama_running()
        ollama_models = get_ollama_models() if ollama_running else []
        
        if ollama_running and ollama_models:
            st.success("✓ Ollama detected")
            model_options = ["Demo Mode"] + [m['name'] for m in ollama_models]
            
            # Default to a good model if available
            default_idx = 0
            for i, m in enumerate(model_options):
                if "mistral" in m.lower() or "phi" in m.lower():
                    default_idx = i
                    break
            
            selected = st.selectbox(
                "Select Model",
                model_options,
                index=default_idx,
                help="Choose an Ollama model for inference"
            )
            
            if selected != "Demo Mode":
                if st.button("🚀 Load Model", type="primary", use_container_width=True):
                    with st.spinner(f"Loading {selected}..."):
                        load_engine(selected)
                    st.success(f"✓ {selected} loaded!")
                    st.rerun()
        else:
            if not ollama_running:
                st.warning("Ollama not running")
                st.code("ollama serve", language="bash")
            else:
                st.info("No Ollama models found")
            st.info("Using demo mode")
        
        # Current engine info
        st.markdown("---")
        st.markdown("### 📊 Current Engine")
        
        if st.session_state.engine:
            info = st.session_state.engine.get_model_info()
            st.markdown(f"**Model:** {info.get('name', 'Demo')}")
            if info.get('backend'):
                st.markdown(f"**Backend:** {info['backend']}")
            st.markdown(f"**Context:** {info.get('context_length', 2048)} tokens")
            if st.session_state.use_demo:
                st.warning("Running in demo mode")
            else:
                st.success("Real LLM inference active")
        else:
            load_engine()  # Load demo engine
            st.info("Demo mode active")
        
        # Quantization comparison
        st.markdown("---")
        st.markdown("### 📈 Quantization Comparison")
        st.markdown("""
        | Bits | Size | Speed | Quality |
        |:----:|:----:|:-----:|:-------:|
        | 2-bit | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
        | 4-bit | ⭐⭐ | ⭐⭐ | ⭐⭐ |
        | 8-bit | ⭐ | ⭐ | ⭐⭐⭐ |
        """)
        
        # Stats
        st.markdown("---")
        st.markdown("### 📉 Session Stats")
        stats = get_interaction_stats()
        st.metric("Total Queries", stats.get('total_interactions', 0))
        if stats.get('avg_latency'):
            st.metric("Avg Latency", f"{stats['avg_latency']:.0f} ms")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🩺 Symptom Assessment")
        
        # Symptom input
        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="Example: I have had a headache and fever for 2 days. Also feeling tired and have body aches.",
            height=120
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            assess_btn = st.button("🔍 Assess Symptoms", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🎲 Try Example", use_container_width=True):
                symptoms = "I have a fever of 101°F, headache, and body aches for the past 2 days"
                st.session_state.example_symptoms = symptoms
                st.rerun()
        
        # Handle example
        if 'example_symptoms' in st.session_state:
            symptoms = st.session_state.example_symptoms
            del st.session_state.example_symptoms
            assess_btn = True
        
        # Perform assessment
        if assess_btn and symptoms:
            if not st.session_state.engine:
                load_engine()
            
            with st.spinner("Analyzing symptoms..."):
                result = st.session_state.engine.assess_symptoms(symptoms)
                st.session_state.last_response = result
                
                # Log to database
                log_interaction(
                    st.session_state.session_id,
                    symptoms,
                    result.response,
                    result.inference_time_ms,
                    result.tokens_generated,
                    st.session_state.engine.model_path if hasattr(st.session_state.engine, 'model_path') else 'demo'
                )
        
        # Display response
        if st.session_state.last_response:
            result = st.session_state.last_response
            
            st.markdown("---")
            st.markdown("### 📋 Assessment Result")
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result.inference_time_ms:.0f}</div>
                    <div class="metric-label">Latency (ms)</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result.tokens_generated}</div>
                    <div class="metric-label">Tokens</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result.tokens_per_second:.1f}</div>
                    <div class="metric-label">Tokens/sec</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                mode = "Demo" if st.session_state.use_demo else "GPU"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mode}</div>
                    <div class="metric-label">Mode</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Response
            st.markdown(f"""
            <div class="response-card">
                {result.response.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This tool is for educational and informational purposes only. 
                It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
                Always consult a qualified healthcare provider.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📚 Knowledge Base Search")
        
        kb_query = st.text_input("Search medical conditions", placeholder="e.g., fever headache")
        
        if kb_query:
            results = search_symptoms(kb_query)
            
            if results:
                for r in results:
                    severity_color = {
                        'low': '🟢',
                        'medium': '🟡', 
                        'high': '🔴'
                    }.get(r['severity'], '⚪')
                    
                    with st.expander(f"{severity_color} {r['condition_name']}"):
                        st.markdown(f"**Symptoms:** {r['symptoms']}")
                        st.markdown(f"**Description:** {r['description']}")
                        st.markdown(f"**Treatment:** {r['treatment_guidelines']}")
            else:
                st.info("No matching conditions found")
        
        # Database info
        st.markdown("---")
        st.markdown("### 🗄️ DBMS Features")
        st.markdown("""
        - **SQLite** with FTS5 full-text search
        - **Schema**: Models, sessions, interactions, medical KB
        - **Indexes**: Optimized for symptom lookup
        - **Logging**: All interactions stored for analysis
        """)
        
        # Technical details
        with st.expander("📐 Technical Details"):
            st.markdown("""
            **Quantization Levels:**
            - Q2_K: ~2 bits/weight (smallest)
            - Q4_K_M: ~4 bits/weight (balanced)
            - Q8_0: ~8 bits/weight (highest quality)
            
            **Storage Efficiency:**
            - FP16 → Q4: ~75% size reduction
            - FP16 → Q2: ~87% size reduction
            
            **Edge Deployment:**
            - Runs on CPU or GPU
            - SQLite: Zero-config database
            - No internet required
            """)


if __name__ == "__main__":
    main()
