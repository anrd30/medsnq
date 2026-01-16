# MedQuant - Quantized LLMs for Edge Medical Diagnosis

A demonstration of extremely quantized LLMs (1-4 bit) for medical symptom assessment with efficient SQLite storage.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download a quantized model (run once)
python download_model.py

# Run the demo
streamlit run app.py
```

## Project Structure

```
medquant/
├── app.py              # Streamlit demo application
├── database.py         # SQLite schema and operations
├── inference.py        # LLM inference engine
├── download_model.py   # Model downloader script
├── requirements.txt    # Python dependencies
└── models/             # Quantized model storage
```

## Key Features

- **Quantized LLM Inference**: Uses llama-cpp-python for efficient inference
- **SQLite Storage**: Stores patient interactions, medical KB, and benchmark results
- **Real-time Metrics**: Displays latency, tokens/sec, and storage usage
- **Medical Knowledge Base**: FTS5-powered symptom search

## Research Contribution

This project benchmarks accuracy vs. latency vs. storage tradeoffs for deploying quantized LLMs in resource-constrained medical settings.
