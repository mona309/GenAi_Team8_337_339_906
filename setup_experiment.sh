#!/bin/bash
# ============================================================
# Setup Script for Q1 Journal Experiment
# Run this on a FRESH GPU system before overnight_experiment.py
# ============================================================

set -e

echo "============================================================"
echo "  Q1 Journal - Text-to-Audio Experiment Setup"
echo "============================================================"

# Detect GPU
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Running on CPU (will be VERY slow)."
fi

# Check Python
echo "[2/6] Checking Python..."
python3 --version || { echo "Python 3 required!"; exit 1; }

# Create directories
echo "[3/6] Creating directories..."
mkdir -p outputs/audio
mkdir -p outputs/experiments
mkdir -p data/reference_audio
mkdir -p src

# Install Python dependencies
echo "[4/6] Installing Python packages..."
pip install --upgrade pip

# Core dependencies
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121 || \
pip install torch>=2.0.0 torchaudio>=2.0.0 || \
echo "Note: PyTorch installation may need CUDA-specific version"

pip install \
    diffusers>=0.30.0 \
    transformers>=4.43.0 \
    accelerate>=0.30.0 \
    laion-clap>=1.1.4 \
    frechet_audio_distance>=0.2.0 \
    pandas>=2.0.0 \
    "numpy<2.0.0" \
    scipy>=1.10.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0

# Verify installations
echo "[5/6] Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

# Test imports
echo "[6/6] Testing module imports..."
cd "$(dirname "$0")"
python3 -c "from src.generation import AudioGenerator; print('Generation: OK')" || echo "Note: Will be tested at runtime"
python3 -c "from src.evaluation import AudioEvaluator; print('Evaluation: OK')" || echo "Note: Will be tested at runtime"
python3 -c "from src.rag_enhancer import PromptEnhancer; print('RAG Enhancer: OK')" || echo "Note: Will be tested at runtime"
python3 -c "from src.visualization import ResultsVisualizer; print('Visualization: OK')" || echo "Note: Will be tested at runtime"

# Download models (optional - will be downloaded at runtime anyway)
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Models will be downloaded automatically on first run."
echo "This may take 10-30 minutes depending on internet speed."
echo ""
echo "To run the experiment:"
echo "  python overnight_experiment.py --inference-steps 50"
echo ""
echo "For quick test first:"
echo "  python overnight_experiment.py --sample-limit 2"
echo ""
echo "Expected runtime: 2-4 hours on GPU with 30 prompts/domain"
echo ""