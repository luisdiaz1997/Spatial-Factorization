#!/bin/bash
# Setup factorization conda environment
# Run: bash scripts/setup_env.sh

set -e

ENV_NAME="factorization"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Creating conda environment: $ENV_NAME"
echo "=========================================="

# Create environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate (need to source conda)
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "Installing core dependencies..."
pip install torch>=2.0.0 numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0
pip install scikit-learn>=1.0.0 tqdm>=4.62.0 pyyaml>=6.0 click>=8.0

echo ""
echo "Installing bioinformatics packages..."
pip install scanpy>=1.9.0 anndata>=0.8.0 squidpy>=1.2.0

echo ""
echo "Installing testing dependencies..."
pip install pytest>=7.0.0

echo ""
echo "Installing spatial-factorization..."
pip install -e .

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "Activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To install PNMF and GPzoo (sibling repos):"
echo "  pip install -e ../Probabilistic-NMF"
echo "  pip install -e ../GPzoo"
echo ""
