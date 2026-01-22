#!/bin/bash
# Install PNMF and GPzoo dependencies locally for development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing PNMF and GPzoo dependencies..."
echo "Project root: $PROJECT_ROOT"

# Check if Probabilistic-NMF exists in parent directory
PNMF_PATH="$PROJECT_ROOT/../Probabilistic-NMF"
if [ -d "$PNMF_PATH" ]; then
    echo "Installing PNMF from: $PNMF_PATH"
    pip install -e "$PNMF_PATH"
else
    echo "Warning: Probabilistic-NMF not found at $PNMF_PATH"
    echo "You may need to clone it or adjust the path"
fi

# Check if GPzoo exists in parent directory
GPZOO_PATH="$PROJECT_ROOT/../GPzoo"
if [ -d "$GPZOO_PATH" ]; then
    echo "Installing GPzoo from: $GPZOO_PATH"
    pip install -e "$GPZOO_PATH"
else
    echo "Warning: GPzoo not found at $GPZOO_PATH"
    echo "You may need to clone it or adjust the path"
fi

echo ""
echo "Dependency installation complete!"
echo ""
echo "Next steps:"
echo "  1. Install spatial-factorization: pip install -e $PROJECT_ROOT"
echo "  2. Verify CLI works: spatial_factorization --help"
