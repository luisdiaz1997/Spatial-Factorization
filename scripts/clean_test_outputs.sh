#!/bin/bash
# Clean up test outputs before running a new test
# Keeps preprocessed data, removes model outputs

DATASET=${1:-slideseq_test}
OUTPUT_DIR="outputs/$DATASET"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR does not exist"
    exit 0
fi

echo "Cleaning test outputs in $OUTPUT_DIR..."

# Remove model directories
for model in pnmf svgp mggp_svgp lcgp mggp_lcgp; do
    if [ -d "$OUTPUT_DIR/$model" ]; then
        echo "  Removing $model/"
        rm -rf "$OUTPUT_DIR/$model"
    fi
done

# Remove logs
if [ -d "$OUTPUT_DIR/logs" ]; then
    echo "  Removing logs/"
    rm -rf "$OUTPUT_DIR/logs"
fi

# Remove run status
if [ -f "$OUTPUT_DIR/run_status.json" ]; then
    echo "  Removing run_status.json"
    rm -f "$OUTPUT_DIR/run_status.json"
fi

# Keep preprocessed data
if [ -d "$OUTPUT_DIR/preprocessed" ]; then
    echo "  Keeping preprocessed/"
fi

echo "Done!"
