#!/usr/bin/env bash
# Download the 12 10x Visium DLPFC h5ad files from the SDMBench server.
#
# Usage:
#   ./download_datasets.sh [DEST_DIR] [PARALLELISM]
#
# Defaults:
#   DEST_DIR    = /gladstone/engelhardt/lab/lchumpitaz/datasets/sdmbench/
#   PARALLELISM = 3

set -euo pipefail

DEST_DIR="${1:-/gladstone/engelhardt/lab/lchumpitaz/datasets/sdmbench}"
PARALLELISM="${2:-3}"

mkdir -p "${DEST_DIR}"

SAMPLES=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
BASE_URL="http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db"

download_one() {
    local sample="$1"
    local dest="${DEST_DIR}/${sample}.h5ad"
    local tmp="${dest}.part"
    local log="${dest}.log"

    if [[ -f "${dest}" ]]; then
        echo "[$(date '+%H:%M:%S')] ${sample}.h5ad already exists, skipping"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] Starting ${sample}"
    if curl -C - -f -L -o "${tmp}" "${BASE_URL}/${sample}.h5ad" >>"${log}" 2>&1; then
        mv "${tmp}" "${dest}"
        echo "[$(date '+%H:%M:%S')] Done ${sample}"
    else
        echo "[$(date '+%H:%M:%S')] FAILED ${sample} — see ${log}"
        return 1
    fi
}

active=0
for sample in "${SAMPLES[@]}"; do
    download_one "${sample}" &
    ((active += 1))
    if (( active >= PARALLELISM )); then
        wait -n
        ((active -= 1))
    fi
done
wait
echo "All downloads complete. Files in ${DEST_DIR}"
