#!/usr/bin/env python3
"""
Download pre-computed SDMBench benchmark results for the 12 10x Visium DLPFC slides.

Fetches scores for all methods benchmarked on the SDMBench website
(http://sdmbench.drai.cn/) and saves them as a CSV.

Usage:
    python download_baselines.py [--out OUTPATH]

Default output: outputs/sdmbench/benchmarks/sdmbench_baselines.csv
"""

import argparse
import csv
import os
import sys

import requests

BASE = "http://sdmbench.drai.cn"

# The SDMBench website uses a TCM proxy. The actual endpoint path goes in
# the API-URL header; the service name goes in API-SERVICE.
MERGE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "API-URL": "/sdmbench/merge",
    "API-SERVICE": "api",
}

# Data IDs 1–12 correspond to the 12 DLPFC slides
SLIDE_MAP = {
    1:  "151507",
    2:  "151508",
    3:  "151509",
    4:  "151510",
    5:  "151669",
    6:  "151670",
    7:  "151671",
    8:  "151672",
    9:  "151673",
    10: "151674",
    11: "151675",
    12: "151676",
}

FIELDS = ["dataset", "model", "NMI", "HOM", "COM", "CHAOS", "PAS", "ASW"]


def fetch_baselines():
    rows = []
    for data_index, slide in SLIDE_MAP.items():
        # Step 1: get the pre-computed TSV path for this dataset
        r1 = requests.post(
            f"{BASE}/tcm/api/?",
            headers=MERGE_HEADERS,
            json={"task_id_list": [], "data_index": data_index},
            timeout=30,
        )
        r1.raise_for_status()
        info = r1.json()
        if not info.get("status"):
            print(f"WARNING: no data for dataset {data_index} ({slide}): {info.get('message','')}")
            continue
        tsv_path = info["data"]["out_tsv"]

        # Step 2: fetch the TSV content via the file-content API
        r2 = requests.post(
            f"{BASE}/file/content/?",
            headers={"Content-Type": "application/json"},
            json={"file_path": tsv_path},
            timeout=30,
        )
        r2.raise_for_status()
        result = r2.json()
        if result.get("message") != "success":
            print(f"WARNING: failed to fetch TSV for {slide}: {result.get('message','')}")
            continue

        methods = result["data"]
        for m in methods:
            rows.append({
                "dataset": slide,
                "model":   m["Method_name"],
                "NMI":     float(m["NMI"]),
                "HOM":     float(m["HOM"]),
                "COM":     float(m["COM"]),
                "CHAOS":   float(m["CHAOS"]),
                "PAS":     float(m["PAS"]),
                "ASW":     float(m["ASW"]),
            })
        print(f"  {slide}: {len(methods)} methods")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "sdmbench", "benchmarks", "sdmbench_baselines.csv"
        ),
        help="Output CSV path",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Downloading SDMBench baselines for 12 DLPFC slides...")
    rows = fetch_baselines()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_path}")
    methods = sorted(set(r["model"] for r in rows))
    print(f"Methods ({len(methods)}): {methods}")


if __name__ == "__main__":
    main()
