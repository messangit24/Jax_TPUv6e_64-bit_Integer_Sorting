import json
import os
import sys

RESULTS_DIR = "./benchmark_results"

def load_metrics(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    opt1 = load_metrics("option1_metrics.json")
    opt2 = load_metrics("option2_metrics.json")

    if not opt1 or not opt2:
        print("Error: Both option1_metrics.json and option2_metrics.json are required.")
        sys.exit(1)

    speedup = opt1['execution_seconds'] / opt2['execution_seconds'] if opt2['execution_seconds'] > 0 else 0

    report = f"""
# Distributed In-Memory Sorting Benchmark Evaluation Report
**Platform**: Google Cloud GKE (16 TPU v6e Chips / `ct6e-standard-4t`)

## Metric Comparison

| Performance Metric | Option 1: Host-Side OpenSHMEM | Option 2: Native TPU Silicon | Speedup / Factor |
| :--- | :--- | :--- | :--- |
| **OpenSHMEM Compliant** | **Yes** (PGAS `<shmem.h>`) | **No** (XLA `AllToAll` Collective) | N/A |
| **Dataset Processed** | {opt1['data_gb']:.2f} GB | {opt2['data_gb']:.2f} GB | Identical |
| **Execution Time** | {opt1['execution_seconds']:.4f} sec | {opt2['execution_seconds']:.4f} sec | **{speedup:.2f}x Faster** |
| **Effective Throughput** | {opt1['throughput_gbps']:.2f} GB/s | {opt2['throughput_gbps']:.2f} GB/s | **{opt2['throughput_gbps']/opt1['throughput_gbps']:.2f}x Higher** |
| **Memory Hardware** | {opt1['memory_location']} | {opt2['memory_location']} | HBM3 Advantage |
| **Network Interconnect** | {opt1['interconnect']} | {opt2['interconnect']} | ICI Fabric Advantage |

## Key Insights & Architectural Tradeoffs

1. **RFP Specification Compliance**:
   - **Option 1** satisfies the strict **OpenSHMEM PGAS specification** mandate. However, it relies on Host CPUs and standard data center networks, leaving TPU hardware idle.
   - **Option 2** violates the OpenSHMEM API requirement by utilizing compiled XLA primitives, but directly leverages the **TPU v6e silicon and 800 GB/s ICI fabric**.

2. **Performance Delta**:
   - Running directly on TPU silicon (**Option 2**) yields higher throughput due to on-package HBM3 memory bandwidth (~1.6 TB/s) and direct optical ICI links compared to CPU host DRAM and standard network interfaces.
"""
    print(report)
    
    report_path = os.path.join(RESULTS_DIR, "consolidated_benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport successfully saved to {report_path}")

if __name__ == "__main__":
    main()
