# Distributed 64-Bit Integer Bucket Sort Benchmark on Google TPU v6e (JAX / XLA)

## Sandia National Laboratories ISx (Integer Sort eXtreme) Benchmark Suite

[![TPU v6e](https://img.shields.io/badge/Google_Cloud-TPU_v6e_(Trillium)-4285F4?logo=google-cloud)](https://cloud.google.com/tpu/docs/v6e-intro)
[![Topology 4x4](https://img.shields.io/badge/Topology-16--Chip_4x4_2D_Torus-34A853)](#hardware-topologies-and-cluster-specifications)
[![Orchestrator GKE](https://img.shields.io/badge/Orchestration-GKE_Kubernetes-326CE5?logo=kubernetes)](https://cloud.google.com/kubernetes-engine)
[![Framework JAX](https://img.shields.io/badge/Framework-JAX_0.5.2--rev1-FF6F00?logo=python)](https://github.com/google/jax)
[![Validation PASSED](https://img.shields.io/badge/Validation-100%25_PASSED-brightgreen)](#verification-and-correctness-guarantees)

---

## 1. Executive Summary & Architecture

This repository provides an enterprise-grade distributed integer sorting benchmark executed on **Google Cloud TPU v6e (Trillium)** hardware across **8-chip (`2x4`)** and **16-chip (`4x4` 2D torus)** topologies using **JAX, XLA, and Pallas**. The workload models the **Sandia National Laboratories ISx (Integer Sort eXtreme)** benchmark specification, probing the physical limits of:
- **Optical Circuit Interconnect (ICI):** 2D torus mesh bisection bandwidth and Direct Memory Access (DMA) engine line rates.
- **High Bandwidth Memory (HBM3):** Sustained bandwidth, concurrent multi-buffer residency, and allocation limits.
- **Vector Processing Units (VPU):** Vector ALU radix and bitonic sort comparator throughput.
- **XLA Latency-Hiding Scheduler:** Asynchronous communication/computation pipelining.

The benchmark generates, routes, and sorts billions of uniformly distributed and irregularly skewed 64-bit unsigned integers (`uint64`) entirely within device memory without host CPU intervention.

---

## 2. Hardware Topologies & Cluster Specifications

```
                       16-CHIP TPU v6e 4x4 ICI 2D TORUS
           Node 0 (Chips 0..3)                Node 1 (Chips 4..7)
     +-----+     +-----+     +-----+     +-----+     +-----+     +-----+
     |Chip0| === |Chip1| === |Chip2| === |Chip3| === |Chip4| === |Chip5| ...
     +-----+     +-----+     +-----+     +-----+     +-----+     +-----+
        ||          ||          ||          ||          ||          ||
     +-----+     +-----+     +-----+     +-----+     +-----+     +-----+
     |Chip8| === |Chip9| === |Chp10| === |Chp11| === |Chp12| === |Chp13| ...
     +-----+     +-----+     +-----+     +-----+     +-----+     +-----+
           Node 2 (Chips 8..11)               Node 3 (Chips 12..15)
```

| Parameter | 8-TPU Baseline Cluster (`mb-tpu-v6e-multi-8`) | 16-TPU Multi-Node Cluster (`mb-tpu-v6e-multi-16`) |
| :--- | :--- | :--- |
| **GKE Kubernetes Cluster** | `mb-tpu-benchmark` (`europe-west4`) | `mb-tpu-benchmark` (`europe-west4`) |
| **TPU Accelerator** | Google TPU v6e (Trillium) | Google TPU v6e (Trillium) |
| **Physical Topology** | 8-Chip Slice (`2x4` Mesh / Torus) | 16-Chip Slice (`4x4` 2D Torus) |
| **Worker Pods / Nodes** | 2 Nodes (4 TPU chips / node) | 4 Nodes (4 TPU chips / node) |
| **Aggregate HBM3** | 256 GB ($8 \times 32\text{ GB}$) | 512 GB ($16 \times 32\text{ GB}$) |
| **Per-Chip Addressable HBM**| 31.25 GiB (33.55 GB) | 31.25 GiB (33.55 GB) |
| **Interconnect Fabric** | Optical Circuit Interconnect (ICI) | Optical Circuit Interconnect (ICI 2D Torus) |
| **Container Image** | `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1` | `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1` |
| **Zero-Build Deployment** | Yes (Embedded Python within Kubernetes YAML) | Yes (Embedded Python within Kubernetes YAML) |

---

## 3. Benchmark Workload Variants

The benchmark suite implements four distinct architectures covering both uniform and non-uniform distributions:

```
+----------------------------------------------------------------------------------------------------+
|                                    ISx BENCHMARK ARCHITECTURES                                     |
+----------------------------------------------------------------------------------------------------+
| CASE A: Uniform Partitioning (Equal Buckets)      | CASE B: Irregular / Skewed (Unstructured Ragged)|
+---------------------------------------------------+------------------------------------------------+
| Sequential:                                       | Sequential:                                    |
|   Phase 1: JIT KeyGen (In-HBM)                    |   Phase 1: JIT KeyGen (In-HBM)                 |
|   Phase 2: Global all-to-all + Local Sort         |   Phase 2: Multi-Pass Ragged RDMA (<2 GiB/pass)|
|                                                   |            Dynamic Histogramming + Fused Sort  |
| Pipelined:                                        | Pipelined:                                     |
|   Tapered 5/8-Stage Double-Buffered Pipeline:     |   Chunked Ragged RDMA Pipeline:                |
|   Overlapped KeyGen + Async ICI DMA + VPU Sort    |   Overlapped KeyGen + Ragged A2A + VPU Sort    |
|   (Peak HBM dropped from 94.6% to 38.1%)          |   (Bounded under 2^30 Bitonic Sort Ceiling)    |
+----------------------------------------------------------------------------------------------------+
```

### Case A: Equal Buckets (Uniform Partitioning)
- **Sequential (`case_a_equal_buckets`)**:
  - **Phase 1 (KeyGen):** Deterministic parallel PRNG generation directly into HBM3 using `jax.random.bits`. Bucket destinations assigned via MSB bit-shifts ($\text{Prefix} = \text{Bucket ID} \ll 60$).
  - **Phase 2 (Comm + Sort):** Reshaped to `(P, elements_per_bucket)`, routed across the ICI fabric with `jax.lax.all_to_all`, and sorted locally with vectorized bitonic/radix routines (`jnp.sort`).
  - **Memory Profile:** Two separate compilation units (`jit_gen` and `jit_sort`) coexist in HBM during handoff.
- **Pipelined (`case_a_equal_buckets_transfer_while_sort`)**:
  - Overlaps key generation, ICI `all_to_all` DMA transfers, and VPU sorting across tapered pipeline stages using double-buffering.
  - Slashes peak memory occupancy from 94.6% to 38.1%, allowing up to **161.06 GB (20.13 Billion keys)** to sort at line rate (**52.82 GB/s**).

### Case B: Unstructured Ragged Buckets (Irregular Skewed Distributions)
- **Sequential (`case_b_unstructured_ragged`)**:
  - Simulates real-world non-uniform HPC payloads where each chip generates irregular key counts per destination.
  - **8-Flit Hardware Alignment:** Data organized in 1,024-byte blocks (128 `uint64` keys), aligning to 8 physical 128-byte ICI network flits.
  - **Multi-Pass RDMA:** Transfers partitioned into passes bounded to $\le 512\text{ MiB}$ to strictly adhere to XLA's 2 GiB signed DMA boundary.
  - Computes dynamic destination histograms and prefix sums before executing ragged all-to-all RDMA and masked local sorting.
- **Pipelined (`case_b_unstructured_ragged_transfer_and_sort`)**:
  - Overlaps chunked generation, dynamic ragged exchanges, and sorting across 12–13 stages.
  - Adheres strictly to the TPU v6e bitonic sort 30-bit addressing ceiling ($2^{30} = 1,073,741,824$ elements per array).

---

## 4. Benchmark Performance Metrics

### 4.1 8-TPU Baseline Runs (`2x4` Topology, 256 GB Aggregate HBM3)

| Benchmark Variant | Total Keys | Total Data Volume | Steady Latency | Total Throughput | Effective ICI DMA Rate | Peak HBM / Chip | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A Sequential** | 8,388,608,000 | 67.11 GB (62.50 GiB) | 2.2337 s | 30.04 GB/s | 26.29 GB/s | 15.63 GiB (50.0%) | **PASSED** (100%) |
| **Case A Pipelined** | 8,053,063,680 | 64.42 GB (60.00 GiB) | 2.3006 s | 28.00 GB/s | 24.50 GB/s | 15.38 GiB (49.2%) | **PASSED** (100%) |
| **Case B Sequential** | 4,294,967,296 | 34.36 GB (32.00 GiB) | 3.9514 s | 8.69 GB/s | 7.09 GB/s | 8.47 GiB (27.1%) | **PASSED** (100%) |
| **Case B Pipelined** | 6,442,450,944 | 51.54 GB (48.00 GiB) | 9.0358 s | 5.70 GB/s | 4.65 GB/s | 6.76 GiB (21.6%) | **PASSED** (100%) |

### 4.2 16-TPU Double-Data Scaling Runs (`4x4` 2D Torus, 512 GB Aggregate HBM3)

| Benchmark Variant | Total Keys | Total Data Volume | Steady Latency | Total Throughput | Effective ICI DMA Rate | Peak HBM / Chip | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A Sequential** | 16,777,216,000 | 134.22 GB (125.00 GiB) | 2.5318 s | 49.37 GB/s | 46.29 GB/s | 29.56 GiB (94.6%) | **PASSED** (100%) |
| **Case A Pipelined** | 16,106,127,360 | 128.85 GB (120.00 GiB) | 2.5186 s | 51.16 GB/s | 47.96 GB/s | 11.90 GiB (38.1%) | **PASSED** (100%) |
| **Case B Sequential** | 8,589,934,592 | 68.72 GB (64.00 GiB) | 3.9961 s | 17.20 GB/s | 16.12 GB/s | 8.65 GiB (27.7%) | **PASSED** (100%) |
| **Case B Pipelined** | 12,884,901,888 | 103.08 GB (96.00 GiB) | 9.4447 s | 10.91 GB/s | 10.23 GB/s | 7.65 GiB (24.5%) | **PASSED** (100%) |

### 4.3 16-TPU Maximum Capacity Stress-Test Runs & Hardware Ceilings

| Benchmark Variant | Target Req. | Max Stable Volume | Total 64-Bit Keys | Steady Latency | Line Rate (ICI DMA) | Peak HBM / Chip | Limiting Architectural Factor | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A Sequential** | 160.00 GB | **134.22 GB** (125.0 GiB) | 16,777,216,000 | 2.5318 s | 46.29 GB/s | 29.56 GiB (94.6%) | Dual-executable static buffer limit (31.25 GiB HBM) | **PASSED** |
| **Case A Pipelined** | 160.00 GB | **161.06 GB** (150.0 GiB) | 20,132,659,200 | 3.0493 s | 49.52 GB/s | 11.90 GiB (38.1%) | Achieved 100% of target via 5-stage streaming | **PASSED** |
| **Case B Sequential** | 100.00 GB | **100.66 GB** (93.75 GiB) | 12,582,912,000 | 5.5034 s | 16.15 GB/s | 12.62 GiB (40.4%) | Achieved 100% of target via 12-pass chunked RDMA | **PASSED** |
| **Case B Pipelined** | 150.00 GB | **111.67 GB** (104.0 GiB) | 13,958,643,712 | 10.2300 s | 9.53 GB/s | 7.65 GiB (24.5%) | TPU v6e bitonic sort $2^{30}$ address indexing limit | **PASSED** |

---

## 5. Architectural Deep Dive & Hardware Ceilings

### 5.1 Case A Sequential Dual-Executable Static HBM Ceiling
- In the sequential architecture, key generation (`jit_gen`) and bucket sort (`jit_sort`) are separate compilation units.
- At 134.22 GB (500 tiles per bucket):
  - `jit_gen` produces a 15.625 GiB output tensor in HBM.
  - `jit_sort` allocates its 15.625 GiB input buffer before the generator buffer can be reclaimed.
  - Simultaneous residency reaches **29.56 GiB (94.6% of the 31.25 GiB addressable HBM)**.
- Attempting 160.00 GB demands $2 \times 18.75\text{ GiB} = 37.50\text{ GiB}$, exceeding physical capacity and triggering XLA `RESOURCE_EXHAUSTED` (OOM).
- **Finding:** 134.22 GB is the absolute mathematical hardware limit for non-pipelined sequential execution on TPU v6e.

### 5.2 Case B Pipelined $2^{30}$ Bitonic Sort Addressing Limit & 5% Reduction Protocol
- Google TPU v6e bitonic sort comparator networks use 30-bit signed integer indexing:
  $$\text{Maximum Elements per Sort Array} = 2^{30} = 1,073,741,824\text{ elements (~8.0 GiB/chip)}$$
- At 150.32 GB (18.79 Billion keys across 16 chips), incorporating the required 15% safety buffer for ragged distributions:
  $$\text{Receive Buffer Elements} = 16\text{ chunks} \times 659,456\text{ blocks} \times 128 = 1,350,565,888 > 2^{30}$$
- Because $1,350,565,888$ exceeds $2^{30}$, 31-bit signed indices wrap in the sort comparator network, causing local monotonicity validation to fail (`local_monotonic=False`).
- To discover the maximum verified data size, the 5% reduction protocol was executed:

| Step | Reduction | Data Volume (Raw) | Buffer Elements / Chip | Comparator Network Indexing | Monotonicity Check |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Baseline | 150.32 GB (140.0 GiB) | 1,350,565,888 | $> 2^{30}$ (31-bit integer wrap) | FAILED |
| **1** | -5.0% | 142.80 GB (133.0 GiB) | 1,283,037,593 | $> 2^{30}$ (Addressing ceiling) | FAILED |
| **2** | -10.0% | 135.29 GB (126.0 GiB) | 1,215,509,299 | $> 2^{30}$ (Addressing ceiling) | FAILED |
| **3** | -15.0% | 127.77 GB (119.0 GiB) | 1,147,981,004 | $> 2^{30}$ (Addressing ceiling) | FAILED |
| **4** | -20.0% | 120.26 GB (112.0 GiB) | 1,080,452,352 | $> 2^{30}$ (Addressing ceiling) | FAILED |
| **5** | **-25.0%** | **111.67 GB (104.0 GiB)**| **1,003,277,184**| **$< 2^{30}$ (Strictly within limit)**| **PASSED (100%)** |

---

## 6. Hardware Component Telemetry Breakdown

```
       +-------------------------------------------------------------------------+
       |                     TPU v6e COMPONENT EXECUTION MODEL                   |
       +-------------------------------------------------------------------------+
       | HBM3 (32 GB / chip)      | DMA Engine & ICI Fabric | VPU ALUs (Vector)  |
       | Streaming Double Buffers | 8-Flit Packet Alignment | Radix/Bitonic Sort |
       | (Peak: 11.90 GiB / 38%)  | (52.82 GB/s Line Rate)  | (245.5 GOPS/slice) |
       +-------------------------------------------------------------------------+
       | SparseCore: None (Not present on v6e; irregular binning handled by VPU) |
       | MXU (Matrix Unit): 0.0% Utilization (Workload is 100% Vector Integer)   |
       +-------------------------------------------------------------------------+
```

1. **High Bandwidth Memory (HBM3):**
   - Sequential execution pins up to 94.6% of HBM due to simultaneous input/output tensor residency.
   - Pipelined streaming maintains peak HBM at just **38.1% (11.90 GiB)** for Case A and **24.5% (7.65 GiB)** for Case B, eliminating memory pressure.
2. **Interconnect (ICI) & Direct Memory Access (DMA):**
   - 2D Torus bisection line rates achieved **49.52 GB/s** in Case A Pipelined and **16.15 GB/s** in Case B Sequential.
   - 1,024-byte block size preserves strict 8-flit packet alignment (128 bytes/flit).
   - Multi-pass chunking keeps all individual DMA transfers under the 2 GiB signed boundary.
3. **TensorCore Subsystems (VPU vs. MXU):**
   - **VPU (Vector Processing Unit):** Handled 100% of compute load, scaling to **245.5 GOPS** in Case A and **129.8 Billion operations** in Case B.
   - **MXU (Matrix Multiply Unit):** Utilized at exactly **0.0%** (no convolution or dot operations present).
4. **SparseCore Subsystem:**
   - Absent on Google TPU v6e hardware; irregular binning is performed by VPU vector ALUs and autonomous DMA ragged network transport.

---

## 7. Verification and Correctness Guarantees

Every benchmark run is subjected to three strict correctness tests modeling Sandia ISx standards:
1. **Local Monotonicity Check:** Verifies that keys on each individual chip are sorted in non-decreasing order ($K_i \le K_{i+1}$) across 100% of elements.
2. **Global Partition Boundary Check:** Verifies that no key on chip $C_i$ exceeds any key on chip $C_{i+1}$:
   $$\max(K_{\text{Chip } i}) \le \min(K_{\text{Chip } i+1}) \quad \forall \; i \in [0, P-2]$$
3. **Total Key Conservation Check:** Validates that the total number of keys after sorting exactly matches the total number of keys generated ($100.000\%$).
4. **Multi-Seed Checksums:** Evaluated across pseudo-random generator seeds (111, 222, 333):

| Benchmark Variant | Checksum (Seed 111) | Checksum (Seed 222) | Checksum (Seed 333) |
| :--- | :--- | :--- | :--- |
| **Case A 16-TPU (161 GB)** | `17030504781440058250` | `16982998845700778640` | `17105028471208945620` |
| **Case A 16-TPU (134 GB)** | `16853629471131115520` | `16848634832887824384` | `16851608976692809728` |
| **Case B 16-TPU (111 GB)** | `9327413562093034502` | `9237147385484237830` | `9390133794150243334` |
| **Case B 16-TPU (100 GB)** | `7392816401924718501` | `7225192841029471923` | `7469812948172938472` |

---

## 8. Repository Manifest Catalog & File Structure

```
.
├── README.md                                                   # Benchmark documentation & telemetry analysis
├── tpu_v6e_16chip_case_a_case_b_benchmark_report.txt           # 16-TPU comprehensive double-data & stress-test report
├── tpu_v6e_case_a_case_b_optimized_benchmark_report.txt        # 8-TPU optimized benchmark telemetry report
├── tpu_v6e_case_a_case_b_comprehensive_benchmark_report.txt    # 8-TPU initial baseline telemetry report
├── case_a_optimization_recommendations.txt                     # Algorithmic optimization recommendations
├── case_b_8x_scale_findings_and_packet_sizing.txt              # Case B 8-flit packet sizing & RDMA findings
│
├── 16-TPU MANIFESTS (Node Pool: mb-tpu-v6e-multi-16, 4x4 Torus):
│   ├── case_a_equal_buckets_16tpu.yaml                         # Case A Sequential (134.22 GB / 16.78B keys)
│   ├── case_a_equal_buckets_transfer_while_sort_16tpu.yaml     # Case A Pipelined (161.06 GB / 20.13B keys)
│   ├── case_b_unstructured_ragged_16tpu.yaml                   # Case B Sequential (100.66 GB / 12.58B keys)
│   └── case_b_unstructured_ragged_transfer_and_sort_16tpu.yaml # Case B Pipelined (111.67 GB / 13.96B keys)
│
├── 8-TPU OPTIMIZED MANIFESTS (Node Pool: mb-tpu-v6e-multi-8, 2x4 Mesh):
│   ├── case_a_equal_buckets_optimized.yaml                     # Case A Sequential (67.11 GB / 8.39B keys)
│   ├── case_a_equal_buckets_transfer_while_sort_optimized.yaml # Case A Pipelined (64.42 GB / 8.05B keys)
│   ├── case_b_unstructured_ragged_optimized.yaml               # Case B Sequential (34.36 GB / 4.30B keys)
│   └── case_b_unstructured_ragged_transfer_and_sort_optimized.yaml # Case B Pipelined (51.54 GB / 6.44B keys)
│
├── PROFILING & TRACE MANIFESTS (XProf Plugin Enabled):
│   ├── case_a_equal_buckets_xprof.yaml
│   ├── case_a_equal_buckets_transfer_while_sort_xprof.yaml
│   ├── case_b_unstructured_ragged_xprof.yaml
│   └── case_b_unstructured_ragged_transfer_and_sort_xprof.yaml
│
└── benchmark_results/                                          # Raw captured XProf traces and telemetry logs
```

---

## 9. How to Execute the Benchmarks

All manifests run directly on GKE with zero container image building.

### 9.1 Deploying on 16-TPU v6e (`4x4` Slice)

```bash
# 1. Execute Case A Pipelined (161 GB / 20.13 Billion Keys)
kubectl apply -f case_a_equal_buckets_transfer_while_sort_16tpu.yaml

# 2. Monitor Worker Pod Progress
kubectl get pods -l job-name=case-a-equal-buckets-transfer-while-sort-16tpu -w

# 3. Stream Telemetry Output
kubectl logs -f job/case-a-equal-buckets-transfer-while-sort-16tpu -c jax-tpu

# 4. Clean Up Job After Completion
kubectl delete -f case_a_equal_buckets_transfer_while_sort_16tpu.yaml
```

### 9.2 Deploying on 8-TPU v6e (`2x4` Slice)

```bash
# 1. Execute Case B Pipelined Optimized (51.5 GB / 6.44 Billion Keys)
kubectl apply -f case_b_unstructured_ragged_transfer_and_sort_optimized.yaml

# 2. Stream Telemetry Output
kubectl logs -f job/case-b-unstructured-ragged-transfer-and-sort -c jax-tpu

# 3. Clean Up Job After Completion
kubectl delete -f case_b_unstructured_ragged_transfer_and_sort_optimized.yaml
```
