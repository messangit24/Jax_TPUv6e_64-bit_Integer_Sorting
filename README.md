# Distributed 64-Bit Integer Bucket Sort Benchmark on Google TPU v6e (JAX)

This repository contains a high-performance distributed in-memory sorting benchmark executed on a **Google Cloud TPU v6e 16-chip slice (`4x4` topology)** using **JAX and XLA**. The benchmark evaluates the fabric bisection bandwidth of the Optical Circuit Interconnect (ICI) and High Bandwidth Memory (HBM3) by generating and sorting billions of 64-bit unsigned integers without host CPU intervention.

Algorithmically, this workload models the **Sandia National Laboratories ISx (Integer Sort eXtreme)** benchmark specification.

---

The benchmark runs across three distinct phases:

1. **Phase 1: Deterministic Parallel Key Generation (In-HBM3)**
   * Each TPU chip generates 64-bit unsigned integers (`uint64`) directly in HBM3 using `jax.random.bits`.
   * MSB bit-shifting applies bucket destination prefixes ($\text{Prefix} = \text{Bucket ID} \ll 60$) to enforce global partitionability across all 16 chips.
2. **Phase 2: Global All-to-All ICI Exchange & Bucket Sorting**
   * Keys are reshaped into target buckets and transferred simultaneously across the 2D-torus Optical Circuit Interconnect using `jax.lax.all_to_all`.
   * Received keys are sorted locally on each chip using XLA-compiled vectorized radix/bitonic sort routines (`jnp.sort`).
3. **Phase 3: Global Boundary Validation**
   * Verifies local array ordering ($\text{Keys}_{i} \le \text{Keys}_{i+1}$) and cross-chip partition boundaries ($\max(\text{Keys}_{\text{Chip } i}) \le \min(\text{Keys}_{\text{Chip } i+1})$) using `jax.lax.all_gather`.

---

## Infrastructure & Zero-Build Deployment Requirements

> **No Custom Docker Build Required:**  
> The benchmark executes directly using Google's official public base image:  
> `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1`  
> 
> You **do not need to build, tag, or push a custom container**. The entire benchmark Python logic is embedded directly within the Kubernetes Job YAML manifest.

| Parameter | Configuration |
| :--- | :--- |
| **Orchestrator** | Google Kubernetes Engine (GKE) |
| **Container Image** | `us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1` |
| **Accelerator Type** | Google TPU v6e (Trillium) |
| **Topology** | 16-Chip Slice (`4x4` 2D Torus Mesh) |
| **Total Physical HBM** | 512 GB Aggregate ($16 \times 32\text{ GB HBM3}$) |
| **Interconnect** | Optical Circuit Interconnect (ICI) |
| **Software Stack** | Python 3, JAX (`jaxlib` 0.5.2), XLA Compiler, GKE Kueue/Job Operator |

## Latest Benchmark Results (100 GB Scale Validation Run)

=======================================================
PHASE-BY-PHASE BENCHMARK METRICS REPORT
=======================================================
Total Active TPU Chips          : 16
Keys Generated per TPU Chip     : 781,250,000
Total 64-Bit Keys Generated     : 12,500,000,000
Total In-Memory Dataset Size    : 100.00 GB
Global Validation Status        : [PASSED]
-------------------------------------------------------
PHASE 1: 64-Bit Integer Generation (In-HBM)
  Execution Time               : 2.8303 seconds
  HBM Generation Throughput    : 35.33 GB/s
  ICI Network Traffic          : 0.00 GB
-------------------------------------------------------
PHASE 2: ICI Exchange & Bucket Sorting
  Execution Time               : 21.3385 seconds
  ICI Network Traffic Routed   : 93.75 GB
  Effective ICI Network Speed  : 4.39 GB/s
  Effective Memory Bandwidth   : 28.12 GB/s
=======================================================

