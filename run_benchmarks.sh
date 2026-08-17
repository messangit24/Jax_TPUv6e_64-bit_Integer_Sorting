#!/usr/bin/env bash
set -e

# Central directory for storing output metrics
RESULTS_DIR="./benchmark_results"
mkdir -p "${RESULTS_DIR}"

echo "================================================="
echo " Starting Sequential Execution on 16 TPU Chips"
echo " Central Results Directory: ${RESULTS_DIR}"
echo "================================================="

# ---------------------------------------------------
# Phase 1: Run Option 1 (Host-Side OpenSHMEM)
# ---------------------------------------------------
echo ""
echo "[Phase 1/2] Deploying Option 1: OpenSHMEM Job..."
kubectl apply -f openshmem-job.yaml

echo "[Phase 1/2] Waiting for OpenSHMEM pods to complete..."
# Wait for completion index 0 pod to enter Completed/Terminated status
while true; do
    POD0_NAME=$(kubectl get pods -l job-name=openshmem-benchmark,batch.kubernetes.io/job-completion-index=0 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "${POD0_NAME}" ]; then
        STATUS=$(kubectl get pod "${POD0_NAME}" -o jsonpath='{.status.containerStatuses[0].state.terminated.reason}' 2>/dev/null || true)
        if [ "${STATUS}" = "Completed" ]; then
            echo "[Phase 1/2] OpenSHMEM Pod 0 finished successfully."
            break
        elif [ "${STATUS}" = "Error" ]; then
            echo "[Phase 1/2] ERROR: OpenSHMEM execution failed."
            kubectl logs "${POD0_NAME}"
            exit 1
        fi
    fi
    sleep 5
done

echo "[Phase 1/2] Copying option1_metrics.json to ${RESULTS_DIR}/..."
kubectl cp "${POD0_NAME}:/tmp/option1_metrics.json" "${RESULTS_DIR}/option1_metrics.json"

echo "[Phase 1/2] Cleaning up OpenSHMEM job..."
kubectl delete job openshmem-benchmark
kubectl delete svc openshmem-svc --ignore-not-found

# Verify all TPU resources are released
echo "[Phase 1/2] Waiting for TPU node pool to clear..."
while [ $(kubectl get pods -l job-name=openshmem-benchmark 2>/dev/null | wc -l) -gt 0 ]; do
    sleep 3
done

# ---------------------------------------------------
# Phase 2: Run Option 2 (Native TPU Silicon JAX)
# ---------------------------------------------------
echo ""
echo "[Phase 2/2] Deploying Option 2: Native TPU JAX Job..."
kubectl apply -f jax-tpu-job.yaml

echo "[Phase 2/2] Waiting for JAX TPU pods to complete..."
while true; do
    POD0_NAME=$(kubectl get pods -l job-name=jax-tpu-benchmark,batch.kubernetes.io/job-completion-index=0 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "${POD0_NAME}" ]; then
        STATUS=$(kubectl get pod "${POD0_NAME}" -o jsonpath='{.status.containerStatuses[0].state.terminated.reason}' 2>/dev/null || true)
        if [ "${STATUS}" = "Completed" ]; then
            echo "[Phase 2/2] JAX TPU Pod 0 finished successfully."
            break
        elif [ "${STATUS}" = "Error" ]; then
            echo "[Phase 2/2] ERROR: JAX TPU execution failed."
            kubectl logs "${POD0_NAME}"
            exit 1
        fi
    fi
    sleep 5
done

echo "[Phase 2/2] Copying option2_metrics.json to ${RESULTS_DIR}/..."
kubectl cp "${POD0_NAME}:/tmp/option2_metrics.json" "${RESULTS_DIR}/option2_metrics.json"

echo "[Phase 2/2] Cleaning up JAX TPU job..."
kubectl delete job jax-tpu-benchmark
kubectl delete svc jax-tpu-svc --ignore-not-found

# ---------------------------------------------------
# Phase 3: Generate Consolidated Report
# ---------------------------------------------------
echo ""
echo "================================================="
echo " Generating Consolidated Performance Report"
echo "================================================="
python3 generate_report.py
