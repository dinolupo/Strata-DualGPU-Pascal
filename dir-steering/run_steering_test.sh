#!/usr/bin/env bash
# Quick validation script for Directional & Transient Steering in Strata
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Select Python binary (prefer repo venv if available)
PYTHON_BIN="python3"
if [ -x "${REPO_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif [ -x "/home/dino/proj/Strata-DualGPU-Pascal/.venv/bin/python" ]; then
    PYTHON_BIN="/home/dino/proj/Strata-DualGPU-Pascal/.venv/bin/python"
fi

echo "=== Strata Directional Steering Test Suite ==="
echo "Repo:   ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"

# 1. Run unit mathematical tests
echo ""
echo "[1/3] Running Gram-Schmidt orthogonalization unit tests..."
"${PYTHON_BIN}" "${REPO_DIR}/tools/test_build_steering_vector.py"

# 2. Check format round-trip parity
echo ""
echo "[2/3] Checking bit-for-bit parity (.f32 <-> .gguf)..."
if [ -f "${SCRIPT_DIR}/vectors/Qwen3.8-Flash-Next-experimental-speed-projection.gguf" ]; then
    "${PYTHON_BIN}" "${REPO_DIR}/tools/ds4_to_cvec.py" \
        --check-roundtrip "${SCRIPT_DIR}/vectors/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
elif [ -f "${SCRIPT_DIR}/vectors/Qwen3.8-Flash-Next-steering.f32" ]; then
    "${PYTHON_BIN}" "${REPO_DIR}/tools/ds4_to_cvec.py" \
        --from-ds4 "${SCRIPT_DIR}/vectors/Qwen3.8-Flash-Next-steering.f32" \
        --to-gguf /tmp/strata_test_roundtrip.gguf
    "${PYTHON_BIN}" "${REPO_DIR}/tools/ds4_to_cvec.py" \
        --check-roundtrip /tmp/strata_test_roundtrip.gguf
    rm -f /tmp/strata_test_roundtrip.gguf
fi

# 3. Server test summary
echo ""
echo "[3/3] Ready for server tests."
echo "To test live against a running Strata server, execute:"
echo "  ${PYTHON_BIN} ${REPO_DIR}/tools/test_steering_server.py"
echo ""
echo "=== All checks completed successfully ==="
