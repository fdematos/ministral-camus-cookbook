#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${PROJECT_ROOT}/../llama.cpp"
MODEL_DIR="${PROJECT_ROOT}/ministral-8b-albert-camus/model-final"
OUT_DIR="${PROJECT_ROOT}/ministral-8b-albert-camus/gguf"

F16_OUT="${OUT_DIR}/ministral-camus-f16.gguf"
Q8_OUT="${OUT_DIR}/ministral-camus-q8_0.gguf"
Q6_OUT="${OUT_DIR}/ministral-camus-q6_k.gguf"

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANT_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

echo "=== GGUF build: F16 + Q8_0 + Q6_K ==="
echo "Project root : ${PROJECT_ROOT}"
echo "llama.cpp    : ${LLAMA_CPP_DIR}"
echo "Model dir    : ${MODEL_DIR}"
echo "Output dir   : ${OUT_DIR}"
echo

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERREUR: modèle final introuvable: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERREUR: convert_hf_to_gguf.py introuvable: ${CONVERT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "ERREUR: llama-quantize introuvable/exécutable: ${QUANT_BIN}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "1) Conversion F16..."
python3 "${CONVERT_SCRIPT}" "${MODEL_DIR}" --outfile "${F16_OUT}" --outtype f16
echo "  OK: ${F16_OUT}"
echo

echo "2) Quantification Q8_0..."
"${QUANT_BIN}" "${F16_OUT}" "${Q8_OUT}" Q8_0
echo "  OK: ${Q8_OUT}"
echo

echo "3) Quantification Q6_K..."
"${QUANT_BIN}" "${F16_OUT}" "${Q6_OUT}" Q6_K
echo "  OK: ${Q6_OUT}"
echo

echo "=== Terminé ==="
ls -lh "${OUT_DIR}"
