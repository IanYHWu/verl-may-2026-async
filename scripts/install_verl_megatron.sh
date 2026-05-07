#!/usr/bin/env bash
# End-to-end install of the verified Qwen3.5 + Megatron stack for B200 / H100.
#
# Prerequisites:
#   - An *active* conda env with Python 3.10 (e.g. `conda create -n
#     verl_megatron python=3.10 && conda activate verl_megatron`).
#   - CUDA toolkit reachable; CUDA_HOME exported (e.g. /usr/local/cuda).
#   - Run from the repo root.
#
# Idempotency: best-effort. The script aborts on the first failing step so
# you can re-run from where it stopped without redoing earlier work.

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" == "" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "ERROR: activate a non-base conda env first (Python 3.10)." >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "ERROR: CUDA_HOME not set. Export it to your CUDA toolkit (e.g. /usr/local/cuda)." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Python: $(python --version)"
echo "==> Conda env: ${CONDA_DEFAULT_ENV}"
echo "==> CUDA_HOME: ${CUDA_HOME}"
echo "==> Repo root: ${REPO_ROOT}"
echo

# 1. cudnn first — TransformerEngine dlopens libcudnn_graph.so.9 at import.
echo "==> [1/10] nvidia-cudnn-cu12"
pip install --quiet nvidia-cudnn-cu12

# 2. Upstream stack installer. This pins older versions of vLLM / TE /
#    Megatron-LM that don't ABI-match torch 2.10 + sm100; we'll upgrade
#    them in steps 4–7 below.
echo "==> [2/10] upstream installer (vLLM + flash-attn + Megatron + cudnn)"
USE_MEGATRON=1 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh

# 3. Discover cudnn / nccl include + lib paths for the source builds.
CUDNN_LOC=$(pip show nvidia-cudnn-cu12 | grep Location | cut -d' ' -f2)
NCCL_LOC=$(pip show nvidia-nccl-cu12  | grep Location | cut -d' ' -f2)
export CPATH="${CUDNN_LOC}/nvidia/cudnn/include:${NCCL_LOC}/nvidia/nccl/include:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_LOC}/nvidia/cudnn/lib:${NCCL_LOC}/nvidia/nccl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_LOC}/nvidia/cudnn/lib:${NCCL_LOC}/nvidia/nccl/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# 4. TransformerEngine 2.13 from source against the active torch.
echo "==> [4/10] TransformerEngine 2.13 (source)"
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-deps --upgrade \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.13

# 5. Megatron-Core 0.16.1 (newer than what install_vllm_sglang_mcore.sh
#    pinned).
echo "==> [5/10] Megatron-LM core_v0.16.1"
pip install --no-deps --upgrade \
    git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.1

# 6. vLLM 0.19.1.
echo "==> [6/10] vLLM 0.19.1"
pip install --upgrade vllm==0.19.1

# 7. flash-attn 2.8.3 (source build for sm100; H100 can use the prebuilt
#    wheel but the source build still works).
echo "==> [7/10] flash-attn 2.8.3"
pip install --upgrade --no-build-isolation flash-attn==2.8.3

# 8. apex (cpp + cuda extensions). Slow.
echo "==> [8/10] apex (source build, slow)"
if [[ ! -d apex ]]; then
    git clone https://github.com/NVIDIA/apex.git
fi
(
    cd apex
    MAX_JOB=16 pip install -v --disable-pip-version-check --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./
)

# 9. mbridge pinned to the verified commit.
echo "==> [9/10] mbridge (pinned commit)"
pip install git+https://github.com/ISEEKYAN/mbridge.git@4cfd6f5eab84ed5424a8202e1a282e6ac584fce5

# 10. flash-linear-attention — Megatron's GatedDeltaNet uses fla.modules.l2norm
#     and fla.ops.gated_delta_rule.
echo "==> [10/10] flash-linear-attention 0.4.2 + verl"
pip install flash-linear-attention==0.4.2

# verl itself, in editable mode.
pip install --no-deps -e .

echo
echo "==> Sanity check"
python - <<'PY'
import torch, transformer_engine, transformer_engine_torch, flash_attn
import megatron.core, vllm, mbridge
import verl
print("torch          ", torch.__version__)
print("transformer_eng", transformer_engine.__version__)
print("TE_torch       ", transformer_engine_torch.__version__)
print("flash_attn     ", flash_attn.__version__)
print("megatron-core  ", megatron.core.__version__)
print("vllm           ", vllm.__version__)
print("mbridge        ", mbridge.__version__)
PY

echo
echo "==> Done. Remember to set LD_LIBRARY_PATH at runtime:"
echo
echo '    CUDNN_LOC=$(pip show nvidia-cudnn-cu12 | grep Location | cut -d" " -f2)'
echo '    NCCL_LOC=$(pip show nvidia-nccl-cu12  | grep Location | cut -d" " -f2)'
echo '    export LD_LIBRARY_PATH=$CUDNN_LOC/nvidia/cudnn/lib:$NCCL_LOC/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH'
echo
echo "    The launchers in scripts/sample_scripts/ set this themselves."
