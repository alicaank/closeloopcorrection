#!/usr/bin/env bash
# usage: run_t5_captions.sh <batch_dir> <out.pt>  -- same env as run_arm_res.sh, own process (UMT5 only)
set -euo pipefail
python_bin=@CONDA_ROOT@/envs/lyra2/bin/python; cuda_home=@CONDA_ROOT@/envs/cuda128
site_dir="$(${python_bin} -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_HOME="${cuda_home}" PATH="$(dirname "${python_bin}"):${cuda_home}/bin:${PATH}" HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${site_dir}/torch/lib:${site_dir}/nvidia/cuda_runtime/lib:${site_dir}/nvidia/cudnn/lib:${cuda_home}/targets/x86_64-linux/lib:${site_dir}/nvidia/cuda_nvrtc/lib"
cd @WORKSPACE@/lyra/Lyra-2
exec "${python_bin}" "$(dirname "$0")/precompute_t5_captions.py" "$1" "$2"
