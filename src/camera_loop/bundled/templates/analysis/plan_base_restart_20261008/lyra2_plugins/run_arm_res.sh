#!/usr/bin/env bash
# usage: run_arm.sh <arm_json> <out_dir> [batch_dir] [num_samples] [num_frames]
set -euo pipefail
arm_json="$1"; out_dir="$2"; batch="${3:-$(dirname "$0")/batch_dev10}"; n="${4:-10}"; frames="${5:-241}"
repo_root=@WORKSPACE@/lyra/Lyra-2
python_bin=@CONDA_ROOT@/envs/lyra2/bin/python
cuda_home=@CONDA_ROOT@/envs/cuda128
da3_model=${HOME}/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
site_dir="$(${python_bin} -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_HOME="${cuda_home}"
export PATH="$(dirname "${python_bin}"):${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${site_dir}/nvidia/cudnn/include:${site_dir}/nvidia/nccl/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${site_dir}/torch/lib:${site_dir}/nvidia/cuda_runtime/lib:${site_dir}/nvidia/cudnn/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${site_dir}/nvidia/cuda_nvrtc/lib"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export NVTE_FLASH_ATTN=0
export GV_LYRA_ARM="${arm_json}"
mkdir -p "${out_dir}"
cd "${repo_root}"
PYTHONPATH="${repo_root}:$(dirname "$0")" exec "${python_bin}" "$(dirname "$0")/gv_lyra_plugins.py" \
  --input_image_path "${batch}/images" --trajectory_path "${batch}/trajectories" --captions_path "${batch}/captions" \
  --experiment lyra2 --checkpoint_dir "${repo_root}/checkpoints/model" --output_path "${out_dir}" \
  --num_samples "${n}" --num_frames "${frames}" --resolution "${LYRA_RES:-256,448}" --pose_scale 1.0 \
  --num_sampling_step 35 --guidance 5.0 --shift 5.0 --seed 7 --da3_model_name "${da3_model}" \
  --offload --sequential_cpu_offload --offload_when_prompt --offload_da3_diffusion --save_png_frames ${LYRA_EXTRA:-}
