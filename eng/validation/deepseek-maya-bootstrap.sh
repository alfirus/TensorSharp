#!/usr/bin/env bash
# Provision build tools only; never install/replace the VM's GPU driver.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/workspace/dotnet:/usr/local/cuda-12.8/bin:$PATH"
apt-get update
apt-get install -y --no-install-recommends ninja-build ccache libicu-dev libssl-dev python3-venv
if ! command -v nvcc >/dev/null; then
  echo 'CUDA toolkit absent: install a matching toolkit before building.' >&2
  exit 1
fi
mkdir -p /workspace/dotnet /workspace/validation
curl -fL --retry 3 https://dot.net/v1/dotnet-install.sh -o /workspace/validation/dotnet-install.sh
bash /workspace/validation/dotnet-install.sh --channel 10.0 --install-dir /workspace/dotnet
python3 -m venv /workspace/hf-env
/workspace/hf-env/bin/python -m pip install 'huggingface_hub>=1.0,<3'
export PATH="/workspace/dotnet:/usr/local/cuda-12.8/bin:$PATH"
dotnet --info
nvcc --version
