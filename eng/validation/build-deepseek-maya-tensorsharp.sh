#!/usr/bin/env bash
# Run from an isolated Linux TensorSharp source tree. Upstream stays unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="/workspace/dotnet:/usr/local/cuda-12.8/bin:$PATH"
export TENSORSHARP_GGML_GIT_REF="${TENSORSHARP_GGML_GIT_REF:-3612c0af93f2cfc8f048295300a61187723b12ae}"
bash eng/fetch-ggml.sh
test -z "$(git -C ExternalProjects/ggml status --porcelain --untracked-files=all)"
export TENSORSHARP_GGML_NO_UPDATE=1
cmake -S TensorSharp.GGML.Native -B TensorSharp.GGML.Native/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON -DTENSORSHARP_GGML_NATIVE_ENABLE_NCCL=OFF
cmake --build TensorSharp.GGML.Native/build -j "${BUILD_JOBS:-8}"
test -z "$(git -C ExternalProjects/ggml status --porcelain --untracked-files=all)"
git -C ExternalProjects/ggml rev-parse HEAD
sha256sum TensorSharp.GGML.Native/build/libGgmlOps.so
# The serving path uses GgmlOps; portable committed PTX is sufficient for the
# unused direct-CUDA backend, avoiding a second unrelated kernel compilation.
export PATH="/workspace/dotnet:/usr/bin:/bin"
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:NuGetAudit=false
dotnet build eng/validation/DeepSeekPipelineWorker/DeepSeekPipelineWorker.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:NuGetAudit=false
dotnet build eng/validation/DeepSeekEngineBench/DeepSeekEngineBench.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:NuGetAudit=false
cp TensorSharp.GGML.Native/build/libGgmlOps.so eng/validation/DeepSeekPipelineWorker/bin/Release/net10.0/
cp TensorSharp.GGML.Native/build/libGgmlOps.so eng/validation/DeepSeekEngineBench/bin/Release/net10.0/
