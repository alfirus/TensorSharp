"""Hash runtime artifacts used by embedding validation outside measured work."""
import hashlib
from pathlib import Path


def command_artifacts(original, command):
    result = original(command)
    known = {str(Path(item["path"]).resolve()) for item in result}
    extra = set()
    for argument in command:
        if argument.startswith("TSG_EMBEDDING_BLAS_LIBRARY="):
            extra.add(Path(argument.split("=", 1)[1]))
        if argument.startswith("TENSORSHARP_CPU_NATIVE_LIBRARY="):
            extra.add(Path(argument.split("=", 1)[1]))
        if argument.startswith("TENSORSHARP_MLX_LIBRARY_DIR="):
            directory = Path(argument.split("=", 1)[1])
            if directory.is_dir():
                for pattern in ("libmlxc*", "libmlx*", "libjaccl*", "*.metallib"):
                    extra.update(directory.glob(pattern))
        path = Path(argument)
        if path.is_file() and path.suffix == ".dll":
            for pattern in ("libmlxc*", "libmlx*", "libjaccl*", "*.metallib", "cuda_kernels/*.ptx", "libTensorSharpCpu.*", "TensorSharpCpu.dll"):
                extra.update(path.parent.glob(pattern))
    for path in sorted(extra):
        if not path.is_file() or str(path.resolve()) in known:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return result
