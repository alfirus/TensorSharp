using System;

namespace TensorSharp.MLX;

/// <summary>Stateless mapped dense weights and BF16 boundaries for LTX's Gemma encoder.</summary>
public static class MlxLtx25PromptOps
{
    public static bool TryMappedDense(Tensor output, Tensor input, IntPtr weights, int ggmlType, int inner, int columns,
        bool bf16Matmul = false)
    {
        if (weights == IntPtr.Zero || ggmlType is not (0 or 1 or 30) || inner < 1 || columns < 1 ||
            output?.Storage is not MlxStorage target || input?.Storage is not MlxStorage source ||
            !ReferenceEquals(target.AllocatorImpl, source.AllocatorImpl) || ReferenceEquals(target, source) ||
            output.ElementType != DType.Float32 || input.ElementType != DType.Float32 ||
            !output.IsContiguous() || !input.IsContiguous() || input.DimensionCount != 2 || output.DimensionCount != 2 ||
            input.Sizes[1] != inner || output.Sizes[1] != columns || output.Sizes[0] != input.Sizes[0] ||
            (bf16Matmul && ggmlType != 30)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, result = default;
            try
            {
                x = source.CreateArrayView(input);
                result = MlxNative.Ltx25MappedDense(x, weights, ggmlType, inner, columns, bf16Matmul);
                // Materialize before dropping the mapped descriptor. This bounds
                // residency to this projection and never expands the full encoder.
                MlxNative.Eval(result);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { target.ReplaceDeviceArray(result); result = default; }
                else target.UpdateDeviceSlice(output, result);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(result); }
        });
    }

    public static bool TryRoundBf16(Tensor tensor)
    {
        if (tensor?.Storage is not MlxStorage storage || tensor.ElementType != DType.Float32 || !tensor.IsContiguous()) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray input = default, output = default;
            try
            {
                input = storage.CreateArrayView(tensor); output = MlxNative.Ltx25RoundBf16(input);
                if (tensor.StorageOffset == 0 && tensor.ElementCount() == tensor.Storage.ElementCount)
                { storage.ReplaceDeviceArray(output); output = default; }
                else storage.UpdateDeviceSlice(tensor, output);
                return true;
            }
            finally { MlxNative.FreeArray(input); MlxNative.FreeArray(output); }
        });
    }
}

internal static partial class MlxNative
{
    internal static MlxArray Ltx25MappedDense(MlxArray input, IntPtr weights, int type, int inner, int columns,
        bool bf16Matmul)
    {
        // mlx-c's public dtype enumeration: float16=9, float32=10, bfloat16=12.
        MlxArray weight = mlx_array_new_data(weights, [columns, inner], 2, type == 30 ? 12 : type == 1 ? 9 : 10);
        MlxArray transposed = default, x = default, product = default;
        try
        {
            if (!weight.IsValid) throw new InvalidOperationException("Could not create mapped LTX dense weights.");
            transposed = Transpose(weight, [1, 0]);
            if (bf16Matmul)
            {
                Check(mlx_astype(out x, input, 12, DefaultStream()), "casting LTX BF16 matmul input");
                Check(mlx_matmul(out product, x, transposed, DefaultStream()), "multiplying LTX BF16 dense weights");
                return Astype(product, DType.Float32);
            }
            Check(mlx_matmul(out product, input, transposed, DefaultStream()), "multiplying LTX dense weights");
            MlxArray result = product; product = default; return result;
        }
        finally { FreeArray(transposed); FreeArray(weight); FreeArray(x); FreeArray(product); }
    }
    internal static MlxArray Ltx25RoundBf16(MlxArray input)
    {
        Check(mlx_astype(out MlxArray bf16, input, 12, DefaultStream()), "rounding LTX BF16 activations");
        try { return Astype(bf16, DType.Float32); }
        finally { FreeArray(bf16); }
    }
}
