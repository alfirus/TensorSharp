using System;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25PromptMath
{
    internal static float Bf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7f800000) != 0x7f800000) bits += 0x7fff + ((bits >> 16) & 1);
        return BitConverter.UInt32BitsToSingle(bits & 0xffff0000);
    }

    internal static void RoundBf16(Tensor tensor)
    {
        if (MlxLtx25PromptOps.TryRoundBf16(tensor)) return;
        float[] data = tensor.GetElementsAsFloat(checked((int)tensor.ElementCount()));
        for (int i = 0; i < data.Length; ++i) data[i] = Bf16(data[i]);
        tensor.SetElementsAsFloat(data);
    }

    internal static Tensor Norm(DirectContext context, Tensor input, Tensor gain, float epsilon, bool bf16)
    {
        var output = DirectOps.RmsNorm(context, input, gain ?? context.Ones(input.Sizes[1]), epsilon);
        if (bf16) RoundBf16(output);
        return output;
    }

    internal static void GemmaRope(Tensor input, int heads, int dim, int firstPosition, float theta, float proportion)
    {
        var data = input.GetElementsAsFloat(checked((int)input.ElementCount()));
        int half = dim / 2, rotated = (int)(proportion * half);
        var inverse = new float[rotated];
        for (int i = 0; i < rotated; ++i) inverse[i] = 1f / MathF.Pow(theta, 2f * i / dim);
        for (int token = 0; token < input.Sizes[0]; ++token)
        for (int head = 0; head < heads; ++head)
        for (int i = 0; i < rotated; ++i)
        {
            float angle = (firstPosition + token) * inverse[i];
            float cosine = Bf16(MathF.Cos(angle)), sine = Bf16(MathF.Sin(angle));
            int a = (token * heads + head) * dim + i, b = a + half;
            float x = data[a], y = data[b];
            data[a] = Bf16(Bf16(x * cosine) - Bf16(y * sine));
            data[b] = Bf16(Bf16(y * cosine) + Bf16(x * sine));
        }
        input.SetElementsAsFloat(data);
    }

    internal static Tensor RepeatKv(DirectContext context, Tensor source, int queryHeads, int kvHeads, int head)
    {
        using var shaped = source.View(source.Sizes[0], kvHeads, 1, head);
        using var expanded = shaped.Expand(source.Sizes[0], kvHeads, queryHeads / kvHeads, head);
        using var contiguous = Ops.NewContiguous(expanded);
        return contiguous.View(source.Sizes[0], (long)queryHeads * head);
    }
}
