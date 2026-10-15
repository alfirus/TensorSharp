using System;
using System.Runtime.InteropServices;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25Architecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = Ltx25Model.ArchitectureId,
        DisplayName = "LTX 2.5 Distilled",
        Aliases = new[] { Ltx25Model.ArchitectureId, "ltx", "ltxv", "ltx2", "ltx-2", "ltx_2_5" },
        Factory = context => new Ltx25Model(context.GgufPath, context.Backend),
        ApplyNativeTunables = ConfigurePrecision,
        DetectFromTensors = file => file.Tensors.ContainsKey("patchify_proj.weight") &&
            file.Tensors.ContainsKey("audio_patchify_proj.weight") &&
            file.Tensors.ContainsKey("transformer_blocks.0.prompt_scale_shift_table") &&
            file.Tensors.ContainsKey("transformer_blocks.0.attn1.to_gate_logits.weight"),
        MultiGpu = MultiGpuMode.SingleDevice,
        MultiGpuLimitation = "LTX 2.5 (ltx-2.5) currently executes on one device; tensor parallelism and layer splitting are not implemented.",
    };

    private static void ConfigurePrecision(ModelCreateContext context)
    {
        if (context.Backend is not (BackendType.Mlx or BackendType.GgmlMetal) ||
            !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MLX_ENABLE_TF32"))) return;
        // MLX caches this choice on its first GEMM. Configure the original F32
        // quality path before this family's allocator/graphs are initialized.
        // A process that previously ran MLX should set it at launch instead.
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst())
        {
            if (SetEnvironment("MLX_ENABLE_TF32", "0", 1) != 0)
                throw new InvalidOperationException("Unable to configure LTX Float32 MLX precision.");
            Environment.SetEnvironmentVariable("MLX_ENABLE_TF32", "0");
        }
    }

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "setenv", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SetEnvironment(string name, string value, int overwrite);
}
