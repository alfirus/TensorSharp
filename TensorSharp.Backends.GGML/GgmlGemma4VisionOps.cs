// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

/// <summary>Borrowed immutable F32 weights of one qualified Gemma4 vision block.
/// Matrices are Q,K,V,out,gate,up,down; norms are ln1,Qnorm,Knorm,attention post,ln2,FFN post.
/// Owners must not mutate or dispose these tensors during a synchronous stack call.
/// Matrix host addresses remain immutable device-cache keys; invalidate each
/// binding with InvalidateTensorHostBuffer before mutation or disposal.</summary>
public sealed record GgmlGemma4VisionBlock(Tensor[] Matrices, Tensor[] Norms, float[] Clamps);

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct GgmlGemma4VisionStackBlock
{
    public GgmlTensorView2D Q, K, V, Out, Gate, Up, Down;
    public GgmlTensorView2D Ln1, QNorm, KNorm, AttnPostNorm, Ln2, FfnPostNorm;
    public fixed float Clamps[28];
}

[StructLayout(LayoutKind.Sequential)]
internal struct GgmlGemma4VisionStackArgs
{
    public int StructBytes, Version, BlockCount, BlockStructBytes;
    public GgmlTensorView2D Input, Result;
    public IntPtr Blocks, PosX, PosY;
    public int PositionCount;
    public float Epsilon, RopeTheta;
    public int Heads, HeadDim;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4VisionStackF32PreciseCuda(in GgmlGemma4VisionStackArgs args);

    internal static bool Gemma4VisionStack(in GgmlGemma4VisionStackArgs args)
    {
        // A pinned TP worker must retain its original per-block path even when
        // its rank happens to be zero. Do not reset its thread-local rank.
        if (s_rankPinned) return false;
        int status;
        try { status = TSGgml_Gemma4VisionStackF32PreciseCuda(in args); }
        catch (EntryPointNotFoundException) { return false; } // Optional optimization; retained precise block API remains authoritative.
        if (status == 0) return false;
        if (status != 1)
            throw new InvalidOperationException("Native GGML Gemma4 precise CUDA stack failed. " + LastNativeError());
        return true;
    }
}

public partial class GgmlBasicOps
{
    /// <summary>Try one transient, precise CUDA graph for the validated sixteen
    /// embedding blocks. False means pre-compute admission refusal and leaves
    /// both caller activations untouched. Compute/precision failures throw.
    /// No graph or activation descriptor is retained. Immutable matrix host
    /// addresses remain cache keys and require InvalidateTensorHostBuffer before
    /// mutation/disposal. Exclusive tensor-owner use and
    /// the existing quiescent global backend lifecycle contract are required.</summary>
    public static unsafe bool TryGemma4VisionStack(Tensor result, Tensor input,
        GgmlGemma4VisionBlock[] blocks, int[] posX, int[] posY, float epsilon, float ropeTheta)
    {
        if (input?.Storage is not GgmlStorage source || result?.Storage is not GgmlStorage destination ||
            source.Context.BackendType != GgmlBackendType.Cuda || destination.Context != source.Context ||
            source.Context.Degree != 1 || destination.DeviceId != source.DeviceId ||
            blocks?.Length != 16 || posX == null || posY == null ||
            input.DimensionCount != 2 || input.Sizes[1] != 768 || input.Sizes[0] is <= 0 or > 16384 ||
            posX.Length != input.Sizes[0] || posY.Length != posX.Length ||
            !float.IsFinite(epsilon) || epsilon <= 0 || !float.IsFinite(ropeTheta) || ropeTheta <= 0 ||
            !TryCreateStandardView(input, out GgmlTensorView2D inputView) || !input.IsContiguous() ||
            result.DimensionCount != 2 || result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != 768 ||
            !TryCreateStandardView(result, out GgmlTensorView2D resultView) || !result.IsContiguous()) return false;
        var views = new GgmlGemma4VisionStackBlock[16];
        var temporaryViews = new List<Tensor>();
        try
        {
            bool View(Tensor tensor, int rows, int columns, out GgmlTensorView2D view)
            {
                view = default;
                if (tensor?.Storage is not GgmlStorage storage || storage.Context != source.Context ||
                    storage.DeviceId != source.DeviceId || tensor.ElementType != DType.Float32 || !tensor.IsContiguous()) return false;
                if (rows == 1 && tensor.DimensionCount == 1 && tensor.Sizes[0] == columns)
                {
                    tensor = tensor.View(1, columns); temporaryViews.Add(tensor);
                }
                return tensor.DimensionCount == 2 && tensor.Sizes[0] == rows && tensor.Sizes[1] == columns &&
                    TryCreateStandardView(tensor, out view);
            }
            for (int b=0;b<16;++b)
            {
                var block=blocks[b];
                if (block?.Matrices?.Length != 7 || block.Norms?.Length != 6 || block.Clamps?.Length != 28) return false;
                var matrices=new GgmlTensorView2D[7]; var norms=new GgmlTensorView2D[6];
                for (int m=0;m<7;++m)
                    if (!View(block.Matrices[m],m==4 || m==5 ? 3072 : 768,m==6 ? 3072 : 768,out matrices[m])) return false;
                for (int n=0;n<6;++n)
                    if (!View(block.Norms[n],1,n==1 || n==2 ? 64 : 768,out norms[n])) return false;
                for (int c=0;c<28;c+=2)
                    if (!float.IsFinite(block.Clamps[c]) || !float.IsFinite(block.Clamps[c+1]) || block.Clamps[c]>block.Clamps[c+1]) return false;
                views[b] = new GgmlGemma4VisionStackBlock {
                    Q=matrices[0],K=matrices[1],V=matrices[2],Out=matrices[3],Gate=matrices[4],Up=matrices[5],Down=matrices[6],
                    Ln1=norms[0],QNorm=norms[1],KNorm=norms[2],AttnPostNorm=norms[3],Ln2=norms[4],FfnPostNorm=norms[5] };
                for (int c=0;c<28;++c) views[b].Clamps[c]=block.Clamps[c];
            }
            fixed (GgmlGemma4VisionStackBlock* blockPointer=views)
            fixed (int* px=posX,py=posY)
            {
                var args = new GgmlGemma4VisionStackArgs {
                    StructBytes=sizeof(GgmlGemma4VisionStackArgs),Version=1,BlockCount=16,
                    BlockStructBytes=sizeof(GgmlGemma4VisionStackBlock),Input=inputView,Result=resultView,
                    Blocks=(IntPtr)blockPointer,PosX=(IntPtr)px,PosY=(IntPtr)py,PositionCount=posX.Length,
                    Epsilon=epsilon,RopeTheta=ropeTheta,Heads=12,HeadDim=64 };
                GgmlNative.SetActiveRankIfUnpinned(destination.DeviceId);
                return GgmlNative.Gemma4VisionStack(in args);
            }
        }
        finally
        {
            foreach (var view in temporaryViews) view.Dispose();
            GC.KeepAlive(blocks); GC.KeepAlive(input); GC.KeepAlive(result);
        }
    }
}
