// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Text;
using TensorSharp.Cuda.Interop;
namespace TensorSharp.Cuda;

/// <summary>Model-owned Float32 spectral square root. The embedded PTX uses
/// CUDA's correctly rounded sqrt instruction and a portable CUDA6/sm50 ISA;
/// it does not require recompiling the common inference module.</summary>
public sealed class CudaLtx25AudioSession : IDisposable
{
    private readonly CudaAllocator _allocator;
    private readonly CudaModule _module;
    private readonly IntPtr _sqrt, _log;
    private readonly object _gate=new();
    private bool _disposed;
    internal const string SpectralPtx="""
.version 6.0
.target sm_50
.address_size 64
.visible .entry ts_ltx25_audio_sqrt_f32(
 .param .u64 input_ptr, .param .u64 output_ptr, .param .u32 count)
{
 .reg .pred %p;
 .reg .b32 %r<6>;
 .reg .b64 %rd<6>;
 .reg .f32 %f<2>;
 ld.param.u64 %rd1,[input_ptr];
 ld.param.u64 %rd2,[output_ptr];
 ld.param.u32 %r5,[count];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x;
 mad.lo.s32 %r4,%r1,%r2,%r3;
 setp.ge.u32 %p,%r4,%r5;
 @%p bra DONE;
 mul.wide.u32 %rd3,%r4,4;
 add.s64 %rd4,%rd1,%rd3;
 ld.global.f32 %f0,[%rd4];
 sqrt.rn.f32 %f1,%f0;
 add.s64 %rd5,%rd2,%rd3;
 st.global.f32 [%rd5],%f1;
DONE:
 ret;
}
.visible .entry ts_ltx25_audio_log_clamped_f32(
 .param .u64 input_ptr, .param .u64 output_ptr, .param .u32 count)
{
 .reg .pred %p;
 .reg .b32 %r<6>;
 .reg .b64 %rd<6>;
 .reg .f32 %f<3>;
 ld.param.u64 %rd1,[input_ptr];
 ld.param.u64 %rd2,[output_ptr];
 ld.param.u32 %r5,[count];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x;
 mad.lo.s32 %r4,%r1,%r2,%r3;
 setp.ge.u32 %p,%r4,%r5;
 @%p bra DONE;
 mul.wide.u32 %rd3,%r4,4;
 add.s64 %rd4,%rd1,%rd3;
 ld.global.f32 %f0,[%rd4];
 max.f32 %f0,%f0,0f3727C5AC;
 lg2.approx.f32 %f1,%f0;
 mul.rn.f32 %f2,%f1,0f3F317218;
 add.s64 %rd5,%rd2,%rd3;
 st.global.f32 [%rd5],%f2;
DONE:
 ret;
}
""";
    public CudaLtx25AudioSession(CudaAllocator allocator)
    {
        _allocator=allocator??throw new ArgumentNullException(nameof(allocator));
        allocator.Context.MakeCurrent();_module=CudaModule.LoadFromBytes(Encoding.ASCII.GetBytes(SpectralPtx));
        try{_sqrt=_module.GetFunction("ts_ltx25_audio_sqrt_f32");_log=_module.GetFunction("ts_ltx25_audio_log_clamped_f32");}catch{_module.Dispose();throw;}
    }
    public bool TrySqrt(Tensor output,Tensor input)=>TryOperation(output,input,_sqrt);
    /// <summary>log(max(input,1e-5)), using native Float32 log2 and ln(2).</summary>
    public bool TryLogClamp(Tensor output,Tensor input)=>TryOperation(output,input,_log);
    private unsafe bool TryOperation(Tensor output,Tensor input,IntPtr operation)
    {
        lock(_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            if(output==null||input==null||!output.Sizes.SequenceEqual(input.Sizes)
                ||!CudaKernelOps.TryGetContiguousFloat(output,out var ys,out var yp,out int count)
                ||!CudaKernelOps.TryGetContiguousFloat(input,out var xs,out var xp,out _)
                ||!ReferenceEquals(ys.AllocatorImpl,_allocator)||!ReferenceEquals(xs.AllocatorImpl,_allocator))return false;
            // Exact in-place operation is supported; shifted overlap races.
            ulong a=unchecked((ulong)xp.ToInt64()),b=unchecked((ulong)yp.ToInt64()),bytes=(ulong)count*sizeof(float);
            if(a!=b&&a<b+bytes&&b<a+bytes)return false;
            if(count==0)return true;
            _allocator.Context.MakeCurrent();xs.EnsureDeviceCurrent();
            if(output.StorageOffset!=0||output.ElementCount()!=output.Storage.ElementCount)ys.EnsureDeviceCurrent();
            void** args=stackalloc void*[3]{&xp,&yp,&count};
            CudaDriverApi.cuLaunchKernel(operation,checked((uint)((count+255L)/256)),1,1,256,1,1,0,
                _allocator.Stream.Handle,(IntPtr)args,IntPtr.Zero).ThrowOnError();ys.MarkDeviceModified();return true;
        }
    }
    public void Dispose()
    {
        lock(_gate){if(_disposed)return;_disposed=true;_allocator.Context.MakeCurrent();_allocator.Stream.Synchronize();_module.Dispose();}
    }
}
