using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
namespace InferenceWeb.Tests;

public sealed class Ltx25AudioSpectralTests
{
    [CudaFact]
    public void CudaCorrectlyRoundedSqrtHandlesZeroSubnormalsAndGuardedViews()
    {
        using var allocator=new CudaAllocator();using var session=new CudaLtx25AudioSession(allocator);
        float[] values={0,float.Epsilon,1e-30f,.0001f,.25f,1,2,3,4,1e20f,float.MaxValue};
        using var input=new Tensor(allocator,DType.Float32,values.Length);input.SetElementsAsFloat(values);
        using var storage=new Tensor(allocator,DType.Float32,values.Length+2);Ops.Fill(storage,-91);
        using var output=storage.Narrow(0,1,values.Length);Assert.True(session.TrySqrt(output,input));
        var result=storage.GetElementsAsFloat(values.Length+2);Assert.Equal(-91,result[0]);Assert.Equal(-91,result[^1]);
        for(int i=0;i<values.Length;i++)Assert.Equal(MathF.Sqrt(values[i]),result[i+1]);
        Assert.True(session.TrySqrt(input,input));Assert.Equal(values.Select(MathF.Sqrt),input.GetElementsAsFloat(values.Length));
    }
    [CudaFact]
    public void CudaRejectsShiftedAliasAndUnsupportedLayoutBeforeWrite()
    {
        using var allocator=new CudaAllocator();using var session=new CudaLtx25AudioSession(allocator);
        using var storage=new Tensor(allocator,DType.Float32,8);storage.SetElementsAsFloat(Enumerable.Range(0,8).Select(i=>(float)i+1).ToArray());
        using var a=storage.Narrow(0,0,7);using var b=storage.Narrow(0,1,7);Assert.False(session.TrySqrt(b,a));
        using var shaped=storage.View(2,4);using var transposed=shaped.Transpose();Assert.False(session.TrySqrt(transposed,transposed));
        Assert.Equal(Enumerable.Range(0,8).Select(i=>(float)i+1),storage.GetElementsAsFloat(8));
        session.Dispose();Assert.Throws<ObjectDisposedException>(()=>session.TrySqrt(storage,storage));
    }
    [CudaFact]
    public void CudaLogMelClampsSilenceAndMatchesIndependentScalarReference()
    {
        using var allocator=new CudaAllocator();using var session=new CudaLtx25AudioSession(allocator);
        float[] values={-1,0,float.Epsilon,1e-6f,1e-5f,.0001f,.25f,1,2,3,4,1e20f,float.MaxValue};
        using var input=new Tensor(allocator,DType.Float32,values.Length);input.SetElementsAsFloat(values);
        Assert.True(session.TryLogClamp(input,input));var actual=input.GetElementsAsFloat(values.Length);
        for(int i=0;i<values.Length;i++)Assert.True(Math.Abs(MathF.Log(Math.Max(values[i],1e-5f))-actual[i])<=1e-5f);
    }
    [MlxFact]
    public void MlxSqrtPreservesZeroAndMatchesIndependentScalarReference()
    {
        // Metal flushes subnormal inputs. Test zero and the normal Float32
        // spectral domain; mel values are subsequently floored at 1e-5.
        using var allocator=new MlxAllocator();float[] values={0,1e-30f,.0001f,.25f,1,2,3,4,1e20f,float.MaxValue};
        using var input=new Tensor(allocator,DType.Float32,values.Length);input.SetElementsAsFloat(values);
        using var output=new Tensor(allocator,DType.Float32,values.Length);Assert.True(MlxLtx25AudioOps.TrySqrt(output,input));
        var actual=output.GetElementsAsFloat(values.Length);
        for(int i=0;i<values.Length;i++)Assert.True(Math.Abs(MathF.Sqrt(values[i])-actual[i])<=Math.Max(float.Epsilon,MathF.Sqrt(values[i])*2e-7f));
    }
}
