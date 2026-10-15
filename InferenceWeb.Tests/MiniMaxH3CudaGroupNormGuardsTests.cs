using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3CudaGroupNormGuardsTests
{
    [CudaFact]
    public void GroupNormRejectsInvalidGeometryAndForeignAffineBeforeChangingOutput()
    {
        using var allocator=new CudaAllocator();
        using var foreignAllocator=new CudaAllocator();
        using var input=new Tensor(allocator,DType.Float32,1,3,5,8);
        using var output=new Tensor(allocator,DType.Float32,1,3,5,8); Ops.Fill(output,37);
        using var wrongShape=new Tensor(allocator,DType.Float32,1,5,3,8);
        using var wrongGain=new Tensor(allocator,DType.Float32,7);
        using var half=new Tensor(allocator,DType.Float16,1,3,5,8);
        using var foreignGain=new Tensor(foreignAllocator,DType.Float32,8);
        foreach(int groups in new[]{0,-1,3})
            Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,null,null,groups,1e-6f));
        foreach(float eps in new[]{0f,-1f,float.NaN,float.PositiveInfinity})
            Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,null,null,4,eps));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,wrongShape,null,null,4,1e-6f));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,wrongGain,null,4,1e-6f));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,null,wrongGain,4,1e-6f));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,foreignGain,null,4,1e-6f));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,input,null,foreignGain,4,1e-6f));
        Assert.False(CudaMiniMaxH3Ops.TryGroupNormSilu(output,half,null,null,4,1e-6f));
        Assert.All(output.GetElementsAsFloat(120),value => Assert.Equal(37,value));
    }
}
