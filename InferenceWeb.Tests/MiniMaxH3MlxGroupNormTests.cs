using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3MlxGroupNormTests
{
    [MlxTheory]
    [InlineData(2, 19, 17, 8, 4, 12000f, 80000f)]
    [InlineData(2, 7, 11, 64, 32, 131072f, .25f)]
    [InlineData(2, 64, 64, 64, 32, 131072f, .25f)]
    [InlineData(1, 1, 1, 16, 4, -17f, 3f)]
    [InlineData(2, 3, 5, 6, 3, 3f, .5f)]
    public void FusedGroupNormMatchesCenteredDoubleOracleWithOptionalAffine(
        int frames,int height,int width,int channels,int groups,float offset,float amplitude)
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator=new MlxAllocator();
            int count=frames*height*width*channels;
            float[] values=Enumerable.Range(0,count).Select(i => offset+MathF.Sin(i*.73f)*amplitude).ToArray();
            float[] gains=Enumerable.Range(0,channels).Select(i => .6f+i*.03f).ToArray();
            float[] biases=Enumerable.Range(0,channels).Select(i => -.2f+i*.07f).ToArray();
            using var gain=new Tensor(allocator,DType.Float32,channels); gain.SetElementsAsFloat(gains);
            using var bias=new Tensor(allocator,DType.Float32,channels); bias.SetElementsAsFloat(biases);
            foreach (bool affine in new[]{false,true})
            {
                using var input=new Tensor(allocator,DType.Float32,frames,height,width,channels); input.SetElementsAsFloat(values);
                Assert.True(MlxMiniMaxH3Ops.TryGroupNormSilu(input,input,affine ? gain : null,affine ? bias : null,groups,1e-6f));
                float[] actual=input.GetElementsAsFloat(count);
                int rows=height*width, groupWidth=channels/groups, groupCount=rows*groupWidth;
                for(int f=0;f<frames;f++) for(int g=0;g<groups;g++)
                {
                    double sum=0;
                    for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++)
                        sum+=values[(f*rows+row)*channels+g*groupWidth+c];
                    float mean=(float)(sum/groupCount);
                    double squares=0;
                    for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++)
                    { float centered=values[(f*rows+row)*channels+g*groupWidth+c]-mean; squares+=(double)centered*centered; }
                    double inverse=1/Math.Sqrt(squares/groupCount+1e-6);
                    for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++)
                    {
                        int ch=g*groupWidth+c,index=(f*rows+row)*channels+ch;
                        double normalized=(values[index]-mean)*inverse;
                        if(affine) normalized=normalized*gains[ch]+biases[ch];
                        double expected=normalized/(1+Math.Exp(-normalized));
                        Assert.True(float.IsFinite(actual[index]));
                        Assert.InRange(Math.Abs(expected-actual[index]),0,2e-5);
                    }
                }
            }
        });

    [MlxFact]
    public void FusedGroupNormUpdatesOnlySelectedContiguousFrameView()
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator=new MlxAllocator();
            using var storage=new Tensor(allocator,DType.Float32,3,3,5,8);
            float[] values=Enumerable.Range(0,360).Select(i => MathF.Sin(i*.19f)).ToArray();
            storage.SetElementsAsFloat(values);
            using var frame=storage.Narrow(0,1,1);
            using var separate=new Tensor(allocator,DType.Float32,1,3,5,8);
            Assert.True(MlxMiniMaxH3Ops.TryGroupNormSilu(separate,frame,null,null,4,1e-6f));
            float[] expected=separate.GetElementsAsFloat(120);
            Assert.True(MlxMiniMaxH3Ops.TryGroupNormSilu(frame,frame,null,null,4,1e-6f));
            float[] actual=storage.GetElementsAsFloat(360);
            Assert.Equal(values.Take(120),actual.Take(120));
            Assert.Equal(expected,actual.Skip(120).Take(120));
            Assert.Equal(values.Skip(240),actual.Skip(240));
        });

    [MlxFact]
    public void FusedGroupNormRejectsInvalidGeometryBeforeChangingOutput()
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator=new MlxAllocator();
            using var input=new Tensor(allocator,DType.Float32,1,3,5,8);
            using var output=new Tensor(allocator,DType.Float32,1,3,5,8); Ops.Fill(output,37);
            using var wrongShape=new Tensor(allocator,DType.Float32,1,5,3,8);
            using var wrongGain=new Tensor(allocator,DType.Float32,7);
            using var half=new Tensor(allocator,DType.Float16,1,3,5,8);
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,null,null,3,1e-6f));
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,null,null,4,0));
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,null,null,4,float.NaN));
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,wrongShape,null,null,4,1e-6f));
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,wrongGain,null,4,1e-6f));
            Assert.False(MlxMiniMaxH3Ops.TryGroupNormSilu(output,half,null,null,4,1e-6f));
            Assert.All(output.GetElementsAsFloat(120),value => Assert.Equal(37,value));
        });
}
