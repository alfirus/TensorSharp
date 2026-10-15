using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class SafetensorsFloat16ConversionTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(65536)]
    [InlineData(1048617)]
    public unsafe void VectorWideningMatchesEveryScalarHalfBitPatternWithOffsetsAndChunkTails(int count)
    {
        const int sourceOffset=3,destinationOffset=5,guard=11;
        ushort[] source=Enumerable.Repeat((ushort)0x55AA,count+sourceOffset+guard).ToArray();
        float[] actual=Enumerable.Repeat(37.25f,count+destinationOffset+guard).ToArray();
        for(int i=0;i<count;++i) source[sourceOffset+i]=unchecked((ushort)(i*17));
        ushort[] original=(ushort[])source.Clone();
        fixed(ushort* input=source)
        fixed(float* output=actual)
            SafetensorsKernels.F16ToF32(input+sourceOffset,output+destinationOffset,count);
        // The odd multiplier visits all 65536 Half encodings, including signed
        // zeros, subnormals, infinities and every signaling/quiet NaN payload.
        // Compare F32 bits so NaN payload changes cannot hide behind equality.
        for(int i=0;i<count;++i)
        {
            float expected=(float)BitConverter.UInt16BitsToHalf(source[sourceOffset+i]);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected),BitConverter.SingleToInt32Bits(actual[destinationOffset+i]));
        }
        Assert.All(actual.Take(destinationOffset),value => Assert.Equal(37.25f,value));
        Assert.All(actual.Skip(destinationOffset+count),value => Assert.Equal(37.25f,value));
        Assert.Equal(original,source);
    }
}
