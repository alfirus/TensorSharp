using System.Runtime.InteropServices;
using TensorSharp.MLX;
using Half=System.Half;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3MlxQ4KRepackTests
{
    [Theory]
    [InlineData(256,1)]
    [InlineData(512,3)]
    [InlineData(1280,67)]
    [InlineData(1024,2048)]
    public unsafe void RepackMatchesIndependentGgmlUnpackAndMlxPacking(int inner,int rows)
    {
        const int guard=11;
        int blocks=inner/256,rawBytes=rows*blocks*144,packedBytes=rows*inner/2,parameterCount=rows*blocks*8;
        byte[] raw=Enumerable.Repeat((byte)0x5A,rawBytes+2*guard).ToArray();
        byte[] expected=Enumerable.Repeat((byte)0xA5,packedBytes+2*guard).ToArray();
        Half[] expectedScales=Enumerable.Repeat((Half)37,parameterCount+2*guard).ToArray();
        Half[] expectedBiases=(Half[])expectedScales.Clone();
        var random=new Random(73819+rows+inner);
        ushort[] halfBits=[0,0x8000,1,0x3C00,0xBC00,0x3000,0x7BFF,0x0400];
        for(int row=0;row<rows;++row) for(int block=0;block<blocks;++block)
        {
            int start=guard+(row*blocks+block)*144;
            random.NextBytes(raw.AsSpan(start,144));
            Half d=BitConverter.UInt16BitsToHalf(halfBits[(row+block)%halfBits.Length]);
            Half min=BitConverter.UInt16BitsToHalf(halfBits[(row*3+block+2)%halfBits.Length]);
            BitConverter.TryWriteBytes(raw.AsSpan(start),BitConverter.HalfToUInt16Bits(d));
            BitConverter.TryWriteBytes(raw.AsSpan(start+2),BitConverter.HalfToUInt16Bits(min));
            byte[] scales=Enumerable.Range(0,8).Select(_=>(byte)random.Next(64)).ToArray();
            byte[] minima=Enumerable.Range(0,8).Select(_=>(byte)random.Next(64)).ToArray();
            // Independently encode the twelve GGML scale/min bytes from known
            // six-bit fields. Expected affine parameters use those known fields.
            for(int j=0;j<4;++j)
            {
                raw[start+4+j]=(byte)(scales[j]|((scales[j+4]>>4)<<6));
                raw[start+8+j]=(byte)(minima[j]|((minima[j+4]>>4)<<6));
                raw[start+12+j]=(byte)((scales[j+4]&15)|((minima[j+4]&15)<<4));
            }
            for(int group=0;group<8;++group)
            {
                int index=guard+(row*blocks+block)*8+group;
                expectedScales[index]=(Half)((float)d*scales[group]);
                expectedBiases[index]=(Half)(-(float)min*minima[group]);
            }
            // Walk logical GGML channels, unpack one four-bit quant, then
            // independently write the MLX low/high nibble for that channel.
            for(int channel=0;channel<256;++channel)
            {
                int quant=(raw[start+16+(channel/64)*32+channel%32]>>((channel%64)/32*4))&15;
                int index=guard+row*(inner/2)+block*128+channel/2;
                if(channel%2 == 0) expected[index]=(byte)quant;
                else expected[index]|=(byte)(quant<<4);
            }
        }
        byte[] original=(byte[])raw.Clone();
        foreach(bool parallel in new[]{false,true})
        {
            byte[] actual=Enumerable.Repeat((byte)0xA5,packedBytes+2*guard).ToArray();
            Half[] scales=Enumerable.Repeat((Half)37,parameterCount+2*guard).ToArray();
            Half[] biases=(Half[])scales.Clone();
            fixed(byte* source=raw,destination=actual)
            fixed(Half* scale=scales,bias=biases)
                MlxQ4KRepack.Pack((IntPtr)(source+guard),(IntPtr)(destination+guard),(IntPtr)(scale+guard),(IntPtr)(bias+guard),inner,rows,parallel);
            Assert.Equal(expected,actual);
            Assert.Equal(MemoryMarshal.AsBytes(expectedScales.AsSpan()).ToArray(),MemoryMarshal.AsBytes(scales.AsSpan()).ToArray());
            Assert.Equal(MemoryMarshal.AsBytes(expectedBiases.AsSpan()).ToArray(),MemoryMarshal.AsBytes(biases.AsSpan()).ToArray());
            Assert.Equal(original,raw);
        }
    }
}
