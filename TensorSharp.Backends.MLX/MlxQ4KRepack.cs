// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading.Tasks;

namespace TensorSharp.MLX;

/// <summary>Pure CPU lossless conversion of GGML Q4_K quants to MLX group-32
/// affine packing. Scale/bias rounding is identical to the existing F16 path;
/// H3's separately cached F32 parameters remain unchanged.</summary>
internal static unsafe class MlxQ4KRepack
{
    internal static void Pack(IntPtr source,IntPtr packed,IntPtr scales,IntPtr biases,int inner,int rows,bool parallel=true)
    {
        if(source == IntPtr.Zero || packed == IntPtr.Zero || scales == IntPtr.Zero || biases == IntPtr.Zero ||
            inner <= 0 || inner%256 != 0 || rows <= 0)
            throw new ArgumentException("Q4_K packing requires valid buffers and positive 256-aligned rows.");
        int blocks=inner/256;
        // One row owns disjoint output bytes, including the affine parameters.
        // Small tensors stay serial to avoid thread-pool scheduling overhead.
        if(parallel && rows >= 64 && (long)inner*rows >= 1L<<20)
            Parallel.For(0,rows,row => PackRow(source,packed,scales,biases,blocks,row));
        else
            for(int row=0;row<rows;++row) PackRow(source,packed,scales,biases,blocks,row);
    }

    private static void PackRow(IntPtr source,IntPtr packed,IntPtr scales,IntPtr biases,int blocks,int row)
    {
        byte* src=(byte*)source.ToPointer()+(long)row*blocks*144;
        byte* destination=(byte*)packed.ToPointer()+(long)row*blocks*128;
        System.Half* scale=(System.Half*)scales.ToPointer()+(long)row*blocks*8;
        System.Half* bias=(System.Half*)biases.ToPointer()+(long)row*blocks*8;
        for(int block=0;block<blocks;++block)
        {
            byte* input=src+block*144;
            byte* output=destination+block*128;
            float d=(float)BitConverter.UInt16BitsToHalf((ushort)(input[0]|input[1]<<8));
            float min=(float)BitConverter.UInt16BitsToHalf((ushort)(input[2]|input[3]<<8));
            byte* parameters=input+4;
            for(int group=0;group<8;++group)
            {
                int scaleByte=group<4 ? parameters[group]&63 :
                    (parameters[group+4]&15)|((parameters[group-4]>>6)<<4);
                int minByte=group<4 ? parameters[group+4]&63 :
                    (parameters[group+4]>>4)|((parameters[group]>>6)<<4);
                scale[block*8+group]=(System.Half)(d*scaleByte);
                bias[block*8+group]=(System.Half)(-min*minByte);
            }
            // A GGML pair stores 32 low-nibble quants followed logically by
            // 32 high-nibble quants. MLX stores each consecutive pair in one
            // byte. Write both groups directly, without an unpack buffer or
            // the generic bit packer's shifts, ORs and zero-fill dependency.
            byte* quants=input+16;
            for(int pair=0;pair<4;++pair)
            {
                byte* q=quants+pair*32;
                byte* low=output+pair*32;
                byte* high=low+16;
                for(int i=0;i<16;++i)
                {
                    int a=q[i*2],b=q[i*2+1];
                    low[i]=(byte)((a&15)|((b&15)<<4));
                    high[i]=(byte)((a>>4)|(b&240));
                }
            }
        }
    }
}
