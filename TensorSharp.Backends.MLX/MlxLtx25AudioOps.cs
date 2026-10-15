// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
namespace TensorSharp.MLX;

/// <summary>Float32 spectral magnitude without a device-to-host fallback.</summary>
public static class MlxLtx25AudioOps
{
    public static bool TrySqrt(Tensor output,Tensor input)
    {
        if(output?.Storage is not MlxStorage os||input?.Storage is not MlxStorage xs
            ||output.ElementType!=DType.Float32||input.ElementType!=DType.Float32
            ||!output.IsContiguous()||!input.IsContiguous()||!output.Sizes.SequenceEqual(input.Sizes)
            ||!ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl))return false;
        return MlxWorker.Shared.Invoke(()=>
        {
            MlxNative.MlxArray view=default,result=default;
            try
            {
                view=xs.CreateArrayView(input);result=MlxNative.Unary(MlxNative.MlxUnaryOp.Sqrt,view);
                if(output.StorageOffset==0&&output.ElementCount()==output.Storage.ElementCount)
                {os.ReplaceDeviceArray(result);result=default;}
                else os.UpdateDeviceSlice(output,result);
                return true;
            }
            finally{MlxNative.FreeArray(view);MlxNative.FreeArray(result);}
        });
    }
}
