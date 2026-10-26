// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    internal static class NativeDequant
    {
        // When true (set by the pure-C# CPU backend, see ModelBase ctor), route every dequant
        // through the managed implementation for types it supports — the CPU backend must not
        // call into native ggml at all. The managed dequant is bit-exact vs native (verified),
        // so this is behaviour-preserving; it is only enabled for BackendType.Cpu so GGML/CUDA
        // backends keep their native load-time dequant.
        internal static bool PreferManaged;

        private static bool UseManaged(int ggmlType)
            => PreferManaged && ManagedQuantizedOps.SupportsDequantization((GgmlTensorType)ggmlType);

        private static void RefuseStorageOnlyType(int ggmlType)
        {
            if ((GgmlTensorType)ggmlType is GgmlTensorType.MXFP8 or GgmlTensorType.MXFP8_Unsloth)
                throw new NotSupportedException("MXFP8 header sizing is supported, but generic dequantization is not. " +
                    "Use the TensorSharp-owned DeepSeek V4.1 Engram loader; unchanged upstream ggml does not implement this type.");
        }

        // One notice for the whole process: once the native library is missing it is
        // missing for every tensor, and the load loop hits these catches thousands of times.
        private static int _managedFallbackWarned;

        private static void WarnManagedFallback(Exception ex)
        {
            if (System.Threading.Interlocked.Exchange(ref _managedFallbackWarned, 1) != 0)
                return;
            Console.Error.WriteLine(
                $"WARNING: native GgmlOps dequantization is unavailable ({ex.Message}); load-time " +
                "dequantization is running on the managed C# path instead, so model loading will be " +
                "substantially slower. Verify the GgmlOps native library is present. Reported once.");
        }

        public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            RefuseStorageOnlyType(ggmlType);
            if (UseManaged(ggmlType))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
                return;
            }
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                WarnManagedFallback(ex);
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
        }

        public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            RefuseStorageOnlyType(ggmlType);
            if (UseManaged(ggmlType))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
                return;
            }
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                WarnManagedFallback(ex);
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
        }

        public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            RefuseStorageOnlyType(ggmlType);
            if (UseManaged(ggmlType))
            {
                ManagedQuantizedOps.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
                return;
            }
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                WarnManagedFallback(ex);
                ManagedQuantizedOps.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
        }

        public static long RowSize(int ggmlType, long ne)
        {
            RefuseStorageOnlyType(ggmlType);
            if (UseManaged(ggmlType))
                return ManagedQuantizedOps.RowSize(ggmlType, ne);
            try
            {
                return GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, ne);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                WarnManagedFallback(ex);
                var type = (GgmlTensorType)ggmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                if (ne % blockSize != 0)
                    throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

                return (ne / blockSize) * GgufFile.GetTypeSize(type);
            }
        }

        private static bool ShouldUseManagedFallback(Exception ex)
        {
            if (ex is DllNotFoundException)
                return true;

            if (ex is TypeInitializationException tie && tie.InnerException != null)
                return ShouldUseManagedFallback(tie.InnerException);

            return false;
        }
    }
}

