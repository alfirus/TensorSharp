// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.InteropServices;
using UIKit;

namespace TensorAgent.Maui;

public static class Program
{
    private const string MlxPrecisionVariable = "MLX_ENABLE_TF32";

    // The Mac Catalyst entry point. UIApplication.Main never returns; the MAUI app is
    // created by AppDelegate.CreateMauiApp.
    private static void Main(string[] args)
    {
        ApplyMlxPrecisionDefault();
        UIApplication.Main(args, null, typeof(AppDelegate));
    }

    private static void ApplyMlxPrecisionDefault()
    {
        // MLX caches TF32 on its first GEMM. Set the app's default before MAUI
        // creates the host or restores any model, so choosing LTX after another
        // MLX model retains the validated Float32 precision. A launch override
        // remains authoritative; other TensorSharp hosts keep their own defaults.
        if (Environment.GetEnvironmentVariable(MlxPrecisionVariable) is not null) return;
        // Mono's managed environment copy alone is not visible to native getenv.
        if (SetNativeEnvironment(MlxPrecisionVariable, "0", 1) != 0)
            throw new InvalidOperationException("Unable to configure TensorAgent MLX Float32 precision.");
        Environment.SetEnvironmentVariable(MlxPrecisionVariable, "0");
        Console.WriteLine("TensorAgent: MLX_ENABLE_TF32=0 before the first model load (Float32 precision)");
    }

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "setenv", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SetNativeEnvironment(string name, string value, int overwrite);
}
