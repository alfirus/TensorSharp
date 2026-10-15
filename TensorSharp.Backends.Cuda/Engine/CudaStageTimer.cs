// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Per-stage timing behind the whole-model engines' <c>TS_*_PERF</c> switches.
    /// <list type="bullet">
    /// <item>Level 2: each stage end synchronizes its stream and is charged the host time since the
    /// previous mark. Stages run serialized with the host's issue time between them, so rank stages
    /// by it; do not read throughput off it.</item>
    /// <item>Level 3: each stage end records an event instead, and after the step each stage is
    /// charged the GPU time between consecutive events on one device's stream. Nothing waits
    /// mid-step. Intervals include stream idle time while the host issues work. A device's first span
    /// starts at <see cref="SpanStart"/>, so a wait on another device belongs to no stage.</item>
    /// </list>
    /// </summary>
    internal sealed class CudaStageTimer : IDisposable
    {
        private readonly string _tag;
        private readonly string[] _names;
        private readonly double[] _ms;
        private long _t0;

        // Level 3: events recorded this step, in issue order, and a pool per context.
        private readonly List<(IntPtr Ctx, int Stage, IntPtr Ev)> _marks = new List<(IntPtr, int, IntPtr)>();
        private readonly Dictionary<IntPtr, Stack<IntPtr>> _pool = new Dictionary<IntPtr, Stack<IntPtr>>();

        public CudaStageTimer(string tag, int level, string[] stageNames)
        {
            _tag = tag;
            Level = level;
            _names = stageNames;
            _ms = new double[stageNames.Length];
        }

        /// <summary>0/1 off, 2 synchronized host time, 3 GPU time from events.</summary>
        public int Level { get; }

        /// <summary>Start a span on this device's stream: the first mark of a step, or a device
        /// taking over at a layer-split boundary once its wait is queued.</summary>
        public void SpanStart(CudaContext context, IntPtr stream)
        {
            if (Level == 2)
                _t0 = Stopwatch.GetTimestamp();
            else if (Level >= 3)
                Record(context, stream, -1);
        }

        /// <summary>Charge the work queued on this device's stream since its previous mark to
        /// <paramref name="stage"/>.</summary>
        public void End(CudaContext context, IntPtr stream, int stage)
        {
            if (Level == 2)
            {
                CudaDriverApi.cuStreamSynchronize(stream);
                long now = Stopwatch.GetTimestamp();
                _ms[stage] += (now - _t0) * 1000.0 / Stopwatch.Frequency;
                _t0 = now;
            }
            else if (Level >= 3)
            {
                Record(context, stream, stage);
            }
        }

        private void Record(CudaContext context, IntPtr stream, int stage)
        {
            // Events belong to a context and are recorded from it; the engine may have another
            // device current here (a boundary), so switch for the record and switch back.
            CudaDriverApi.cuCtxGetCurrent(out IntPtr previous).ThrowOnError();
            context.MakeCurrent();
            try
            {
                if (!_pool.TryGetValue(context.Handle, out Stack<IntPtr> free))
                    _pool[context.Handle] = free = new Stack<IntPtr>();
                if (!free.TryPop(out IntPtr ev))
                    CudaDriverApi.cuEventCreate(out ev, 0).ThrowOnError();
                try
                {
                    CudaDriverApi.cuEventRecord(ev, stream).ThrowOnError();
                    _marks.Add((context.Handle, stage, ev));
                }
                catch { free.Push(ev); throw; }
            }
            finally
            {
                if (previous != context.Handle)
                    CudaDriverApi.cuCtxSetCurrent(previous).ThrowOnError();
            }
        }

        /// <summary>Print the step's stages (the caller has synchronized every device, as the end of
        /// a forward does) and start the next step from zero.</summary>
        public void Report()
        {
            if (Level < 2)
                return;
            if (Level >= 3)
                Collect();
            var sb = new StringBuilder($"[{_tag}]   {(Level >= 3 ? "gpu" : "synced")} stages:");
            double total = 0;
            for (int i = 0; i < _ms.Length; i++)
            {
                if (_ms[i] == 0)
                    continue;
                sb.Append($" {_names[i]}={_ms[i]:F2}ms");
                total += _ms[i];
                _ms[i] = 0;
            }
            sb.Append($" | sum={total:F2}ms");
            Console.Error.WriteLine(sb.ToString());
        }

        /// <summary>Collect structured timings without synchronizing when the
        /// caller already completed its normal readback. Existing Report calls
        /// retain their synchronized collection behavior.</summary>
        internal double[] CollectAndReset(bool synchronize = false)
        {
            if (Level >= 3)
                Collect(synchronize);
            var result = (double[])_ms.Clone();
            Array.Clear(_ms, 0, _ms.Length);
            return result;
        }

        private void Collect(bool synchronize = true)
        {
            var last = new Dictionary<IntPtr, IntPtr>();
            foreach ((IntPtr ctx, int stage, IntPtr ev) in _marks)
            {
                if (stage >= 0 && last.TryGetValue(ctx, out IntPtr prev))
                {
                    if (synchronize)
                    {
                        CudaDriverApi.cuEventSynchronize(ev);
                        if (CudaDriverApi.cuEventElapsedTime(out float ms, prev, ev) == 0)
                            _ms[stage] += ms;
                    }
                    else
                    {
                        // CUDA_ERROR_NOT_READY is an integration error: normal
                        // callers must finish their existing readback first.
                        CudaDriverApi.cuEventElapsedTime(out float ms, prev, ev).ThrowOnError();
                        _ms[stage] += ms;
                    }
                }
                last[ctx] = ev;
            }
            foreach ((IntPtr ctx, _, IntPtr ev) in _marks)
                _pool[ctx].Push(ev);
            _marks.Clear();
        }

        public void Dispose()
        {
            // Exceptional exits may not have reached Report/Collect. Destroying
            // an event is safe while its recorded work is pending on the stream.
            foreach ((IntPtr ctx, _, IntPtr ev) in _marks)
            {
                CudaDriverApi.cuCtxSetCurrent(ctx);
                CudaDriverApi.cuEventDestroy(ev);
            }
            _marks.Clear();
            foreach (var kv in _pool)
            {
                CudaDriverApi.cuCtxSetCurrent(kv.Key);
                foreach (IntPtr ev in kv.Value)
                    CudaDriverApi.cuEventDestroy(ev);
            }
            _pool.Clear();
        }
    }
}
