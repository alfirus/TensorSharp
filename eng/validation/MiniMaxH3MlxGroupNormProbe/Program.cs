using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using TensorSharp;
using TensorSharp.MLX;

// Operator benchmark, not a complete video/model benchmark. Reflect only the
// internal synchronous evaluation barrier so timings exclude output downloads.
// The expression is the H3 MLX implementation before the fused kernel.
const BindingFlags flags=BindingFlags.Static|BindingFlags.Instance|BindingFlags.NonPublic;
Type native=typeof(MlxAllocator).Assembly.GetType("TensorSharp.MLX.MlxNative",throwOnError:true)!;
Type array=native.GetNestedType("MlxArray",BindingFlags.Public|BindingFlags.NonPublic)!;
MethodInfo view=typeof(MlxStorage).GetMethod("CreateArrayView",flags,null,[typeof(Tensor)],null)!;
MethodInfo eval=native.GetMethod("Eval",flags,null,[array],null)!;
MethodInfo free=native.GetMethod("FreeArray",flags,null,[array],null)!;
int iterations=args.Length > 0 ? int.Parse(args[0]) : 7;
if(iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations));
MlxWorker.Shared.Invoke(() =>
{
    using var allocator=new MlxAllocator();
    foreach(var shape in new[]{(F:1,H:32,W:32,C:256),(F:1,H:96,W:160,C:128),(F:2,H:96,W:160,C:64),
        (F:5,H:64,W:96,C:128),(F:17,H:64,W:96,C:128),(F:5,H:32,W:48,C:256),(F:5,H:8,W:12,C:512),(F:5,H:1,W:2,C:1024)})
    {
        int count=shape.F*shape.H*shape.W*shape.C;
        float[] values=Enumerable.Range(0,count).Select(i=>MathF.Sin(i*.19f)*3).ToArray();
        using var input=new Tensor(allocator,DType.Float32,shape.F,shape.H,shape.W,shape.C); input.SetElementsAsFloat(values);
        using var output=new Tensor(allocator,DType.Float32,input.Sizes);
        using var gain=new Tensor(allocator,DType.Float32,shape.C); Ops.Fill(gain,.7f);
        using var bias=new Tensor(allocator,DType.Float32,shape.C); Ops.Fill(bias,.1f);
        float[]? reference=null,actual=null;
        for(int i=0;i<2;i++)
        {
            Expression(output,input,gain,bias,32,1e-6f); reference=output.GetElementsAsFloat(count);
            if(!MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,gain,bias,32,1e-6f)) throw new InvalidOperationException("Fused operation unavailable");
            actual=output.GetElementsAsFloat(count);
        }
        double maxError=reference!.Zip(actual!, (a,b)=>Math.Abs((double)a-b)).Max();
        if(maxError > 2e-5) throw new InvalidOperationException($"Group norm comparison failed: {maxError}");
        var expressionTimes=new List<double>(); var fusedTimes=new List<double>();
        for(int i=0;i<iterations;i++)
        {
            // Alternate order to avoid assigning warm-device drift to one path.
            if(i%2 == 0) { Measure(false,expressionTimes); Measure(true,fusedTimes); }
            else { Measure(true,fusedTimes); Measure(false,expressionTimes); }
        }
        expressionTimes.Sort(); fusedTimes.Sort();
        double expressionMs=expressionTimes[iterations/2],fusedMs=fusedTimes[iterations/2];
        Console.WriteLine(JsonSerializer.Serialize(new{backend="mlx",shape=new[]{shape.F,shape.H,shape.W,shape.C},groups=32,
            iterations,maxAbsoluteDifferenceFromExpression=maxError,expressionMedianMs=expressionMs,optimizedMedianMs=fusedMs,
            selectedPath=shape.C < 128 && shape.H*shape.W >= 4096 ? "contiguous expression" : "fused Metal",
            speedup=expressionMs/fusedMs,scope="operator with synchronous device evaluation; no model weights or end-to-end claim"}));
        void Measure(bool fused,List<double> timings)
        {
            var watch=Stopwatch.StartNew();
            if(fused) { if(!MlxMiniMaxH3Ops.TryGroupNormSilu(output,input,gain,bias,32,1e-6f)) throw new InvalidOperationException(); }
            else Expression(output,input,gain,bias,32,1e-6f);
            object outputView=view.Invoke(output.Storage,[output])!;
            try { eval.Invoke(null,[outputView]); }
            finally { free.Invoke(null,[outputView]); }
            watch.Stop(); timings.Add(watch.Elapsed.TotalMilliseconds);
        }
    }
});

static void Expression(Tensor output,Tensor input,Tensor gain,Tensor bias,int groups,float eps)
{
    long frames=input.Sizes[0],rows=input.Sizes[1]*input.Sizes[2],channels=input.Sizes[3],width=channels/groups;
    using var shaped=input.View(frames,rows,groups,width);
    using var permuted=shaped.Permute(0,2,1,3);
    using var contiguous=Ops.NewContiguous(permuted);
    using var normalizedInput=contiguous.View(frames*groups,rows*width);
    using var ones=new Tensor(input.Allocator,DType.Float32,rows*width); Ops.Fill(ones,1);
    using var normalized=Ops.LayerNorm(null,normalizedInput,ones,null,eps);
    using var groupsView=normalized.View(frames,groups,rows,width);
    using var restored=groupsView.Permute(0,2,1,3);
    using var restoredContiguous=Ops.NewContiguous(restored);
    using var values=restoredContiguous.View(frames*rows,channels);
    using var g=gain.View(1,channels); Ops.Mul(values,values,g);
    using var b=bias.View(1,channels); Ops.Add(values,values,b);
    using var activated=Ops.Sigmoid(null,values); Ops.Mul(values,values,activated);
    using var result=values.View(input.Sizes); Ops.Copy(output,result);
}
