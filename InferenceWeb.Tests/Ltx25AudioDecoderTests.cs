using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

/// <summary>Independently generated unchanged-original complete audio graph.
/// The checkpoint reduces hidden widths only; causal, stereo, all AMP stages,
/// checkpoint spectral buffers and the learned 48k bandwidth extension execute.</summary>
public sealed class Ltx25AudioDecoderTests
{
    private static readonly JsonElement Oracle=Load();
    public static IEnumerable<object[]> Cases=>Enumerable.Range(0,Oracle.GetProperty("cases").GetArrayLength()).Select(i=>new object[]{i});
    [Theory,MemberData(nameof(Cases))]
    public void CompleteAudioDecoderAndBandwidthExtensionMatchOriginal(int index)
    {
        using var fixture=new Checkpoint();using var backend=new Backend();using var decoder=new Ltx25AudioDecoder(fixture.Path,backend.Allocator);
        var c=Oracle.GetProperty("cases")[index];var traces=new Dictionary<string,float[]>();
        var output=decoder.Decode(Floats(c,"latents"),c.GetProperty("tokens").GetInt32(),trace:(n,v,s)=>traces[n]=v);
        foreach(string name in new[]{"mel","native16k","bweMel","bweResidual","pcm48k"})Equal(Floats(c,name),traces[name],name=="bweMel"?2e-4:2e-5,name);
        Assert.Equal(48000,output.SampleRate);Assert.Equal(2,output.ChannelCount);
        Assert.Equal((4*c.GetProperty("tokens").GetInt32()-3)*480,output.SampleCount);
        for(int t=0;t<output.SampleCount;t++)for(int channel=0;channel<2;channel++)Assert.Equal(traces["pcm48k"][t*2+channel],output.Channels[channel][t]);
    }
    [Fact]
    public void OptionalLengthTrimsWithoutPaddingAndRepeatedDecodeIsStable()
    {
        using var fixture=new Checkpoint();using var backend=new Backend();using var decoder=new Ltx25AudioDecoder(fixture.Path,backend.Allocator);
        var c=Oracle.GetProperty("cases")[1];var input=Floats(c,"latents");var saved=(float[])input.Clone();
        var full=decoder.Decode(input,2);var shortAudio=decoder.Decode(input,2,17);var longer=decoder.Decode(input,2,100000);
        Assert.Equal(saved,input);Assert.Equal(17,shortAudio.SampleCount);Assert.Equal(full.SampleCount,longer.SampleCount);
        for(int ch=0;ch<2;ch++){Assert.Equal(full.Channels[ch],longer.Channels[ch]);Assert.Equal(full.Channels[ch].Take(17),shortAudio.Channels[ch]);}
    }
    [Fact]
    public void InvalidGeometryNonFiniteInputCancellationAndDisposalAreRejected()
    {
        using var fixture=new Checkpoint();using var backend=new Backend();var decoder=new Ltx25AudioDecoder(fixture.Path,backend.Allocator);
        Assert.Throws<ArgumentException>(()=>decoder.Decode(new float[127],1));Assert.Throws<ArgumentException>(()=>decoder.Decode(Array.Empty<float>(),0));
        var bad=new float[128];bad[7]=float.NaN;Assert.Throws<ArgumentException>(()=>decoder.Decode(bad,1));
        Assert.Throws<ArgumentOutOfRangeException>(()=>decoder.Decode(new float[128],1,0));
        Assert.Throws<OperationCanceledException>(()=>decoder.Decode(new float[128],1,cancellationToken:new CancellationToken(true)));
        decoder.Dispose();Assert.Throws<ObjectDisposedException>(()=>decoder.Decode(new float[128],1));
    }
    private static void Equal(float[] expected,float[] actual,double tolerance,string label)
    {
        Assert.Equal(expected.Length,actual.Length);double largest=0;int at=0;
        for(int i=0;i<actual.Length;i++){Assert.True(float.IsFinite(actual[i]),$"{label}[{i}] is nonfinite");double error=Math.Abs(expected[i]-actual[i]);if(error>largest){largest=error;at=i;}}
        Assert.True(largest<=tolerance,$"{label}: max error {largest:G9} at {at}: expected {expected[at]:G9}, got {actual[at]:G9} (tolerance {tolerance})");
    }
    private static float[] Floats(JsonElement e,string name)=>e.GetProperty(name).EnumerateArray().Select(v=>v.GetSingle()).ToArray();
    private static Stream Resource(string name)=>Assembly.GetExecutingAssembly().GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Ltx25."+name)??throw new InvalidOperationException("Missing audio fixture "+name);
    private static JsonElement Load(){using var stream=Resource("audio-oracle.json");using var json=JsonDocument.Parse(stream);return json.RootElement.Clone();}
    private sealed class Checkpoint:IDisposable
    {
        public string Path{get;}=System.IO.Path.Combine(System.IO.Path.GetTempPath(),"ltx25-audio-"+Guid.NewGuid().ToString("N")+".safetensors");
        public Checkpoint(){using var source=Resource("audio-decoder.bin");using(var target=File.Create(Path))source.CopyTo(target);Assert.Equal(Oracle.GetProperty("checkpoint_sha256").GetString(),Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path))));}
        public void Dispose()=>File.Delete(Path);
    }
    private sealed class Backend:IDisposable
    {
        public IAllocator Allocator{get;}
        public Backend(){Allocator=Environment.GetEnvironmentVariable("TS_TEST_LTX25_AUDIO_BACKEND")?.ToLowerInvariant() switch{null or "" or "cpu"=>new CpuAllocator(BlasEnum.DotNet),"cuda"=>new CudaAllocator(0),"mlx"=>new MlxAllocator(0),var s=>throw new ArgumentException("Unknown audio test backend "+s)};}
        public void Dispose(){(Allocator as IDisposable)?.Dispose();}
    }
}
