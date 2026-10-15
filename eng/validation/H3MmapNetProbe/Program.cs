using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Text.Json;

if(args.Length != 1) throw new ArgumentException("Pass one existing read-only model file");
using var stream = new FileStream(args[0],FileMode.Open,FileAccess.Read,FileShare.Read);
// Exact mapping calls used by GgufFile.EnsureMappedView; no GPU or model compute.
using var map = MemoryMappedFile.CreateFromFile(stream,null,0,MemoryMappedFileAccess.Read,
    HandleInheritability.None,leaveOpen:true);
using var view = map.CreateViewAccessor(0,0,MemoryMappedFileAccess.Read);
unsafe
{
    byte* pointer=null;
    view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
    try
    {
        pointer+=view.PointerOffset;
        ulong address=(ulong)pointer;
        var selected=new List<string>();
        bool active=false;
        foreach(string line in File.ReadLines("/proc/self/smaps"))
        {
            int dash=line.IndexOf('-'), space=line.IndexOf(' ');
            if(dash>0 && space>dash && ulong.TryParse(line[..dash],NumberStyles.HexNumber,CultureInfo.InvariantCulture,out ulong start) &&
                ulong.TryParse(line[(dash+1)..space],NumberStyles.HexNumber,CultureInfo.InvariantCulture,out ulong end))
            {
                active=address>=start && address<end;
                if(active) selected.Add(line);
            }
            else if(active && (line.StartsWith("VmFlags:") || line.StartsWith("KernelPageSize:") ||
                line.StartsWith("MMUPageSize:") || line.StartsWith("Rss:") || line.StartsWith("Shared_Clean:") ||
                line.StartsWith("Private_Clean:") || line.StartsWith("Private_Dirty:"))) selected.Add(line);
        }
        Console.WriteLine(JsonSerializer.Serialize(new {pid=Environment.ProcessId,path=Path.GetFullPath(args[0]),
            bytes=stream.Length,address=address.ToString("x"),firstByte=*pointer,smaps=selected,
            scope="Exact System.IO.MemoryMappedFiles read-only calls from GgufFile.EnsureMappedView;CPU-only;one byte read"}));
    }
    finally {view.SafeMemoryMappedViewHandle.ReleasePointer();}
}
