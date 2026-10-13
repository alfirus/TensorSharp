using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

switch (args[0])
{
    case "arguments":
        Console.Write(JsonSerializer.Serialize(args.Skip(1).ToArray()));
        break;
    case "environment":
        Console.Write(Environment.GetEnvironmentVariable(args[1]));
        break;
    case "duplex":
        Task errors = Console.OpenStandardError().WriteAsync(new byte[96 * 1024]).AsTask();
        await Console.OpenStandardInput().CopyToAsync(Console.OpenStandardOutput());
        await errors;
        break;
    case "overflow":
        File.WriteAllText(args[1], Environment.ProcessId.ToString());
        await Console.OpenStandardOutput().WriteAsync(new byte[int.Parse(args[2])]);
        await Task.Delay(Timeout.Infinite);
        break;
    case "tree":
        var start = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
        start.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        start.ArgumentList.Add("sleep");
        using (Process child = Process.Start(start)!)
        {
            File.WriteAllText(args[1], $"{Environment.ProcessId},{child.Id}");
            await Task.Delay(Timeout.Infinite);
        }
        break;
    case "sleep":
        await Task.Delay(Timeout.Infinite);
        break;
    case "reject":
        Console.OpenStandardInput().Dispose();
        Console.Error.Write("fixture rejected request");
        return 17;
    case "invalid-utf8":
        await Console.OpenStandardError().WriteAsync(new byte[] { 255, 254, 0x61 });
        break;
    default:
        return 19;
}
return 0;
