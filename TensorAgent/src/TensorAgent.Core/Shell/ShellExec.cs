// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Core.Shell;

/// <summary>The interpreter proper: walks the AST, expands words, applies redirections, dispatches commands.</summary>
internal sealed class ShellExec
{
    private const int MaxDepth = 200;
    private int _depth;
    private int _conditionDepth;
    private int _loopDepth;
    private int _lastSubstitutionStatus;
    private readonly Random _random = new();

    public ShellExec(InProcessShell shell, ShellContext context, ShellState state, ConfinedPaths paths, CancellationToken cancellationToken, ILogger logger)
    {
        Shell = shell;
        Context = context;
        State = state;
        Paths = paths;
        Cancellation = cancellationToken;
        Logger = logger;
    }

    public InProcessShell Shell { get; }
    public ShellContext Context { get; }
    public ShellState State { get; private set; }
    public ConfinedPaths Paths { get; }
    public CancellationToken Cancellation { get; }
    public ILogger Logger { get; }
    public ExecutionPolicy Policy => Context.Policy;

    public void CheckCancel() => Cancellation.ThrowIfCancellationRequested();

    // --- paths, for builtins -----------------------------------------------------------------

    public string ResolveRead(string path) => Paths.Resolve(path, State.Cwd, PathAccess.Read);
    public string ResolveWrite(string path) => Paths.Resolve(path, State.Cwd, PathAccess.Write);
    public string Absolute(string path) => Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(State.Cwd, path));
    public bool CanRead(string absoluteDirectory) => Paths.IsAllowed(ConfinedPaths.RealPath(absoluteDirectory), PathAccess.Read);

    // --- entry points ---------------------------------------------------------------------------

    public int RunArgv(IReadOnlyList<string> argv, ShellStreams io)
    {
        var cmd = new SimpleCommand();
        foreach (string arg in argv)
            cmd.Words.Add(Word.Literal(arg));
        return RunPipelineCommands(new Pipeline { Commands = { cmd } }, io);
    }

    /// <summary>Parse and run text in THIS shell (source, eval).</summary>
    public int RunText(string text, ShellStreams io)
    {
        CommandList list;
        try
        {
            list = ShellParser.Parse(text);
        }
        catch (ShellSyntaxException ex)
        {
            io.Error("sh", ex.Message);
            return 2;
        }
        return RunList(list, io);
    }

    /// <summary>Run text in a copy of this shell: its cwd and variables do not come back.</summary>
    public int RunTextInSubshell(string text, ShellStreams io, IReadOnlyList<string>? positional = null, string? arg0 = null)
    {
        ShellState saved = State;
        State = saved.Clone();
        if (positional != null)
            State.Positional = new List<string>(positional);
        if (arg0 != null)
            State.Arg0 = arg0;
        try
        {
            return RunText(text, io);
        }
        catch (ShellExitException ex)
        {
            return ex.Code;
        }
        finally
        {
            State = saved;
        }
    }

    public int RunList(CommandList list, ShellStreams io)
    {
        int status = 0;
        foreach (AndOrList andOr in list.Items)
        {
            CheckCancel();
            status = RunAndOr(andOr, io);
        }
        return status;
    }

    private int RunAndOr(AndOrList list, ShellStreams io)
    {
        int status = RunPipeline(list.Pipelines[0], io, errexit: list.Ops.Count == 0);
        for (int i = 0; i < list.Ops.Count; i++)
        {
            bool last = i == list.Ops.Count - 1;
            if (list.Ops[i] == AndOrOp.And)
            {
                if (status != 0)
                    continue;
            }
            else if (status == 0)
                continue;
            status = RunPipeline(list.Pipelines[i + 1], io, errexit: last);
        }
        State.LastStatus = status;
        return status;
    }

    private int RunPipeline(Pipeline pipeline, ShellStreams io, bool errexit)
    {
        int status = RunPipelineCommands(pipeline, io);
        if (pipeline.Negate)
            status = status == 0 ? 1 : 0;
        State.LastStatus = status;
        if (status != 0 && errexit && State.ErrExit && _conditionDepth == 0 && !pipeline.Negate)
            throw new ShellExitException(status);
        return status;
    }

    private int RunPipelineCommands(Pipeline pipeline, ShellStreams io)
    {
        if (pipeline.Commands.Count == 1)
            return RunCommand(pipeline.Commands[0], io);

        Stream input = io.In;
        var statuses = new List<int>(pipeline.Commands.Count);
        for (int i = 0; i < pipeline.Commands.Count; i++)
        {
            CheckCancel();
            bool last = i == pipeline.Commands.Count - 1;
            MemoryStream? buffer = last ? null : new MemoryStream();
            ShellWriter output = last ? io.Out : new StreamShellWriter(buffer!, owns: false);
            ShellStreams sub = io.With(input: input, output: output);
            int status;
            try
            {
                status = RunCommand(pipeline.Commands[i], sub);
            }
            catch (ShellExitException ex)
            {
                // Each element of a pipeline is its own subshell: `exit` ends only it.
                status = ex.Code;
            }
            statuses.Add(status);
            if (!last)
            {
                buffer!.Position = 0;
                input = buffer;
            }
        }

        if (State.PipeFail)
        {
            for (int i = statuses.Count - 1; i >= 0; i--)
            {
                if (statuses[i] != 0)
                    return statuses[i];
            }
            return 0;
        }
        return statuses[^1];
    }

    // --- commands -------------------------------------------------------------------------------

    public int RunCommand(Command command, ShellStreams io)
    {
        CheckCancel();
        if (++_depth > MaxDepth)
        {
            _depth--;
            throw new ShellFatalException("maximum nesting level exceeded (recursion?)", 1);
        }
        var opened = new List<IDisposable>();
        try
        {
            ShellStreams streams;
            try
            {
                streams = command.Redirects.Count == 0 ? io : ApplyRedirects(command.Redirects, io, opened);
            }
            catch (ShellCommandException ex)
            {
                io.Error("sh", ex.Message);
                return ex.Code;
            }

            switch (command)
            {
                case SimpleCommand simple:
                    return RunSimple(simple, streams);
                case SubshellCommand subshell:
                    return RunSubshell(subshell.Body, streams);
                case GroupCommand group:
                    return RunList(group.Body, streams);
                case IfCommand ifc:
                    return RunIf(ifc, streams);
                case ForCommand forc:
                    return RunFor(forc, streams);
                case ArithForCommand aforc:
                    return RunArithFor(aforc, streams);
                case WhileCommand whilec:
                    return RunWhile(whilec, streams);
                case CaseCommand casec:
                    return RunCase(casec, streams);
                case FunctionCommand func:
                    State.Functions[func.Name] = func.Body;
                    return 0;
                case ConditionalCommand cond:
                    return RunConditional(cond, streams);
                case ArithmeticCommand arith:
                    return RunArithmeticCommand(arith, streams);
                default:
                    throw new InvalidOperationException("unknown command node");
            }
        }
        catch (ShellCommandException ex)
        {
            io.Error("sh", ex.Message);
            return ex.Code;
        }
        finally
        {
            _depth--;
            foreach (IDisposable d in opened)
            {
                try
                {
                    d.Dispose();
                }
                catch (IOException)
                {
                }
            }
        }
    }

    private int RunSubshell(CommandList body, ShellStreams io)
    {
        ShellState saved = State;
        State = saved.Clone();
        try
        {
            return RunList(body, io);
        }
        catch (ShellExitException ex)
        {
            return ex.Code;
        }
        finally
        {
            State = saved;
        }
    }

    private int RunIf(IfCommand cmd, ShellStreams io)
    {
        foreach ((CommandList condition, CommandList body) in cmd.Branches)
        {
            int status = RunCondition(condition, io);
            if (status == 0)
                return RunList(body, io);
        }
        return cmd.Else != null ? RunList(cmd.Else, io) : 0;
    }

    private int RunCondition(CommandList condition, ShellStreams io)
    {
        _conditionDepth++;
        try
        {
            return RunList(condition, io);
        }
        finally
        {
            _conditionDepth--;
        }
    }

    private int RunFor(ForCommand cmd, ShellStreams io)
    {
        List<string> items = cmd.Words == null ? new List<string>(State.Positional) : ExpandWords(cmd.Words);
        int status = 0;
        _loopDepth++;
        try
        {
            foreach (string item in items)
            {
                CheckCancel();
                State.Set(cmd.Variable, item);
                try
                {
                    status = RunList(cmd.Body, io);
                }
                catch (ShellBreakException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                    break;
                }
                catch (ShellContinueException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                }
            }
        }
        finally
        {
            _loopDepth--;
        }
        return status;
    }

    private int RunArithFor(ArithForCommand cmd, ShellStreams io)
    {
        int status = 0;
        _loopDepth++;
        try
        {
            if (cmd.Init.Length > 0)
                Arithmetic(cmd.Init);
            while (true)
            {
                CheckCancel();
                if (cmd.Condition.Length > 0 && Arithmetic(cmd.Condition) == 0)
                    break;
                try
                {
                    status = RunList(cmd.Body, io);
                }
                catch (ShellBreakException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                    break;
                }
                catch (ShellContinueException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                }
                if (cmd.Step.Length > 0)
                    Arithmetic(cmd.Step);
            }
        }
        finally
        {
            _loopDepth--;
        }
        return status;
    }

    private int RunWhile(WhileCommand cmd, ShellStreams io)
    {
        int status = 0;
        _loopDepth++;
        try
        {
            while (true)
            {
                CheckCancel();
                int cond = RunCondition(cmd.Condition, io);
                if ((cond == 0) == cmd.Until)
                    break;
                try
                {
                    status = RunList(cmd.Body, io);
                }
                catch (ShellBreakException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                    break;
                }
                catch (ShellContinueException ex)
                {
                    if (ex.Levels > 1)
                    {
                        ex.Levels--;
                        throw;
                    }
                }
            }
        }
        finally
        {
            _loopDepth--;
        }
        return status;
    }

    private int RunCase(CaseCommand cmd, ShellStreams io)
    {
        string subject = ExpandSingle(cmd.Subject);
        foreach ((List<Word> patterns, CommandList body) in cmd.Items)
        {
            foreach (Word pattern in patterns)
            {
                if (GlobMatcher.IsMatch(ExpandPattern(pattern), subject))
                    return RunList(body, io);
            }
        }
        return 0;
    }

    private int RunConditional(ConditionalCommand cmd, ShellStreams io)
    {
        try
        {
            return ShellTest.EvaluateConditional(this, cmd.Words) ? 0 : 1;
        }
        catch (ShellCommandException ex)
        {
            io.Error("sh", "[[: " + ex.Message);
            return 2;
        }
    }

    private int RunArithmeticCommand(ArithmeticCommand cmd, ShellStreams io)
    {
        try
        {
            return Arithmetic(cmd.Expression) != 0 ? 0 : 1;
        }
        catch (ShellArithmeticException ex)
        {
            io.Error("sh", cmd.Expression.Trim() + ": " + ex.Message);
            return 1;
        }
    }

    public long Arithmetic(string expression)
        => ShellArithmetic.Evaluate(expression, GetVariableForArithmetic, (n, v) => SetVariableForArithmetic(n, v));

    private string GetVariableForArithmetic(string name)
    {
        int bracket = name.IndexOf('[');
        if (bracket > 0 && name.EndsWith(']'))
        {
            string array = name.Substring(0, bracket);
            if (State.Arrays.TryGetValue(array, out List<string>? items) && int.TryParse(name.AsSpan(bracket + 1, name.Length - bracket - 2), out int index) && index >= 0 && index < items.Count)
                return items[index];
            return string.Empty;
        }
        return name switch
        {
            "RANDOM" => _random.Next(32768).ToString(CultureInfo.InvariantCulture),
            _ => State.Get(name),
        };
    }

    private void SetVariableForArithmetic(string name, string value)
    {
        int bracket = name.IndexOf('[');
        if (bracket > 0 && name.EndsWith(']'))
        {
            string array = name.Substring(0, bracket);
            if (!State.Arrays.TryGetValue(array, out List<string>? items))
                State.Arrays[array] = items = new List<string>();
            if (int.TryParse(name.AsSpan(bracket + 1, name.Length - bracket - 2), out int index) && index >= 0 && index < 100000)
            {
                while (items.Count <= index)
                    items.Add(string.Empty);
                items[index] = value;
            }
            return;
        }
        State.Set(name, value);
    }

    // --- simple commands --------------------------------------------------------------------------

    private int RunSimple(SimpleCommand cmd, ShellStreams io)
    {
        _lastSubstitutionStatus = 0;
        List<string> words;
        try
        {
            words = ExpandWords(cmd.Words);
        }
        catch (ShellArithmeticException ex)
        {
            io.Error("sh", ex.Message);
            return 1;
        }

        if (words.Count == 0)
        {
            foreach (Assignment assignment in cmd.Assignments)
                Assign(assignment, temporary: false);
            return cmd.Assignments.Count > 0 ? 0 : _lastSubstitutionStatus;
        }

        if (State.XTrace)
        {
            var trace = new StringBuilder("+ ");
            foreach (Assignment a in cmd.Assignments)
                trace.Append(a.Name).Append('=').Append(a.Value?.Source ?? string.Empty).Append(' ');
            trace.Append(string.Join(' ', words.Select(TraceQuote)));
            io.Err.WriteLine(trace.ToString());
        }

        var saved = new List<(string Name, bool Existed, string? Value, bool WasExported)>();
        foreach (Assignment assignment in cmd.Assignments)
        {
            saved.Add((assignment.Name, State.Vars.TryGetValue(assignment.Name, out string? old), old, State.Exported.Contains(assignment.Name)));
            Assign(assignment, temporary: true);
        }
        try
        {
            return Dispatch(words, io);
        }
        finally
        {
            for (int i = saved.Count - 1; i >= 0; i--)
            {
                (string name, bool existed, string? value, bool wasExported) = saved[i];
                if (existed)
                    State.Vars[name] = value!;
                else
                    State.Vars.Remove(name);
                if (!wasExported)
                    State.Exported.Remove(name);
            }
        }
    }

    private static string TraceQuote(string word)
        => word.Length == 0 || word.Any(c => char.IsWhiteSpace(c) || c is '\'' or '"' or '$' or '`' or '\\' or '|' or '&' or ';' or '<' or '>' or '(' or ')' or '*' or '?' or '[')
            ? "'" + word.Replace("'", "'\\''") + "'"
            : word;

    private void Assign(Assignment assignment, bool temporary)
    {
        if (assignment.ArrayValue != null)
        {
            List<string> items = ExpandWords(assignment.ArrayValue);
            if (assignment.Append && State.Arrays.TryGetValue(assignment.Name, out List<string>? existing))
                existing.AddRange(items);
            else
                State.Arrays[assignment.Name] = items;
            State.Set(assignment.Name, items.Count > 0 ? items[0] : string.Empty);
            return;
        }

        string value = assignment.Value == null ? string.Empty : ExpandSingle(assignment.Value);
        if (assignment.Subscript != null)
        {
            if (!State.Arrays.TryGetValue(assignment.Name, out List<string>? array))
                State.Arrays[assignment.Name] = array = new List<string>();
            int index = (int)Arithmetic(assignment.Subscript);
            if (index < 0 || index > 100000)
                throw new ShellCommandException($"{assignment.Name}[{assignment.Subscript}]: bad array subscript");
            while (array.Count <= index)
                array.Add(string.Empty);
            array[index] = assignment.Append ? array[index] + value : value;
            return;
        }

        if (assignment.Append)
            value = State.Get(assignment.Name) + value;
        State.Set(assignment.Name, value);
        if (temporary)
            State.Exported.Add(assignment.Name);
    }

    private int Dispatch(List<string> words, ShellStreams io)
    {
        string name = words[0];
        string[] argv = words.ToArray();

        if (State.Functions.TryGetValue(name, out Command? body))
            return CallFunction(name, body, argv, io);

        if (ShellBuiltins.Table.TryGetValue(name, out ShellBuiltin? builtin))
            return Invoke(builtin, name, argv, io);

        if (name.Contains('/'))
        {
            string baseName = Path.GetFileName(name);
            if (IsSystemBinDirectory(Path.GetDirectoryName(name)) && ShellBuiltins.Table.TryGetValue(baseName, out builtin))
            {
                argv[0] = baseName;
                return Invoke(builtin, baseName, argv, io);
            }
            return RunScriptFile(name, argv, io);
        }

        // Not just "not found": what to use instead. A dead end here is where a model
        // stops using the shell and starts inventing the answer -- see
        // ShellMissingCommand for the case this was written from.
        io.Error("sh", ShellMissingCommand.Describe(
            name,
            ShellBuiltins.Table.Keys,
            Context.Python is { IsAvailable: true },
            Context.JavaScript is { IsAvailable: true }));
        return ExecutionResult.CommandNotFoundExitCode;
    }

    private static bool IsSystemBinDirectory(string? directory)
        => directory is "/bin" or "/usr/bin" or "/usr/local/bin" or "/opt/homebrew/bin" or "/sbin" or "/usr/sbin" or "/opt/local/bin";

    private int Invoke(ShellBuiltin builtin, string name, string[] argv, ShellStreams io)
    {
        try
        {
            return builtin(this, argv, io);
        }
        catch (ConfinementException ex)
        {
            io.Error(name, ex.Message);
            return 1;
        }
        catch (ShellUsageException ex)
        {
            io.Error(name, ex.Message);
            return ex.Code;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException or FormatException or RegexMatchTimeoutException)
        {
            io.Error(name, ex.Message);
            return 1;
        }
    }

    /// <summary>`./script.sh`, `./tool.py`, `scripts/run.js`: run by extension or shebang.</summary>
    private int RunScriptFile(string name, string[] argv, ShellStreams io)
    {
        string path;
        try
        {
            path = ResolveRead(name);
        }
        catch (ConfinementException ex)
        {
            io.Error("sh", ex.Message);
            return 126;
        }
        if (!File.Exists(path))
        {
            io.Error("sh", $"{name}: No such file or directory");
            return ExecutionResult.CommandNotFoundExitCode;
        }

        string extension = Path.GetExtension(path).ToLowerInvariant();
        string firstLine;
        try
        {
            using var reader = new StreamReader(path, ShellText.Utf8, detectEncodingFromByteOrderMarks: true);
            firstLine = reader.ReadLine() ?? string.Empty;
        }
        catch (IOException ex)
        {
            io.Error("sh", $"{name}: {ex.Message}");
            return 126;
        }

        string interpreter = extension switch
        {
            ".py" => "python3",
            ".js" or ".mjs" or ".cjs" => "node",
            ".sh" or ".bash" => "sh",
            _ => string.Empty,
        };
        if (interpreter.Length == 0 && firstLine.StartsWith("#!", StringComparison.Ordinal))
        {
            string shebang = firstLine.Substring(2).Trim();
            string[] parts = shebang.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            string program = parts.Length > 0 ? Path.GetFileName(parts[0]) : string.Empty;
            if (program == "env" && parts.Length > 1)
                program = parts[1];
            if (program.StartsWith("python", StringComparison.Ordinal))
                interpreter = "python3";
            else if (program is "node" or "nodejs")
                interpreter = "node";
            else if (program is "sh" or "bash" or "zsh" or "dash")
                interpreter = "sh";
        }
        if (interpreter.Length == 0)
        {
            if (ShellText.LooksBinary(File.ReadAllBytes(path).AsSpan()))
            {
                io.Error("sh", $"{name}: cannot execute binary file");
                return 126;
            }
            interpreter = "sh";
        }

        var forwarded = new string[argv.Length + 1];
        forwarded[0] = interpreter;
        forwarded[1] = name;
        Array.Copy(argv, 1, forwarded, 2, argv.Length - 1);
        return Invoke(ShellBuiltins.Table[interpreter], interpreter, forwarded, io);
    }

    private int CallFunction(string name, Command body, string[] argv, ShellStreams io)
    {
        List<string> savedPositional = State.Positional;
        string savedArg0 = State.Arg0;
        State.Positional = argv.Skip(1).ToList();
        State.LocalScopes.Push(new Dictionary<string, (bool, string?)>(StringComparer.Ordinal));
        try
        {
            return RunCommand(body, io);
        }
        catch (ShellReturnException ex)
        {
            return ex.Code;
        }
        finally
        {
            Dictionary<string, (bool Existed, string? Value)> scope = State.LocalScopes.Pop();
            foreach (KeyValuePair<string, (bool Existed, string? Value)> entry in scope)
            {
                if (entry.Value.Existed)
                    State.Vars[entry.Key] = entry.Value.Value!;
                else
                    State.Vars.Remove(entry.Key);
            }
            State.Positional = savedPositional;
            State.Arg0 = savedArg0;
        }
    }

    public bool InLoop => _loopDepth > 0;

    // --- redirections ------------------------------------------------------------------------------

    private ShellStreams ApplyRedirects(List<Redirect> redirects, ShellStreams io, List<IDisposable> opened)
    {
        Stream input = io.In;
        ShellWriter output = io.Out;
        ShellWriter error = io.Err;

        foreach (Redirect r in redirects)
        {
            switch (r.Kind)
            {
                case RedirectKind.Output:
                case RedirectKind.Clobber:
                case RedirectKind.Append:
                case RedirectKind.OutputBoth:
                case RedirectKind.AppendBoth:
                {
                    string target = ExpandSingle(r.Target!);
                    bool append = r.Kind is RedirectKind.Append or RedirectKind.AppendBoth;
                    ShellWriter writer = OpenOutput(target, append, output, error, opened);
                    if (r.Kind is RedirectKind.OutputBoth or RedirectKind.AppendBoth)
                    {
                        output = writer;
                        error = writer;
                    }
                    else if (r.Fd == 2)
                        error = writer;
                    else if (r.Fd == 1)
                        output = writer;
                    else
                        throw new ShellCommandException($"{r.Fd}: file descriptors above 2 are not supported by this host");
                    break;
                }
                case RedirectKind.Input:
                {
                    string target = ExpandSingle(r.Target!);
                    if (ConfinedPaths.IsDevNull(target))
                    {
                        input = Stream.Null;
                        break;
                    }
                    string path;
                    try
                    {
                        path = ResolveRead(target);
                    }
                    catch (ConfinementException ex)
                    {
                        throw new ShellCommandException(ex.Message);
                    }
                    if (!File.Exists(path))
                        throw new ShellCommandException($"{target}: No such file or directory");
                    var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                    opened.Add(fs);
                    input = fs;
                    break;
                }
                case RedirectKind.DupOutput:
                {
                    ShellWriter source = r.DupFd switch
                    {
                        1 => output,
                        2 => error,
                        _ => throw new ShellCommandException($"{r.DupFd}: bad file descriptor"),
                    };
                    if (r.Fd == 2)
                        error = source;
                    else if (r.Fd == 1)
                        output = source;
                    else
                        throw new ShellCommandException($"{r.Fd}: file descriptors above 2 are not supported by this host");
                    break;
                }
                case RedirectKind.DupInput:
                    if (r.DupFd != 0)
                        throw new ShellCommandException($"{r.DupFd}: bad file descriptor");
                    break;
                case RedirectKind.Close:
                    if (r.Fd == 2)
                        error = NullShellWriter.Instance;
                    else if (r.Fd == 1)
                        output = NullShellWriter.Instance;
                    else
                        input = Stream.Null;
                    break;
                case RedirectKind.HereDoc:
                {
                    string body = r.HereDocBody == null ? string.Empty : ExpandSingle(r.HereDocBody);
                    input = ShellText.FromString(body);
                    break;
                }
                case RedirectKind.HereString:
                    input = ShellText.FromString(ExpandSingle(r.Target!) + "\n");
                    break;
            }
        }

        return new ShellStreams(input, output, error);
    }

    private ShellWriter OpenOutput(string target, bool append, ShellWriter currentOut, ShellWriter currentErr, List<IDisposable> opened)
    {
        if (ConfinedPaths.IsDevNull(target))
            return NullShellWriter.Instance;
        if (target == "/dev/stdout")
            return currentOut;
        if (target == "/dev/stderr")
            return currentErr;

        string path;
        try
        {
            path = ResolveWrite(target);
        }
        catch (ConfinementException ex)
        {
            throw new ShellCommandException(ex.Message);
        }
        if (Directory.Exists(path))
            throw new ShellCommandException($"{target}: Is a directory");
        string? parent = Path.GetDirectoryName(path);
        if (parent != null && !Directory.Exists(parent))
            throw new ShellCommandException($"{target}: No such file or directory");
        try
        {
            var fs = new FileStream(path, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
            var writer = new StreamShellWriter(fs, owns: true);
            opened.Add(writer);
            return writer;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ShellCommandException($"{target}: {ex.Message}");
        }
    }

    // --- expansion -------------------------------------------------------------------------------------

    private sealed class Field
    {
        public StringBuilder Text { get; } = new();
        public StringBuilder Quoted { get; } = new(); // one char per Text char: 'q' or 'u'
        public bool HasQuotedPart { get; set; }

        public void Append(string text, bool quoted)
        {
            Text.Append(text);
            Quoted.Append(quoted ? 'q' : 'u', text.Length);
            if (quoted)
                HasQuotedPart = true;
        }
    }

    public List<string> ExpandWords(IEnumerable<Word> words)
    {
        var result = new List<string>();
        foreach (Word word in words)
            result.AddRange(ExpandWord(word, splitAndGlob: true));
        return result;
    }

    /// <summary>One string, no field splitting, no globbing (assignments, redirect targets, here-documents).</summary>
    public string ExpandSingle(Word word)
    {
        List<string> fields = ExpandWord(word, splitAndGlob: false);
        return fields.Count == 0 ? string.Empty : string.Join(' ', fields);
    }

    /// <summary>A glob pattern with the quoted characters escaped, for `case` and `[[ == ]]`.</summary>
    public string ExpandPattern(Word word)
    {
        List<Field> fields = ExpandFields(word);
        var sb = new StringBuilder();
        foreach (Field field in fields)
        {
            for (int i = 0; i < field.Text.Length; i++)
            {
                char c = field.Text[i];
                if (field.Quoted[i] == 'q' && c is '*' or '?' or '[' or ']' or '\\')
                    sb.Append('\\');
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    private List<string> ExpandWord(Word word, bool splitAndGlob)
    {
        List<Field> fields = ExpandFields(word);
        var result = new List<string>(fields.Count);
        foreach (Field field in fields)
        {
            if (!splitAndGlob || State.NoGlob)
            {
                result.Add(field.Text.ToString());
                continue;
            }
            string text = field.Text.ToString();
            bool hasUnquotedGlob = false;
            for (int i = 0; i < text.Length; i++)
            {
                if (field.Quoted[i] == 'u' && text[i] is '*' or '?' or '[')
                {
                    hasUnquotedGlob = true;
                    break;
                }
            }
            if (!hasUnquotedGlob)
            {
                result.Add(text);
                continue;
            }
            var pattern = new StringBuilder(text.Length + 8);
            for (int i = 0; i < text.Length; i++)
            {
                if (field.Quoted[i] == 'q' && text[i] is '*' or '?' or '[' or ']' or '\\')
                    pattern.Append('\\');
                pattern.Append(text[i]);
            }
            List<string> matches = GlobMatcher.ExpandPath(pattern.ToString(), State.Cwd, CanRead);
            if (matches.Count == 0)
                result.Add(text);
            else
                result.AddRange(matches);
        }
        return result;
    }

    private List<Field> ExpandFields(Word word)
    {
        var fields = new List<Field> { new() };
        bool onlyUnquotedExpansion = true; // a word made only of unquoted expansions that produce nothing vanishes
        string ifs = State.IsSet("IFS") ? State.Get("IFS") : " \t\n";

        void AppendSplit(string value, bool quoted)
        {
            if (quoted)
            {
                fields[^1].Append(value, quoted: true);
                return;
            }
            if (value.Length == 0)
                return;
            if (ifs.Length == 0)
            {
                fields[^1].Append(value, quoted: false);
                return;
            }
            // Field splitting: leading/trailing IFS whitespace is dropped, runs of it
            // separate fields, and non-whitespace IFS characters separate on each.
            string ifsWhite = new(ifs.Where(char.IsWhiteSpace).ToArray());
            int i = 0;
            bool first = true;
            while (i < value.Length)
            {
                // skip whitespace separators
                int start = i;
                while (i < value.Length && ifsWhite.IndexOf(value[i]) >= 0)
                    i++;
                if (i > start && !first)
                    fields.Add(new Field());
                if (i > start && first && fields[^1].Text.Length > 0)
                    fields.Add(new Field());
                if (i < value.Length && ifs.IndexOf(value[i]) >= 0 && ifsWhite.IndexOf(value[i]) < 0)
                {
                    // non-whitespace separator
                    if (!first || fields[^1].Text.Length > 0 || i > start)
                        fields.Add(new Field());
                    i++;
                    first = false;
                    continue;
                }
                int wordStart = i;
                while (i < value.Length && ifs.IndexOf(value[i]) < 0)
                    i++;
                if (i > wordStart)
                {
                    fields[^1].Append(value.Substring(wordStart, i - wordStart), quoted: false);
                    first = false;
                }
            }
            if (value.Length > 0 && ifsWhite.IndexOf(value[^1]) >= 0 && fields[^1].Text.Length > 0)
                fields.Add(new Field());
        }

        foreach (WordPart part in word.Parts)
        {
            switch (part)
            {
                case LiteralPart lit:
                    onlyUnquotedExpansion = false;
                    fields[^1].Append(lit.Text, lit.Quoted);
                    break;
                case TildePart tilde:
                    onlyUnquotedExpansion = false;
                    fields[^1].Append(tilde.User.Length == 0 ? State.Get("HOME") : "~" + tilde.User, quoted: true);
                    break;
                case ArithPart arith:
                    onlyUnquotedExpansion = false;
                    fields[^1].Append(Arithmetic(ExpandArithmeticText(arith.Expression)).ToString(CultureInfo.InvariantCulture), quoted: true);
                    break;
                case CommandSubstPart subst:
                {
                    string output = CaptureOutput(subst.Body).TrimEnd('\n');
                    if (subst.Quoted)
                        onlyUnquotedExpansion = false;
                    AppendSplit(output, subst.Quoted);
                    break;
                }
                case ParamPart param:
                {
                    if (param.Quoted)
                        onlyUnquotedExpansion = false;
                    List<string>? multiple = ExpandParamMultiple(param);
                    if (multiple != null)
                    {
                        // "$@" / "${a[@]}": one field per element, joined to neighbours at the ends.
                        if (multiple.Count == 0)
                        {
                            if (param.InDoubleQuotes)
                                onlyUnquotedExpansion = true;
                            break;
                        }
                        for (int i = 0; i < multiple.Count; i++)
                        {
                            if (i > 0)
                                fields.Add(new Field());
                            if (param.InDoubleQuotes)
                                fields[^1].Append(multiple[i], quoted: true);
                            else
                                AppendSplit(multiple[i], quoted: false);
                        }
                        break;
                    }
                    AppendSplit(ExpandParam(param), param.Quoted);
                    break;
                }
            }
        }

        // Drop empty fields that came from nothing but unquoted expansions; keep a
        // field that any quoted text touched ("" is an argument, $empty is not).
        var kept = new List<Field>(fields.Count);
        foreach (Field field in fields)
        {
            if (field.Text.Length == 0 && !field.HasQuotedPart && (onlyUnquotedExpansion || fields.Count > 1))
                continue;
            kept.Add(field);
        }
        if (kept.Count == 0 && !onlyUnquotedExpansion && word.Parts.Count > 0 && fields.Any(f => f.HasQuotedPart))
            kept.Add(new Field());
        return kept;
    }

    /// <summary>Variables and positional parameters inside `$(( ))` expand before evaluation.</summary>
    private string ExpandArithmeticText(string expression)
    {
        if (expression.IndexOf('$') < 0)
            return expression;
        var sb = new StringBuilder(expression.Length);
        for (int i = 0; i < expression.Length; i++)
        {
            char c = expression[i];
            if (c == '$' && i + 1 < expression.Length)
            {
                if (expression[i + 1] == '{')
                {
                    int close = expression.IndexOf('}', i);
                    if (close > 0)
                    {
                        string inner = expression.Substring(i + 2, close - i - 2);
                        sb.Append(ArithmeticParameterValue(inner));
                        i = close;
                        continue;
                    }
                }
                else if (IsSpecialName(expression[i + 1].ToString()))
                {
                    // Unbraced positional parameters consume one digit: $10 is
                    // ${1} followed by 0, while ${10} names the tenth argument.
                    sb.Append(ArithmeticParameterValue(expression[++i].ToString()));
                    continue;
                }
                else if (char.IsAsciiLetter(expression[i + 1]) || expression[i + 1] == '_')
                {
                    int j = i + 1;
                    while (j < expression.Length && (char.IsAsciiLetterOrDigit(expression[j]) || expression[j] == '_'))
                        j++;
                    string name = expression.Substring(i + 1, j - i - 1);
                    string value = GetVariableForArithmetic(name);
                    sb.Append(value.Length == 0 ? "0" : value);
                    i = j - 1;
                    continue;
                }
                else if (expression[i + 1] == '(')
                {
                    // $(cmd) inside arithmetic
                    int depth = 0;
                    int j = i + 1;
                    for (; j < expression.Length; j++)
                    {
                        if (expression[j] == '(')
                            depth++;
                        else if (expression[j] == ')' && --depth == 0)
                            break;
                    }
                    string inner = expression.Substring(i + 2, j - i - 2);
                    string output = CaptureOutput(ShellParser.Parse(inner)).Trim();
                    sb.Append(output.Length == 0 ? "0" : output);
                    i = j;
                    continue;
                }
            }
            sb.Append(c);
        }
        return sb.ToString();
    }

    private string ArithmeticParameterValue(string name)
    {
        string value = name.Length > 0 && (IsSpecialName(name) || name.All(char.IsAsciiDigit))
            ? SpecialOrVariable(name).Value
            : GetVariableForArithmetic(name);
        return value.Length == 0 ? "0" : value;
    }

    public string CaptureOutput(CommandList body)
    {
        var buffer = new MemoryStream();
        var writer = new StreamShellWriter(buffer, owns: false);
        var io = new ShellStreams(Stream.Null, writer, CurrentErr ?? NullShellWriter.Instance);
        ShellState saved = State;
        State = saved.Clone();
        int status;
        try
        {
            status = RunList(body, io);
        }
        catch (ShellExitException ex)
        {
            status = ex.Code;
        }
        finally
        {
            State = saved;
        }
        _lastSubstitutionStatus = status;
        State.LastStatus = status;
        return ShellText.Utf8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    /// <summary>The stderr a command substitution inherits; set by builtins that run nested scripts.</summary>
    internal ShellWriter? CurrentErr { get; set; }

    private List<string>? ExpandParamMultiple(ParamPart param)
    {
        if (param.Op.Length > 0 && param.Op != "#len")
            return null;
        if (param.Name is "@" or "*")
        {
            if (param.Op == "#len")
                return null;
            if (param.Name == "*" && param.InDoubleQuotes)
                return null;
            return new List<string>(State.Positional);
        }
        if (param.Subscript is "@" or "*")
        {
            if (param.Op == "#len")
                return null;
            if (param.Subscript == "*" && param.InDoubleQuotes)
                return null;
            return State.Arrays.TryGetValue(param.Name, out List<string>? items) ? new List<string>(items) : new List<string>();
        }
        return null;
    }

    private string ExpandParam(ParamPart param)
    {
        string name = param.Name;
        if (param.Op == "!")
        {
            string indirect = State.Get(name);
            return indirect.Length == 0 ? string.Empty : State.Get(indirect);
        }

        bool isSet;
        string value;
        if (param.Subscript != null)
        {
            if (param.Subscript is "@" or "*")
            {
                List<string> items = State.Arrays.TryGetValue(name, out List<string>? arr) ? arr : new List<string>();
                if (param.Op == "#len")
                    return items.Count.ToString(CultureInfo.InvariantCulture);
                string sep = State.Get("IFS");
                value = string.Join(sep.Length > 0 ? sep[0].ToString() : string.Empty, items);
                isSet = items.Count > 0;
            }
            else
            {
                int index = (int)Arithmetic(param.Subscript);
                isSet = State.Arrays.TryGetValue(name, out List<string>? arr) && index >= 0 && index < arr.Count;
                value = isSet ? State.Arrays[name][index] : string.Empty;
            }
        }
        else
        {
            (isSet, value) = SpecialOrVariable(name);
        }

        switch (param.Op)
        {
            case "":
                if (!isSet && State.NoUnset && !IsSpecialName(name))
                    throw new ShellCommandException($"{name}: unbound variable");
                return value;
            case "#len":
                if (name is "@" or "*")
                    return State.Positional.Count.ToString(CultureInfo.InvariantCulture);
                return value.Length.ToString(CultureInfo.InvariantCulture);
            case ":-":
                return value.Length == 0 ? ArgText(param.Arg) : value;
            case "-":
                return isSet ? value : ArgText(param.Arg);
            case ":=":
            case "=":
            {
                bool use = param.Op == ":=" ? value.Length == 0 : !isSet;
                if (!use)
                    return value;
                string fallback = ArgText(param.Arg);
                State.Set(name, fallback);
                return fallback;
            }
            case ":?":
            case "?":
            {
                bool fail = param.Op == ":?" ? value.Length == 0 : !isSet;
                if (!fail)
                    return value;
                string message = param.Arg == null || param.Arg.Parts.Count == 0 ? "parameter null or not set" : ArgText(param.Arg);
                throw new ShellFatalException($"{name}: {message}");
            }
            case ":+":
                return value.Length == 0 ? string.Empty : ArgText(param.Arg);
            case "+":
                return isSet ? ArgText(param.Arg) : string.Empty;
            case "#":
            case "##":
            case "%":
            case "%%":
                return TrimPattern(value, ExpandPattern(param.Arg!), param.Op);
            case "/":
            case "//":
                return ReplacePattern(value, ExpandPattern(param.Arg!), param.Arg2 == null ? string.Empty : ArgText(param.Arg2), param.Op == "//");
            case ":":
            {
                long offset = Arithmetic(ExpandArithmeticText(param.Arg!.Source));
                if (offset < 0)
                    offset = Math.Max(0, value.Length + offset);
                if (offset > value.Length)
                    return string.Empty;
                if (param.Arg2 == null)
                    return value.Substring((int)offset);
                long length = Arithmetic(ExpandArithmeticText(param.Arg2.Source));
                if (length < 0)
                {
                    long end = value.Length + length;
                    return end <= offset ? string.Empty : value.Substring((int)offset, (int)(end - offset));
                }
                return value.Substring((int)offset, (int)Math.Min(length, value.Length - offset));
            }
            case "^":
                return value.Length == 0 ? value : char.ToUpperInvariant(value[0]) + value.Substring(1);
            case "^^":
                return value.ToUpperInvariant();
            case ",":
                return value.Length == 0 ? value : char.ToLowerInvariant(value[0]) + value.Substring(1);
            case ",,":
                return value.ToLowerInvariant();
            default:
                throw new ShellCommandException($"${{{name}{param.Op}}}: bad substitution");
        }
    }

    private static bool IsSpecialName(string name)
        => name.Length == 1 && (char.IsAsciiDigit(name[0]) || name[0] is '@' or '*' or '#' or '?' or '$' or '!' or '-');

    private string ArgText(Word? arg) => arg == null ? string.Empty : ExpandSingle(arg);

    private (bool IsSet, string Value) SpecialOrVariable(string name)
    {
        switch (name)
        {
            case "?":
                return (true, State.LastStatus.ToString(CultureInfo.InvariantCulture));
            case "$":
                return (true, Environment.ProcessId.ToString(CultureInfo.InvariantCulture));
            case "!":
                return (true, string.Empty);
            case "#":
                return (true, State.Positional.Count.ToString(CultureInfo.InvariantCulture));
            case "@":
            case "*":
            {
                string sep = name == "*" && State.IsSet("IFS") && State.Get("IFS").Length > 0 ? State.Get("IFS")[0].ToString() : " ";
                return (true, string.Join(sep, State.Positional));
            }
            case "0":
                return (true, State.Arg0);
            case "-":
                return (true, (State.ErrExit ? "e" : string.Empty) + (State.XTrace ? "x" : string.Empty) + (State.NoUnset ? "u" : string.Empty));
            case "RANDOM":
                return (true, _random.Next(32768).ToString(CultureInfo.InvariantCulture));
            case "SECONDS":
                return (true, ((long)(DateTime.UtcNow - State.Started).TotalSeconds).ToString(CultureInfo.InvariantCulture));
            case "PWD":
                return (true, State.Cwd);
        }
        if (char.IsAsciiDigit(name[0]))
        {
            int index = int.Parse(name, CultureInfo.InvariantCulture);
            return index >= 1 && index <= State.Positional.Count ? (true, State.Positional[index - 1]) : (false, string.Empty);
        }
        if (State.Vars.TryGetValue(name, out string? value))
            return (true, value);
        if (State.Arrays.TryGetValue(name, out List<string>? items) && items.Count > 0)
            return (true, items[0]);
        return (false, string.Empty);
    }

    private static string TrimPattern(string value, string pattern, string op)
    {
        Regex regex = GlobMatcher.ToRegex(pattern, ignoreCase: false, matchSlash: true);
        switch (op)
        {
            case "#":
                for (int i = 0; i <= value.Length; i++)
                    if (regex.IsMatch(value.Substring(0, i)))
                        return value.Substring(i);
                return value;
            case "##":
                for (int i = value.Length; i >= 0; i--)
                    if (regex.IsMatch(value.Substring(0, i)))
                        return value.Substring(i);
                return value;
            case "%":
                for (int i = value.Length; i >= 0; i--)
                    if (regex.IsMatch(value.Substring(i)))
                        return value.Substring(0, i);
                return value;
            default:
                for (int i = 0; i <= value.Length; i++)
                    if (regex.IsMatch(value.Substring(i)))
                        return value.Substring(0, i);
                return value;
        }
    }

    private static string ReplacePattern(string value, string pattern, string replacement, bool all)
    {
        bool anchorStart = pattern.StartsWith('#');
        bool anchorEnd = pattern.StartsWith('%');
        if (anchorStart || anchorEnd)
            pattern = pattern.Substring(1);
        string text = GlobMatcher.ToRegexText(pattern, matchSlash: true);
        if (anchorStart)
            text = "^" + text;
        if (anchorEnd)
            text += "$";
        var regex = new Regex(text, RegexOptions.CultureInvariant | RegexOptions.Singleline);
        string escaped = replacement.Replace("$", "$$");
        return all ? regex.Replace(value, escaped) : regex.Replace(value, escaped, 1);
    }
}

/// <summary>A builtin's usage error: message on stderr, given exit status.</summary>
internal sealed class ShellUsageException : Exception
{
    public ShellUsageException(string message, int code = 1) : base(message) => Code = code;
    public int Code { get; }
}

internal delegate int ShellBuiltin(ShellExec exec, string[] argv, ShellStreams io);
