using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Shell;

namespace TensorAgent.Tests;

public sealed class InProcessShellTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-sh-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _temp;
    private readonly string _readable;
    private readonly InProcessShell _shell = new();

    public InProcessShellTests()
    {
        _work = Path.Combine(_root, "work");
        _temp = Path.Combine(_root, "tmp");
        _readable = Path.Combine(_root, "skill");
        Directory.CreateDirectory(_work);
        Directory.CreateDirectory(_temp);
        Directory.CreateDirectory(_readable);
    }

    public void Dispose() { try { Directory.Delete(_root, true); } catch { } }

    private ExecutionPolicy Policy(bool network = false, bool scripts = true) => new(
        AllowScripts: scripts,
        AllowNetwork: network,
        WorkRoot: _work,
        ReadableRoots: new[] { _readable },
        TempRoot: _temp)
    { DefaultTimeout = TimeSpan.FromSeconds(20) };

    private ShellContext Context(ExecutionPolicy? policy = null, string? stdin = null, IReadOnlyDictionary<string, string>? env = null)
        => new()
        {
            WorkingDirectory = _work,
            Policy = policy ?? Policy(),
            StandardInput = stdin,
            Environment = env ?? new Dictionary<string, string>(),
        };

    private ExecutionResult Run(string command, ShellContext? context = null)
        => _shell.RunCommandAsync(command, context ?? Context(), CancellationToken.None).GetAwaiter().GetResult();

    private string Out(string command, ShellContext? context = null)
    {
        ExecutionResult result = Run(command, context);
        Assert.True(result.ExitCode == 0, $"`{command}` exited {result.ExitCode}: {result.Stderr}");
        return result.Stdout;
    }

    // ---- a command that is not here ---------------------------------------------------

    /// <summary>
    /// A missing command has to leave the model a move to make.
    ///
    /// <para>
    /// "bc: command not found" is true and useless, and useless is where a model stops
    /// using the shell. Observed on a phone: asked for the days between two dates it
    /// reached for the shell correctly, got one date's epoch seconds out of it, reached
    /// for <c>bc</c> to subtract, was told 127 — and finished the arithmetic in its own
    /// head, answering 164 days where the answer is 20,863, with a formula underneath to
    /// make it look checked. Nothing in the output suggested a next step, so it invented
    /// one.
    /// </para>
    /// </summary>
    [Fact]
    public void AMissingCommandSaysWhatToUseInstead()
    {
        ExecutionResult result = Run("bc <<< '1+1'");
        Assert.Equal(ExecutionResult.CommandNotFoundExitCode, result.ExitCode);
        Assert.Contains("bc: command not found", result.Stderr, StringComparison.Ordinal);
        // The two things that actually work here, named.
        Assert.Contains("$(( ", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("python3", result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingPackageManagerSaysWhatCanAndCannotBeInstalled()
    {
        // The model is in the middle of trying to fix a missing tool; "not found" is
        // the least useful sentence available at that moment.
        string stderr = Run("apt-get install bc").Stderr;
        Assert.Contains("no system package manager", stderr, StringComparison.Ordinal);
        Assert.Contains("no native program can be installed", stderr, StringComparison.Ordinal);
        // ...and the half that DOES work, which is the whole point of saying anything.
        Assert.Contains("pure-Python", stderr, StringComparison.Ordinal);
        Assert.Contains("none-any", stderr, StringComparison.Ordinal);
        Assert.Contains("pip install", stderr, StringComparison.Ordinal);
        Assert.Contains("python3 -m pip install", stderr, StringComparison.Ordinal);
        Assert.Contains("Dependencies are not resolved automatically", stderr, StringComparison.Ordinal);
        Assert.Contains("npm/JavaScript packages cannot be installed", stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("`npm install", stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingNpmDoesNotPromiseThatJavaScriptPackagesCanBeInstalled()
    {
        string stderr = Run("npm install left-pad").Stderr;

        Assert.Contains("npm: command not found", stderr, StringComparison.Ordinal);
        Assert.Contains("npm/JavaScript packages cannot be installed", stderr, StringComparison.Ordinal);
        Assert.Contains("Node's built-in modules", stderr, StringComparison.Ordinal);
        Assert.Contains("pure-Python", stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("npm install <name>", stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void AMistypedCommandIsRecognisedAsOne()
    {
        Assert.Contains("Did you mean `echo`?", Run("ehco hi").Stderr, StringComparison.Ordinal);
        Assert.Contains("Did you mean `grep`?", Run("gerp x f").Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void ACommandWithNoSpecificAdviceStillGetsTheGeneralFact()
    {
        string stderr = Run("some-tool-nobody-has --version").Stderr;
        Assert.Contains("command not found", stderr, StringComparison.Ordinal);
        // Nothing is invented as a near miss for a name unlike anything here.
        Assert.DoesNotContain("Did you mean", stderr, StringComparison.Ordinal);
    }

    // ---- date ------------------------------------------------------------------------

    /// <summary>
    /// <c>date -d</c> has to mean the date it was given.
    ///
    /// <para>
    /// It used to be declared as an option that takes a value, and the value was then
    /// never read: <c>date -d 1969-07-20 +%s</c> printed the epoch seconds of right now.
    /// A model asking for the days between two dates therefore subtracted two copies of
    /// today and reported the difference with confidence — 20699 days between
    /// 1969-07-20 and 2026-09-02, observed on a phone, where the answer is 20863. There
    /// was nothing in the output to suggest anything had gone wrong, which is the worst
    /// property a shell builtin can have.
    /// </para>
    /// </summary>
    [Fact]
    public void DateHonoursTheDateItWasGiven()
    {
        Assert.Equal("1969-07-20\n", Out("date -d 1969-07-20 +%F"));
        Assert.Equal("2026-09-02\n", Out("date --date=2026-09-02 +%F"));
        Assert.Equal("2026-09-02 13:45\n", Out("date -d '2026-09-02 13:45' '+%F %H:%M'"));

        // The arithmetic the model was actually doing, end to end through the shell.
        Assert.Equal("20863\n", Out(
            "echo $(( ( $(date -u -d 2026-09-02 +%s) - $(date -u -d 1969-07-20 +%s) ) / 86400 ))"));
    }

    [Fact]
    public void DateReadsEpochsWordsAndOffsets()
    {
        Assert.Equal("1970-01-01\n", Out("date -u -d @0 +%F"));
        Assert.Equal("2001-09-09\n", Out("date -u -d @1000000000 +%F"));

        string today = Out("date +%F").Trim();
        Assert.Equal(today, Out("date -d today +%F").Trim());
        Assert.Equal(DateTime.Parse(today).AddDays(-1).ToString("yyyy-MM-dd"), Out("date -d yesterday +%F").Trim());
        Assert.Equal(DateTime.Parse(today).AddDays(1).ToString("yyyy-MM-dd"), Out("date -d tomorrow +%F").Trim());
        Assert.Equal(DateTime.Parse(today).AddDays(-3).ToString("yyyy-MM-dd"), Out("date -d '3 days ago' +%F").Trim());
        Assert.Equal(DateTime.Parse(today).AddDays(14).ToString("yyyy-MM-dd"), Out("date -d '+2 weeks' +%F").Trim());
    }

    /// <summary>
    /// A date it cannot read is an error, not today. Falling back to now is precisely
    /// the failure this whole builtin was fixed for: a wrong answer nothing marks as
    /// wrong.
    /// </summary>
    [Fact]
    public void DateRefusesASpellingItCannotRead()
    {
        ExecutionResult result = Run("date -d 'next tuesday afternoon' +%F");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("invalid date", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("next tuesday afternoon", result.Stderr, StringComparison.Ordinal);
        Assert.Equal(string.Empty, result.Stdout);
    }

    // ---- parsing and expansion -------------------------------------------------------

    [Fact]
    public void EchoQuotingAndExpansion()
    {
        Assert.Equal("hello world\n", Out("echo hello world"));
        Assert.Equal("hello   world\n", Out("echo 'hello   world'"));
        Assert.Equal("a b\n", Out("echo \"a b\""));
        Assert.Equal("no-newline", Out("echo -n no-newline"));
        Assert.Equal("tab\there\n", Out("echo -e 'tab\\there'"));
        Assert.Equal("x=1\n", Out("x=1; echo x=$x"));
        Assert.Equal("1-2\n", Out("a=1 b=2; echo ${a}-${b}"));
        Assert.Equal("fallback\n", Out("echo ${missing:-fallback}"));
        Assert.Equal("3\n", Out("set -- a b c; echo $#"));
        Assert.Equal("literal $x\n", Out("echo 'literal $x'"));
        Assert.Equal("$\n", Out("echo \\$"));
    }

    [Fact]
    public void CommandSubstitutionAndArithmetic()
    {
        Assert.Equal("hi\n", Out("echo $(echo hi)"));
        Assert.Equal("hi\n", Out("echo `echo hi`"));
        Assert.Equal("7\n", Out("echo $((3 + 4))"));
        Assert.Equal("2\n", Out("n=6; echo $((n / 3))"));
        Assert.Equal("nested\n", Out("echo $(echo $(echo nested))"));
    }

    [Theory]
    [InlineData("set -- 6 7; echo $(( $1 * $2 ))", "42\n")]
    [InlineData("set -- -3 8; echo $(( ${1} * $2 + $# ))", "-22\n")]
    [InlineData("set -- 1 2 3 4 5 6 7 8 9 12; echo $(( ${10} + $10 ))", "22\n")]
    public void ArithmeticUsesScriptPositionalParameters(string command, string expected)
    {
        Assert.Equal(expected, Out(command));
    }

    [Fact]
    public void PipelinesConditionalsAndExitCodes()
    {
        Assert.Equal("b\n", Out("printf 'a\\nb\\nc\\n' | sed -n '2p'"));
        Assert.Equal("3\n", Out("printf 'a\\nb\\nc\\n' | wc -l").Trim() + "\n");
        Assert.Equal("ok\n", Out("true && echo ok"));
        Assert.Equal("ok\n", Out("false || echo ok"));
        ExecutionResult shortCircuit = Run("false && echo no");
        Assert.Equal(1, shortCircuit.ExitCode);
        Assert.Equal(string.Empty, shortCircuit.Stdout);
        Assert.Equal(0, Run("true").ExitCode);
        Assert.Equal(1, Run("false").ExitCode);
        Assert.Equal(3, Run("exit 3").ExitCode);
        Assert.Equal(127, Run("definitely-not-a-command").ExitCode);
        Assert.Contains("command not found", Run("definitely-not-a-command").Stderr, StringComparison.Ordinal);
        Assert.Equal("1\n", Out("false; echo $?"));
    }

    [Fact]
    public void SetErrExitStopsAtTheFirstFailure()
    {
        ExecutionResult result = Run("set -e\nfalse\necho unreachable");
        Assert.NotEqual(0, result.ExitCode);
        Assert.DoesNotContain("unreachable", result.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void RedirectionsAndHeredocs()
    {
        Assert.Equal(string.Empty, Out("echo written > out.txt"));
        Assert.Equal("written\n", File.ReadAllText(Path.Combine(_work, "out.txt")));
        Out("echo appended >> out.txt");
        Assert.Equal("written\nappended\n", File.ReadAllText(Path.Combine(_work, "out.txt")));
        Assert.Equal("written\nappended\n", Out("cat < out.txt"));
        Assert.Equal(string.Empty, Out("echo hidden > /dev/null"));

        Out("cat > heredoc.txt <<'EOF'\nline $notexpanded\nsecond\nEOF");
        Assert.Equal("line $notexpanded\nsecond\n", File.ReadAllText(Path.Combine(_work, "heredoc.txt")));

        Out("v=expanded; cat > heredoc2.txt <<EOF\nvalue $v\nEOF");
        Assert.Equal("value expanded\n", File.ReadAllText(Path.Combine(_work, "heredoc2.txt")));

        ExecutionResult errors = Run("ls /definitely/missing 2>err.txt");
        Assert.NotEqual(0, errors.ExitCode);
        Assert.Contains("ls:", File.ReadAllText(Path.Combine(_work, "err.txt")), StringComparison.Ordinal);
        Assert.Equal("both\n", Out("echo both 2>&1"));
    }

    [Fact]
    public void GlobsAndBraces()
    {
        File.WriteAllText(Path.Combine(_work, "a.txt"), "1");
        File.WriteAllText(Path.Combine(_work, "b.txt"), "2");
        File.WriteAllText(Path.Combine(_work, "c.md"), "3");
        Assert.Equal("a.txt b.txt\n", Out("echo *.txt"));
        Assert.Equal("c.md\n", Out("echo *.md"));
        Assert.Equal("a.txt\n", Out("echo ?.txt | cut -d' ' -f1"));
    }

    [Fact]
    public void ControlFlowAndFunctions()
    {
        Assert.Equal("1\n2\n3\n", Out("for i in 1 2 3; do echo $i; done"));
        Assert.Equal("0\n1\n", Out("i=0; while [ $i -lt 2 ]; do echo $i; i=$((i+1)); done"));
        Assert.Equal("yes\n", Out("if [ 1 -eq 1 ]; then echo yes; else echo no; fi"));
        Assert.Equal("no\n", Out("if [ 1 -eq 2 ]; then echo yes; else echo no; fi"));
        Assert.Equal("match\n", Out("case abc in a*) echo match;; *) echo other;; esac"));
        Assert.Equal("hi bob\n", Out("greet() { echo hi $1; }; greet bob"));
        Assert.Equal("a\nb\n", Out("for f in a b c; do if [ $f = c ]; then break; fi; echo $f; done"));
    }

    [Fact]
    public void TestOperators()
    {
        File.WriteAllText(Path.Combine(_work, "file.txt"), "x");
        Assert.Equal(0, Run("[ -f file.txt ]").ExitCode);
        Assert.Equal(1, Run("[ -f nothing.txt ]").ExitCode);
        Assert.Equal(0, Run("[ -d . ]").ExitCode);
        Assert.Equal(0, Run("[ -s file.txt ]").ExitCode);
        Assert.Equal(0, Run("[ -z '' ]").ExitCode);
        Assert.Equal(0, Run("[ -n 'x' ]").ExitCode);
        Assert.Equal(0, Run("[ a = a ]").ExitCode);
        Assert.Equal(0, Run("[ a != b ]").ExitCode);
        Assert.Equal(0, Run("[ 2 -gt 1 ]").ExitCode);
        Assert.Equal(0, Run("[ 1 -lt 2 -a 3 -ge 3 ]").ExitCode);
        Assert.Equal(0, Run("[ ! -f nothing.txt ]").ExitCode);
        // A file outside the session is not visible, so the test is false rather than an error.
        Assert.Equal(1, Run("[ -f /etc/passwd ]").ExitCode);
        Assert.Equal(0, Run("[[ abc == a* ]]").ExitCode);
        Assert.Equal(1, Run("[[ abc == b* ]]").ExitCode);
        Assert.Equal(0, Run("[[ abc =~ ^a.c$ ]]").ExitCode);
    }

    // ---- confinement -----------------------------------------------------------------

    [Fact]
    public void WritesOutsideTheWorkspaceAreRefused()
    {
        ExecutionResult outside = Run("echo escaped > /tmp/tensoragent-escape.txt");
        Assert.NotEqual(0, outside.ExitCode);
        Assert.False(File.Exists("/tmp/tensoragent-escape.txt"));

        ExecutionResult parent = Run("echo escaped > ../escape.txt");
        Assert.NotEqual(0, parent.ExitCode);
        Assert.False(File.Exists(Path.Combine(_root, "escape.txt")));

        ExecutionResult hosts = Run("cat /etc/hosts");
        Assert.NotEqual(0, hosts.ExitCode);
        Assert.Equal(string.Empty, hosts.Stdout);
        Assert.Contains("denied", hosts.Stderr, StringComparison.OrdinalIgnoreCase);
    }

    [SkippableFact]
    public void ASymlinkOutOfTheWorkspaceIsNotAWayThrough()
    {
        string secret = Path.Combine(_root, "secret.txt");
        File.WriteAllText(secret, "classified");
        TestPlatforms.CreateFileSymlink(Path.Combine(_work, "link.txt"), secret);
        ExecutionResult read = Run("cat link.txt");
        Assert.NotEqual(0, read.ExitCode);
        Assert.DoesNotContain("classified", read.Stdout, StringComparison.Ordinal);

        ExecutionResult write = Run("echo overwritten > link.txt");
        Assert.NotEqual(0, write.ExitCode);
        Assert.Equal("classified", File.ReadAllText(secret));
    }

    [Fact]
    public void ReadableRootsAreReadableButNotWritable()
    {
        File.WriteAllText(Path.Combine(_readable, "SKILL.md"), "# skill");
        ShellContext context = Context();
        Assert.Equal("# skill\n", Out($"cat '{Path.Combine(_readable, "SKILL.md")}'", context).Replace("# skill", "# skill").TrimEnd() + "\n");
        Assert.NotEqual(0, Run($"echo x > '{Path.Combine(_readable, "new.txt")}'", context).ExitCode);
        Assert.False(File.Exists(Path.Combine(_readable, "new.txt")));
    }

    [Fact]
    public void TheTempRootIsWritable()
    {
        Assert.Equal(0, Run("echo scratch > $TMPDIR/scratch.txt").ExitCode);
        Assert.Equal("scratch\n", File.ReadAllText(Path.Combine(_temp, "scratch.txt")));
    }

    // ---- state -------------------------------------------------------------------------

    [Fact]
    public void WorkingDirectoryAndEnvironmentComeBackToTheCaller()
    {
        Directory.CreateDirectory(Path.Combine(_work, "sub"));
        ExecutionResult result = Run("cd sub && export MARKER=kept && echo $PWD");
        Assert.Equal(0, result.ExitCode);
        // The shell hands back a fully resolved path, and on macOS the temp root is
        // reached through the /var -> /private/var link, so compare resolved to resolved.
        Assert.Equal(ConfinedPaths.RealPath(Path.Combine(_work, "sub")), result.WorkingDirectory);
        Assert.Equal("kept", result.Environment["MARKER"]);
        Assert.Contains("sub", result.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void TheHostEnvironmentIsVisibleToTheScript()
    {
        var env = new Dictionary<string, string> { ["PYTHONPATH"] = "/env", ["CUSTOM"] = "value" };
        Assert.Equal("value\n", Out("echo $CUSTOM", Context(env: env)));
        Assert.Equal("/env\n", Out("echo $PYTHONPATH", Context(env: env)));
    }

    // ---- utilities ---------------------------------------------------------------------

    [Fact]
    public void FileUtilities()
    {
        Assert.Equal(0, Run("mkdir -p a/b/c").ExitCode);
        Assert.True(Directory.Exists(Path.Combine(_work, "a/b/c")));
        Out("echo one > a/one.txt");
        Out("cp a/one.txt a/two.txt");
        Assert.True(File.Exists(Path.Combine(_work, "a/two.txt")));
        Out("mv a/two.txt a/three.txt");
        Assert.False(File.Exists(Path.Combine(_work, "a/two.txt")));
        Assert.True(File.Exists(Path.Combine(_work, "a/three.txt")));
        Out("touch a/empty.txt");
        Assert.True(File.Exists(Path.Combine(_work, "a/empty.txt")));
        Assert.Contains("one.txt", Out("ls a"), StringComparison.Ordinal);
        Assert.Contains("one.txt", Out("ls -la a"), StringComparison.Ordinal);
        Out("rm a/empty.txt");
        Assert.False(File.Exists(Path.Combine(_work, "a/empty.txt")));
        Out("rm -rf a/b");
        Assert.False(Directory.Exists(Path.Combine(_work, "a/b")));
        Assert.Equal("one.txt\n", Out("basename a/one.txt"));
        Assert.Equal("a\n", Out("dirname a/one.txt"));
        Assert.Equal(0, Run("rm -f does-not-exist").ExitCode);
    }

    [Fact]
    public void FindAndDu()
    {
        Directory.CreateDirectory(Path.Combine(_work, "src"));
        File.WriteAllText(Path.Combine(_work, "src", "a.py"), "print(1)\n");
        File.WriteAllText(Path.Combine(_work, "src", "b.txt"), "text\n");
        string found = Out("find . -name '*.py'");
        Assert.Contains("a.py", found, StringComparison.Ordinal);
        Assert.DoesNotContain("b.txt", found, StringComparison.Ordinal);
        Assert.Contains("src", Out("find . -type d"), StringComparison.Ordinal);
        Assert.Equal(2, Out("find . -type f").Trim().Split('\n').Length);
        Assert.Contains("\t", Out("du -sh ."), StringComparison.Ordinal);
    }

    [Fact]
    public void TextUtilities()
    {
        File.WriteAllText(Path.Combine(_work, "data.txt"), "banana\napple\ncherry\napple\n");
        Assert.Equal("banana\napple\n", Out("head -2 data.txt"));
        Assert.Equal("apple\n", Out("tail -1 data.txt"));
        Assert.Equal("4", Out("wc -l data.txt").Trim().Split(' ')[0]);
        Assert.Equal("apple\napple\n", Out("grep apple data.txt"));
        Assert.Equal("2:apple\n4:apple\n", Out("grep -n apple data.txt"));
        Assert.Equal("2\n", Out("grep -c apple data.txt"));
        Assert.Equal("banana\ncherry\n", Out("grep -v apple data.txt"));
        Assert.Equal("apple\napple\nbanana\ncherry\n", Out("sort data.txt"));
        Assert.Equal("apple\nbanana\ncherry\n", Out("sort -u data.txt"));
        Assert.Equal("APPLE\n", Out("echo apple | tr '[:lower:]' '[:upper:]'"));
        Assert.Equal("elppa\n", Out("echo apple | rev"));
        Assert.Equal("banana-apple-cherry-apple-\n", Out("cat data.txt | tr '\\n' '-'").TrimEnd() + "\n");
        Assert.Equal("b\n", Out("echo 'a:b:c' | cut -d: -f2"));
        Assert.Equal("a:c\n", Out("echo 'a:b:c' | cut -d: -f1,3"));
        Assert.Equal("ba\n", Out("echo banana | cut -c1-2"));
        Assert.Equal("hello\n", Out("echo world | sed 's/world/hello/'"));
        Assert.Equal("xxx\n", Out("echo aaa | sed 's/a/x/g'"));
        Assert.Equal("2\n", Out("printf 'a\\nb\\na\\n' | sort | uniq -c | grep a | head -1 | awk '{print $1}'"));
    }

    [Fact]
    public void SedInPlaceEditsTheFile()
    {
        string path = Path.Combine(_work, "edit.txt");
        File.WriteAllText(path, "alpha\nbeta\n");
        Assert.Equal(0, Run("sed -i 's/beta/gamma/' edit.txt").ExitCode);
        Assert.Equal("alpha\ngamma\n", File.ReadAllText(path));
        // BSD's `sed -i ''` spelling, which the tool declaration teaches on macOS.
        Assert.Equal(0, Run("sed -i '' 's/alpha/delta/' edit.txt").ExitCode);
        Assert.Equal("delta\ngamma\n", File.ReadAllText(path));
    }

    [Fact]
    public void GrepRecursesWithInclude()
    {
        Directory.CreateDirectory(Path.Combine(_work, "pkg"));
        File.WriteAllText(Path.Combine(_work, "pkg", "one.py"), "import os\nvalue = 1\n");
        File.WriteAllText(Path.Combine(_work, "pkg", "two.py"), "value = 2\n");
        File.WriteAllText(Path.Combine(_work, "pkg", "notes.md"), "value = 3\n");
        string result = Out("grep -rn value pkg --include='*.py'");
        Assert.Contains("one.py:2:value = 1", result, StringComparison.Ordinal);
        Assert.Contains("two.py:1:value = 2", result, StringComparison.Ordinal);
        Assert.DoesNotContain("notes.md", result, StringComparison.Ordinal);
        Assert.Equal(1, Run("grep -q nothing-here pkg -r").ExitCode);
    }

    [Fact]
    public void Awk()
    {
        File.WriteAllText(Path.Combine(_work, "rows.txt"), "alice 30 eng\nbob 25 sales\ncarol 41 eng\n");
        Assert.Equal("alice\nbob\ncarol\n", Out("awk '{print $1}' rows.txt"));
        Assert.Equal("alice eng\ncarol eng\n", Out("awk '$3 == \"eng\" {print $1, $3}' rows.txt"));
        Assert.Equal("3\n", Out("awk 'END {print NR}' rows.txt"));
        Assert.Equal("96\n", Out("awk '{sum += $2} END {print sum}' rows.txt"));
        Assert.Equal("eng\n", Out("awk 'NR==1 {print $NF}' rows.txt"));
        Assert.Equal("ALICE\n", Out("awk 'NR==1 {print toupper($1)}' rows.txt"));
        Assert.Equal("a:b\n", Out("echo 'a b' | awk '{print $1 \":\" $2}'"));
        Assert.Equal("bob\n", Out("awk -F' ' '/bob/ {print $1}' rows.txt"));
        Assert.Equal("x=1\n", Out("awk 'BEGIN {printf \"x=%d\\n\", 1}'"));
        Assert.Equal("2\n", Out("awk 'BEGIN {n=0; for (i=0;i<2;i++) n++; print n}'"));
        Assert.Equal("root\n", Out("echo 'root:x:0' | awk -F: '{print $1}'"));
    }

    [Fact]
    public void XargsAndTeeAndDiff()
    {
        Assert.Equal("a b c\n", Out("printf 'a\\nb\\nc\\n' | xargs echo"));
        Assert.Equal("a\nb\n", Out("printf 'a\\nb\\n' | xargs -n1 echo"));
        Out("printf 'x\\n' | tee copy.txt > /dev/null");
        Assert.Equal("x\n", File.ReadAllText(Path.Combine(_work, "copy.txt")));

        File.WriteAllText(Path.Combine(_work, "left.txt"), "one\ntwo\n");
        File.WriteAllText(Path.Combine(_work, "right.txt"), "one\nthree\n");
        ExecutionResult diff = Run("diff -u left.txt right.txt");
        Assert.Equal(1, diff.ExitCode);
        Assert.Contains("-two", diff.Stdout, StringComparison.Ordinal);
        Assert.Contains("+three", diff.Stdout, StringComparison.Ordinal);
        Assert.Equal(0, Run("diff left.txt left.txt").ExitCode);
    }

    [Fact]
    public void Base64AndDigests()
    {
        Assert.Equal("aGk=\n", Out("echo -n hi | base64"));
        Assert.Equal("hi", Out("echo -n aGk= | base64 -d"));
        string sha = Out("echo -n abc | sha256sum");
        Assert.StartsWith("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", sha, StringComparison.Ordinal);
    }

    [Fact]
    public void Archives()
    {
        Directory.CreateDirectory(Path.Combine(_work, "bundle"));
        File.WriteAllText(Path.Combine(_work, "bundle", "a.txt"), "content\n");
        Assert.Equal(0, Run("zip -r bundle.zip bundle").ExitCode);
        Assert.True(File.Exists(Path.Combine(_work, "bundle.zip")));
        Assert.Equal(0, Run("mkdir -p out && unzip -o bundle.zip -d out").ExitCode);
        Assert.Equal("content\n", File.ReadAllText(Path.Combine(_work, "out", "bundle", "a.txt")));
        Assert.Contains("a.txt", Out("unzip -l bundle.zip"), StringComparison.Ordinal);

        Assert.Equal(0, Run("tar -czf bundle.tar.gz bundle").ExitCode);
        Assert.Equal(0, Run("mkdir -p out2 && tar -xzf bundle.tar.gz -C out2").ExitCode);
        Assert.Equal("content\n", File.ReadAllText(Path.Combine(_work, "out2", "bundle", "a.txt")));
        Assert.Contains("bundle/a.txt", Out("tar -tzf bundle.tar.gz"), StringComparison.Ordinal);
    }

    [Fact]
    public void AZipMemberCannotEscapeTheExtractionDirectory()
    {
        // A crafted archive with a ../ member is the classic path-traversal bug.
        string archive = Path.Combine(_work, "evil.zip");
        using (FileStream stream = File.Create(archive))
        using (var zip = new System.IO.Compression.ZipArchive(stream, System.IO.Compression.ZipArchiveMode.Create))
        {
            System.IO.Compression.ZipArchiveEntry entry = zip.CreateEntry("../escaped.txt");
            using StreamWriter writer = new(entry.Open());
            writer.Write("owned");
        }
        ExecutionResult result = Run("unzip -o evil.zip -d out");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("outside the destination", result.Stderr, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(_root, "escaped.txt")));
        Assert.False(File.Exists(Path.Combine(_work, "escaped.txt")));
    }

    // ---- policy ------------------------------------------------------------------------

    [Fact]
    public void TheNetworkIsRefusedWhenTheUserHasNotAllowedIt()
    {
        ExecutionResult result = Run("curl https://example.com");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr, StringComparison.Ordinal);

        ExecutionResult wget = Run("wget https://example.com");
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, wget.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void ScriptsAreRefusedWhenTheUserHasTurnedThemOff()
    {
        ShellContext context = Context(Policy(scripts: false));
        ExecutionResult result = Run("python3 -c 'print(1)'", context);
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.ScriptsDisabledMessage, result.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void PipInstallIsRefusedWithoutAHostInstaller()
    {
        ExecutionResult result = Run("pip install requests");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.InstallsByHostMessage, result.Stderr, StringComparison.Ordinal);
        // The options that would change WHERE a package comes from are refused by name.
        Assert.Contains("--index-url", Run("pip install --index-url http://x requests").Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void PythonReportsThatNoInterpreterIsEmbeddedRatherThanPretending()
    {
        ExecutionResult result = Run("python3 script.py");
        Assert.Equal(127, result.ExitCode);
        Assert.Contains("Python", result.Stderr, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OutputIsCappedAndTheCapIsVisible()
    {
        var policy = Policy() with { MaxOutputBytes = 2048 };
        ExecutionResult result = Run("i=0; while [ $i -lt 500 ]; do echo 0123456789012345678901234567890123456789; i=$((i+1)); done", Context(policy));
        Assert.Equal(0, result.ExitCode);
        Assert.True(result.Stdout.Length < 4096, $"stdout was {result.Stdout.Length} bytes");
        Assert.Contains("dropped", result.Stdout, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ATimeoutStopsTheCommandAndSaysSo()
    {
        var policy = Policy() with { DefaultTimeout = TimeSpan.FromMilliseconds(700) };
        ExecutionResult result = Run("while true; do :; done", Context(policy));
        Assert.True(result.TimedOut);
        Assert.Equal(ExecutionResult.TimeoutExitCode, result.ExitCode);
        Assert.Contains("timed out", result.Stderr, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ArgvRunsOneCommandWithoutParsing()
    {
        ExecutionResult result = _shell.RunArgvAsync(new[] { "echo", "a b", "$notexpanded" }, Context(), CancellationToken.None)
            .GetAwaiter().GetResult();
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("a b $notexpanded\n", result.Stdout);
    }

    [Fact]
    public void LiveOutputIsTappedLineByLine()
    {
        var lines = new List<string>();
        var context = new ShellContext
        {
            WorkingDirectory = _work,
            Policy = Policy(),
            OnStdoutLine = lines.Add,
        };
        Run("echo first; echo second", context);
        Assert.Equal(new[] { "first", "second" }, lines);
    }

    [Fact]
    public void TheBuiltinListIsWhatTheDeclarationWouldAdvertise()
    {
        IReadOnlyCollection<string> names = InProcessShell.BuiltinNames;
        foreach (string expected in new[] { "ls", "cat", "grep", "sed", "awk", "head", "tail", "mkdir", "python3", "node", "sh", "find", "sort", "wc", "cut", "tr", "diff", "tar", "unzip" })
            Assert.Contains(expected, names);
    }
}
