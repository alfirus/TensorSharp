using DirectCudaProbe;

int checks = 0;
void Check(bool condition, string description)
{
    if (!condition) throw new InvalidOperationException(description);
    checks++;
}
void Reject(Action action, string description)
{
    try { action(); }
    catch (ArgumentException) { checks++; return; }
    throw new InvalidOperationException(description);
}

string raw = "The sum is 17 + 25 = 42.</think>42";
var glm = QualityOutput.Parse(raw, "glm5-next", "<|assistant|><think>", "production");
Check(glm.RawText == raw && glm.Content == "42", "GLM production framing preserves raw while selecting answer");
Check(glm.Thinking.Contains("17 + 25 = 42", StringComparison.Ordinal), "GLM reasoning must remain in evidence");
Check(QualityOutput.CompletePlainAnswer(glm, true), "EOS plain answer is eligible for independent semantic oracle");
Check(!QualityOutput.CompletePlainAnswer(glm, false), "Even a correct-looking parsed answer without EOS is incomplete");
var calledTool = glm with { ToolCalls = [new TensorSharp.Runtime.ToolCall { Name = "unexpected" }] };
Check(!QualityOutput.CompletePlainAnswer(calledTool, true), "Answer-only quality cases cannot pass after an unexpected tool call");
var historical = QualityOutput.Parse(raw, "glm5-next", "<|assistant|><think>", "raw");
Check(historical.Content == raw, "Default raw behavior must not strip arbitrary reasoning");
var empty = QualityOutput.Parse("<think>\n</think>\n42", "qwen3", "", "raw");
Check(empty.Content == "42", "Historical empty thinking block normalization remains unchanged");
var prose = QualityOutput.Parse("The answer is 42.", "unknown-architecture", "", "production");
Check(prose.Content == "The answer is 42.", "Production parser does not rewrite an answer to satisfy an oracle");
var unclosed = QualityOutput.Parse("{\"name\":\"Mars\"}", "glm5next", "<|assistant|><think>", "production");
Check(unclosed.Content == "{\"name\":\"Mars\"}", "GLM production no-close JSON fallback is preserved");
Check(QualityOutput.SelectCases("math,extraction").SequenceEqual(new[] { "math", "extraction" }), "Explicit subset is recorded without invented cases");
Check(QualityOutput.SelectCases("squares").Single() == "squares", "Square-list can be measured independently");
Reject(() => QualityOutput.SelectCases(""), "Empty selection cannot pass zero cases");
Reject(() => QualityOutput.SelectCases("math,math"), "Duplicate cases are not extra coverage");
Reject(() => QualityOutput.SelectCases("math,other"), "Unknown cases cannot silently disappear");
Reject(() => QualityOutput.Parse("42", "glm5next", "", "other"), "Unknown parser cannot silently fall back");
Check(QualityPrompt.ParseReasoningEffort(null) is null, "Omitted effort must preserve the historical default");
Check(QualityPrompt.ParseReasoningEffort(" LOW ") == "low", "Explicit effort uses production normalization");
foreach (string level in new[] { "low", "medium", "high" })
    Check(QualityPrompt.ParseReasoningEffort(level) == level, "All production effort levels are accepted");
foreach (string invalid in new[] { "", " ", "none", "max", "0", "invalid" })
    Reject(() => QualityPrompt.ParseReasoningEffort(invalid), "Malformed explicit effort cannot silently select a family default");
const string question = "Compute 17 + 25. Reply with only the integer.";
var defaultPrompt = QualityPrompt.Create("", "glm5-next", question, null);
string historicalPrompt = new TensorSharp.Runtime.GgufPromptRenderer().Render("", new()
{
    new() { Role = "system", Content = "You are a helpful assistant." },
    new() { Role = "user", Content = question },
}, architecture: "glm5-next", enableThinking: false);
Check(defaultPrompt.Text == historicalPrompt, "Default prompt must remain byte-for-byte unchanged");
Check(defaultPrompt.Text.Contains("Reasoning Effort: Max", StringComparison.Ordinal), "GLM omitted effort requests Max");
foreach ((string effort, string rendered) in new[] { ("low", "Low"), ("medium", "Max"), ("high", "High") })
{
    var prompt = QualityPrompt.Create("", "glm5-next", question, effort);
    Check(prompt.Text.Contains("Reasoning Effort: " + rendered, StringComparison.Ordinal), "Production GLM effort line must match requested level");
    Check(prompt.Text.EndsWith("<|assistant|><think>", StringComparison.Ordinal), "GLM low effort retains the thinking boundary");
    string expectedSha = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(prompt.Text))).ToLowerInvariant();
    Check(prompt.Utf8Sha256 == expectedSha, "Reported prompt hash covers the exact rendered UTF-8 bytes");
    Check(QualityOutput.Parse(raw, "glm5-next", prompt.Text, "production").Content == "42", "Actual effort prompt still parses the completed answer");
}
HostBudgetLifetimeTests.Run(Check);
Console.WriteLine($"PASS {checks} probe parser/case-selection/effort/prompt/host-budget-lifetime checks; no model or GPU executed.");
