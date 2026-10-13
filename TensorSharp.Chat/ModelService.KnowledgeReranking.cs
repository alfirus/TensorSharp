// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Server.Knowledge;

namespace TensorSharp.Server;

public partial class ModelService
{
    internal IKnowledgeReranker? CreateKnowledgeReranker() => ModelKnowledgeReranker.Create(this,
        (outcome, candidates, elapsed) => _logger.LogInformation(
            "Knowledge rerank {Outcome}: {CandidateCount} candidates, {ElapsedMilliseconds} ms",
            outcome, candidates, elapsed));
}
