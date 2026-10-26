// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Runtime.Scheduling;

/// <summary>
/// Optional exact one-token forward for plain greedy decoding. This is a
/// separate result type from logits and must not be used with sampling,
/// penalties, constraints, speculative verification, or logit capture.
/// </summary>
public interface IGreedyForwardModel
{
    /// <summary>
    /// Append <paramref name="token"/> to the current sequence and return the
    /// lowest vocabulary index attaining the maximum of the complete finite
    /// F32 logit row. Success advances the sequence by exactly one input token.
    /// False declines without changing model/cache state; callers may then use
    /// ordinary forwarding. Execution failures throw and require request reset;
    /// callers must not retry the same token through another forward path.
    /// </summary>
    bool TryForwardGreedyToken(int token, out int nextToken);
}
