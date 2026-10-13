// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Runtime;

/// <summary>
/// An in-process host continuation rendered as an ordinary chat message. A user-role
/// continuation still belongs to the current task; it must not become the task anchor
/// when earlier tool rounds are compacted. The marker has no wire-format fields and
/// cannot be supplied by model text or an incoming JSON chat message.
/// </summary>
public class HostContinuationMessage : ChatMessage
{
}
