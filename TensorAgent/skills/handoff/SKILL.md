---
name: handoff
description: Write a concise handoff from the conversation so another agent can continue; redact sensitive values. Tested with Gemma4 E4B at 16K context and 24 tool rounds.
license: MIT
compatibility: TensorSharp or TensorAgent; workspace file tools required for saved artifacts and code changes. Tested on Gemma4 E4B Q8_0 with 16384-token context and up to 24 tool rounds. E2B failed the earlier eight-round task; E2B at 24 rounds and other models are not validated.
metadata:
  source: mattpocock/skills
  revision: 49dd158d1076134a641b33efb035946536778336
---

Write a handoff document summarising the current conversation so a fresh agent can continue the work. Save `handoff.md` in the session workspace using the offered file tool and return its artifact link. If no file tool/workspace is available, provide the full handoff in the answer and state that no file was saved.

Include a "suggested skills" section in the document, naming only relevant skills actually listed in the available catalog (the next agent can read them with `skills_read`).

Do not duplicate content already captured in other artifacts (specs, plans, ADRs, issues, commits, diffs). Reference them by path or URL instead.

Redact any sensitive information, such as API keys, passwords, or personally identifiable information.

If the user passed arguments, treat them as a description of what the next session will focus on and tailor the doc accordingly.

Replace secret values with `[REDACTED]` everywhere, including the saved file, response, annotations, examples, and any explanation of redaction. Never repeat an original secret in a sentence saying it was redacted. Preserve only the fact that a credential was removed.

## Host capabilities

Use only tools offered in this conversation. Read supplied workspace files with `read_file`; create artifacts with `write_file` and edit existing files with `apply_patch` when offered. Do not invent a `Skill` tool. After creating any file, include its exact Markdown download link from the tool result in your answer, even when the rest of your answer asks the user a decision question. Never replace that link with a filename or a claim that the file exists. If a required file, workspace, executable, or tool is unavailable, explain the limitation and do not claim it ran. A user's already-agreed scope and test interface count as agreement; do not ask the same question again. Apply the workflow directly after reading it; keep progress narration brief.
