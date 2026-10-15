---
name: grill-with-docs
description: Interview a plan, record agreed terms and decisions, and return saved document links before questions. Tested with Gemma4 E4B at 16K context and 24 tool rounds.
license: MIT
compatibility: TensorSharp or TensorAgent; workspace file tools required for saved artifacts and code changes. Tested on Gemma4 E4B Q8_0 with 16384-token context and up to 24 tool rounds. E2B failed the earlier eight-round task; E2B at 24 rounds and other models are not validated.
metadata:
  source: mattpocock/skills
  revision: 49dd158d1076134a641b33efb035946536778336
---

# Save decisions and interview

When the user requests a glossary or decision document, save the already-agreed facts first. Immediately copy the exact Markdown artifact link from the successful file-tool result as the first line of your answer; then ask the unresolved frontier questions. A filename or a statement that the document was saved does not deliver it. Keep agreed facts separate from recommendations.

# Plan interview

Interview the user relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **frontier** is every decision whose prerequisites are already settled: the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole frontier in one round: number each question and give your recommended answer. Then wait for the user's answers before the next round.

Format a round like so:

```
❓ **Q1** - **<question title>**: <question body, might be multiple paragraphs, including multiple choices>

➡️ <your recommended answer>

---

❓ **Q2** - **<question title>**: <question body, might be multiple paragraphs, including multiple choices>

➡️ <your recommended answer>
```

Word each question so "yes" accepts your recommended answer.

Each round the user answers reshapes the tree: settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the user's. When a frontier question needs a fact from the environment (filesystem, tools, etc.), use the available file tools to find it (delegate only if an agent tool is offered); don't ask the user for anything you could look up yourself. Don't block on it: a running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the sub-agent to report; ask the rest of the frontier now. The _decisions_ are the user's: put each to them and wait.

The session is done when the frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the user confirms you have reached a shared understanding.

As terminology and durable decisions are resolved, read `skills_read` with `skill="grill-with-docs"`, `path="domain-modeling.md"`. Use that guidance to maintain the glossary and ADRs in the session workspace.

## Host capabilities

Use only tools offered in this conversation. Read supplied workspace files with `read_file`; create artifacts with `write_file` and edit existing files with `apply_patch` when offered. Do not invent a `Skill` tool. After creating any file, include its exact Markdown download link from the tool result in your answer, even when the rest of your answer asks the user a decision question. Never replace that link with a filename or a claim that the file exists. If a required file, workspace, executable, or tool is unavailable, explain the limitation and do not claim it ran. A user's already-agreed scope and test interface count as agreement; do not ask the same question again. Apply the workflow directly after reading it; keep progress narration brief.
