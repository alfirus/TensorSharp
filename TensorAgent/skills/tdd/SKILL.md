---
name: tdd
description: Build features or fix bugs test-first through agreed public interfaces using the red-green loop. Tested with Gemma4 E4B at 16K context and 24 tool rounds.
license: MIT
compatibility: TensorSharp or TensorAgent; workspace file tools required for saved artifacts and code changes. Tested on Gemma4 E4B Q8_0 with 16384-token context and up to 24 tool rounds. E2B failed the earlier eight-round task; E2B at 24 rounds and other models are not validated.
metadata:
  source: mattpocock/skills
  revision: 49dd158d1076134a641b33efb035946536778336
---

# Test-Driven Development

TDD is the red → green loop. This skill is the reference that makes that loop produce tests worth keeping: what a good test is, where tests go, the anti-patterns, and the rules of the loop. Every section applies on every cycle: consult them before and during the loop, not after.

When exploring the codebase, read `GLOSSARY.md` (if it exists) so test names and interface vocabulary match the project's domain language, and respect ADRs in the area you're touching.

## What a good test is

Tests verify behavior through public interfaces, not implementation details. Code can change entirely; tests shouldn't. A good test reads like a specification: "user can checkout with valid cart" tells you exactly what capability exists, and it survives refactors because it doesn't care about internal structure.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for mocking guidelines.

## Seams: where tests go

A **seam** is the public boundary you test at: the interface where you observe behavior without reaching inside. Tests live at seams, never against internals.

**Test only at pre-agreed seams.** Before writing any test, write down the seams under test and confirm them with the user. No test is written at an unconfirmed seam. You can't test everything, so agreeing the seams up front is how testing effort lands on the critical paths and complex logic instead of every edge case.

Ask: "What's the public interface, and which seams should we test?" Give each proposed seam a one-line note on what it catches and what it misses.

When the shape of that interface is itself in question (how deep the module is, where the seam belongs, what the interface should expose), read `skills_read` with `skill="tdd"`, `path="codebase-design.md"` for the vocabulary. It is the shared source of the module, interface, depth, seam, adapter, leverage and locality terms, and it is a reference to consult, not a session to run.

## Anti-patterns

- **Implementation-coupled**: mocks internal collaborators, tests private methods, or verifies through a side channel (querying the database instead of using the interface). The tell: the test breaks when you refactor but behavior hasn't changed.
- **Tautological**: the assertion recomputes the expected value the way the code does (`expect(add(a, b)).toBe(a + b)`, a snapshot derived by hand the same way, a constant asserted equal to itself), so it passes by construction and can never disagree with the code. Expected values must come from an independent source of truth: a known-good literal, a worked example, the spec.
- **Horizontal slicing**: writing all tests first, then all implementation. Bulk tests verify _imagined_ behavior: you test the _shape_ of things rather than user-facing behavior, the tests go insensitive to real changes, and you commit to test structure before understanding the implementation. Work in **vertical slices** instead: one test → one implementation → repeat, each test a **tracer bullet** that responds to what the last cycle taught you.

## Rules of the loop

- **Red before green.** Write the failing test first, then only enough code to pass it. Don't anticipate future tests or add speculative features.
- **One slice at a time.** One seam, one test, one minimal implementation per cycle.
- **Refactoring is not part of the loop.** It belongs to the review stage (review the agreed public interface and test results with the user), not the red → green implementation cycle.

## Execute and prove each cycle

For an approved interface, first create one public test and the minimal failing implementation (for example a stub). Run that test with the offered shell tool and confirm its real failure before implementing the behavior. A test name containing 'red' is not evidence of a red run. Implement only that slice and run it again to prove green; then repeat for the remaining agreed behavior. Report actual failing and passing command results, never inferred outcomes or runs you did not perform.

## Host capabilities

Use only tools offered in this conversation. Read supplied workspace files with `read_file`; create artifacts with `write_file` and edit existing files with `apply_patch` when offered. Do not invent a `Skill` tool. After creating any file, include its exact Markdown download link from the tool result in your answer, even when the rest of your answer asks the user a decision question. Never replace that link with a filename or a claim that the file exists. If a required file, workspace, executable, or tool is unavailable, explain the limitation and do not claim it ran. A user's already-agreed scope and test interface count as agreement; do not ask the same question again. Apply the workflow directly after reading it; keep progress narration brief.
