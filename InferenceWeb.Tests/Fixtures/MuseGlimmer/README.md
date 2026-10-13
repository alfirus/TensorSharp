# Muse-Glimmer chat template

`chat_template.jinja` is the unmodified `tokenizer.chat_template` metadata from
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, inspected on 2026-10-11. It ignores
`enable_thinking` and ends every generation prompt with a bare assistant header.
The fixture exercises the actual template's reasoning, recipient, and tool
branches without requiring model weights or a native backend.
