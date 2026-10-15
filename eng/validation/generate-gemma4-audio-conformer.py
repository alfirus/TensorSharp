#!/usr/bin/env python3
"""Generate tiny independent scalar/double Gemma 4 conformer block fixtures.

No tensor runtime or reference-engine output is used. The fixture exercises
RMSNorm, clip-before/after-linear, FFN residual halves, content+relative local
attention, GLU channel ordering, causal depthwise convolution, and block norm.
"""
import argparse
import json
import math
from pathlib import Path


def pattern(count, period, divisor, offset=0):
    return [((i * 7 + offset) % period - period // 2) / divisor for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/EmbeddingGemma2/audio-conformer.json"))
    args = parser.parse_args()
    hidden, heads, head, kernel = 8, 2, 4, 5
    norm_names = ["ff1pre", "ff1post", "attnpre", "attnpost", "convpre", "convpost", "ff2pre", "ff2post", "block"]
    norms = [[.875 + ((i + index) % 5) / 16 for i in range(hidden)] for index in range(9)]
    per_dim = [.75, 1.0, 1.125, .875]
    depthwise = pattern(hidden * kernel, 7, 16, 2)
    shapes = [(hidden, 12), (12, hidden), *[(hidden, hidden)] * 4,
              (hidden, 2 * hidden), (hidden, hidden), (hidden, 12), (12, hidden)]
    linears = []
    for index, (inputs, outputs) in enumerate(shapes):
        linears.append(dict(input=inputs, output=outputs, weight=pattern(inputs * outputs, 11, 32, index),
            bias=pattern(outputs, 5, 32, index + 1), has_clamp=index % 3 != 1,
            input_min=-.75, input_max=.75, output_min=-1.25, output_max=1.25))

    def clip(rows, low, high):
        return [[min(high, max(low, value)) for value in row] for row in rows]

    def norm(rows, weight, epsilon):
        if weight is None:
            return [row[:] for row in rows]
        return [[value * gain / math.sqrt(sum(x * x for x in row) / len(row) + epsilon)
                 for value, gain in zip(row, weight)] for row in rows]

    def add(left, right, scale=1):
        return [[a + scale * b for a, b in zip(x, y)] for x, y in zip(left, right)]

    def linear(rows, descriptor):
        if descriptor["has_clamp"]:
            rows = clip(rows, descriptor["input_min"], descriptor["input_max"])
        out = [[sum(a * b for a, b in zip(row, descriptor["weight"][r * descriptor["input"]:(r + 1) * descriptor["input"]]))
                + descriptor["bias"][r] for r in range(descriptor["output"])] for row in rows]
        return clip(out, descriptor["output_min"], descriptor["output_max"]) if descriptor["has_clamp"] else out

    def silu(rows):
        return [[value / (1 + math.exp(-value)) for value in row] for row in rows]

    cases = []
    for sequence, past, future, large, absent_norms in [(1, 12, 0, False, False), (25, 3, 2, False, False), (13, 12, 0, True, False), (7, 2, 0, False, True), (96, 12, 0, False, False), (128, 12, 0, False, False)]:
        epsilon, cap, grad_clip, residual_scale = .0001, 2.0, 1024.0, .5
        ns = [None] * 9 if absent_norms else norms
        flat = pattern(sequence * hidden, 19, 8)
        if large:
            flat = [(80000 if i % 2 == 0 else -80000) + value for i, value in enumerate(flat)]
        original = [flat[t * hidden:(t + 1) * hidden] for t in range(sequence)]
        relative = [pattern(hidden, 13, 64, r * 3) for r in range(past + future + 1)]

        def ff(rows, norm_index, linear_index):
            projected = linear(norm(clip(rows, -grad_clip, grad_clip), ns[norm_index], epsilon), linears[linear_index])
            projected = linear(silu(projected), linears[linear_index + 1])
            return add(rows, norm(clip(projected, -grad_clip, grad_clip), ns[norm_index + 1], epsilon), residual_scale)

        def block(initial):
            rows = ff(initial, 0, 0)
            normalized = norm(clip(rows, -grad_clip, grad_clip), ns[2], epsilon)
            q, k, v = [linear(normalized, linears[index]) for index in (2, 3, 4)]
            q_scale = head ** -.5 / math.log(2)
            k_scale = math.log1p(math.e) / math.log(2)
            q = [[value * q_scale * per_dim[i % head] for i, value in enumerate(row)] for row in q]
            k = [[value * k_scale for value in row] for row in k]
            attention = [[0.0] * hidden for _ in range(sequence)]
            for t in range(sequence):
                for h in range(heads):
                    indices = list(range(max(0, t - past), min(sequence - 1, t + future) + 1))
                    logits = [cap * math.tanh(sum(q[t][h * head + d] * (k[s][h * head + d] + relative[s - t + past][h * head + d])
                                                   for d in range(head)) / cap) for s in indices]
                    values = [math.exp(value - max(logits)) for value in logits]
                    for d in range(head):
                        attention[t][h * head + d] = sum(probability * v[s][h * head + d] for s, probability in zip(indices, values)) / sum(values)
            projected = linear(attention, linears[5])
            rows = add(rows, norm(clip(projected, -grad_clip, grad_clip), ns[3], epsilon))
            pw = linear(norm(rows, ns[4], epsilon), linears[6])
            glu = [[row[d] / (1 + math.exp(-row[hidden + d])) for d in range(hidden)] for row in pw]
            dw = [[sum(glu[t - (kernel - 1 - tap)][channel] * depthwise[channel * kernel + tap]
                       for tap in range(kernel) if t >= kernel - 1 - tap) for channel in range(hidden)] for t in range(sequence)]
            conv = linear(silu(norm(clip(dw, -grad_clip, grad_clip), ns[5], epsilon)), linears[7])
            rows = ff(add(rows, conv), 6, 8)
            return norm(clip(rows, -grad_clip, grad_clip), ns[8], epsilon)
        original_relative = relative
        expected = block(original)
        relative_two = [[-value / 2 for value in row] for row in original_relative]
        relative = relative_two
        expected_two = block(expected)
        relative_three = [[value / 4 for value in row] for row in original_relative]
        relative = relative_three
        expected_three = block(expected_two)
        relative = original_relative
        cases.append(dict(name=f"seq{sequence}_past{past}_future{future}" + ("_large" if large else "") + ("_identity_norms" if absent_norms else ""),
            sequence=sequence, past=past, future=future, epsilon=epsilon, cap=cap, grad_clip=grad_clip,
            residual_weight=residual_scale, absent_norms=absent_norms, precise_long=sequence >= 96, input=flat,
            relative=[value for row in relative for value in row], expected=[value for row in expected for value in row],
            relative_two=[value for row in relative_two for value in row], relative_three=[value for row in relative_three for value in row],
            expected_two=[value for row in expected_two for value in row],
            expected_three=[value for row in expected_three for value in row]))
    result = dict(method="Independent scalar double-precision complete conformer block; no backend or model output copied",
        source_revision="976e7a41c6615559f7a2ccb79fd5212c378b2c95",
        source_url="https://github.com/huggingface/transformers/blob/976e7a41c6615559f7a2ccb79fd5212c378b2c95/src/transformers/models/gemma4/modeling_gemma4.py",
        regeneration="python3 eng/validation/generate-gemma4-audio-conformer.py", hidden=hidden, heads=heads, head=head, kernel=kernel,
        norm_order=norm_names, norms=norms, per_dim_scale=per_dim, depthwise=depthwise, linears=linears, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
