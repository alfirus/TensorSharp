#!/usr/bin/env python3
"""Generate independent scalar/double two-convolution Gemma 4 audio fixtures.

The reference performs direct convolution, mean-centered channel LayerNorm,
ReLU, then the channel-fast frequency flattening and learned projection. It
does not invoke a tensor backend, im2col, or copy any engine output.
"""
import argparse
import json
import math
from pathlib import Path


def pattern(count, period, divisor, offset=0):
    return [((i * 7 + offset) % period - period // 2) / divisor for i in range(count)]


def convolution(planar, width, height, descriptor, epsilon):
    inputs, outputs = descriptor["input_channels"], descriptor["output_channels"]
    kw, kh = descriptor["kernel_frequency"], descriptor["kernel_time"]
    out_w, out_h = (width + 2 - kw) // 2 + 1, (height + 2 - kh) // 2 + 1
    result = [0.0] * (out_w * out_h * outputs)
    for y in range(out_h):
        for x in range(out_w):
            channels = []
            for oc in range(outputs):
                value = 0.0
                for ic in range(inputs):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = y * 2 - 1 + ky, x * 2 - 1 + kx
                            if 0 <= iy < height and 0 <= ix < width:
                                index = ((oc * inputs + ic) * kh + ky) * kw + kx
                                value += planar[(ic * height + iy) * width + ix] * descriptor["weight"][index]
                channels.append(value)
            if descriptor["norm_weight"] is not None or descriptor["norm_bias"] is not None:
                mean = sum(channels) / outputs
                variance = sum((value - mean) ** 2 for value in channels) / outputs
                scale = 1 / math.sqrt(variance + epsilon)
                channels = [(value - mean) * scale for value in channels]
                if descriptor["norm_weight"] is not None:
                    channels = [value * weight for value, weight in zip(channels, descriptor["norm_weight"])]
                if descriptor["norm_bias"] is not None:
                    channels = [value + bias for value, bias in zip(channels, descriptor["norm_bias"])]
            for oc, value in enumerate(channels):
                result[(oc * out_h + y) * out_w + x] = max(0.0, value)
    return result, out_w, out_h


def make_case(name, width, height, channels, shapes, normalization):
    epsilon = .0001
    integer_case = name == "q8_integer_projection_equivalence"
    data = [float((i * 3 + i // width) % 2) for i in range(width * height * channels)] if integer_case else pattern(width * height * channels, 17, 16)
    original = data[:]
    layers, intermediates = [], []
    w, h, inputs = width, height, channels
    for index, (kw, kh, outputs) in enumerate(shapes):
        normalize = normalization[index]
        descriptor = dict(input_channels=inputs, output_channels=outputs,
            kernel_frequency=kw, kernel_time=kh,
            weight=pattern(inputs * outputs * kw * kh, 11, 32, index + 1),
            norm_weight=[.75 + ((i + index) % 5) / 16 for i in range(outputs)] if normalize else None,
            norm_bias=pattern(outputs, 7, 32, index + 2) if normalize else None)
        if integer_case:
            descriptor["weight"] = [0.0] * len(descriptor["weight"])
            for output in range(outputs):
                descriptor["weight"][((output * inputs) * kh + kh // 2) * kw + kw // 2] = 1.0
        data, w, h = convolution(data, w, h, descriptor, epsilon)
        intermediates.append(dict(width=w, height=h, channels=outputs, planar=data[:]))
        layers.append(descriptor)
        inputs = outputs
    columns, outputs = w * inputs, 6
    # Every projection coefficient is exactly representable in Q8_0 with
    # binary scale 1/32. A 32-aligned case exercises compressed equivalence.
    projection = dict(input=columns, output=outputs,
        weight=pattern(columns * outputs, 13, 32, 3), bias=pattern(outputs, 5, 64, 1),
        q8_scale=1 / 32 if integer_case and columns % 32 == 0 else None)
    flattened = [[data[(channel * h + time) * w + frequency]
                  for frequency in range(w) for channel in range(inputs)] for time in range(h)]
    expected = [sum(row[column] * projection["weight"][output * columns + column]
                    for column in range(columns)) + projection["bias"][output]
                for row in flattened for output in range(outputs)]
    return dict(name=name, width=width, height=height, channels=channels, epsilon=epsilon,
        input=original, convolutions=layers, intermediates=intermediates,
        projection=projection, output_rows=h, output_columns=outputs,
        expected=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=Path("InferenceWeb.Tests/Fixtures/EmbeddingGemma2/audio-sscp.json"))
    args = parser.parse_args()
    cases = [
        make_case("single_pixel", 1, 1, 1, [(3, 3, 3), (3, 3, 2)], [True, True]),
        make_case("odd_time_frequency", 9, 7, 1, [(3, 3, 4), (3, 3, 3)], [True, True]),
        make_case("rectangular_kernel_channel_order", 8, 5, 2, [(3, 1, 3), (1, 3, 4)], [True, True]),
        make_case("optional_norms", 6, 8, 2, [(1, 1, 3), (3, 3, 2)], [False, False]),
        make_case("q8_projection_equivalence", 13, 9, 1, [(3, 3, 2), (3, 3, 8)], [True, True]),
        make_case("q8_integer_projection_equivalence", 13, 9, 1, [(3, 3, 2), (3, 3, 8)], [False, False]),
    ]
    fixture = dict(method="Independent scalar double convolution/channel LayerNorm/ReLU/projection; no engine output",
        source_revision="976e7a41c6615559f7a2ccb79fd5212c378b2c95",
        source_url="https://github.com/huggingface/transformers/blob/976e7a41c6615559f7a2ccb79fd5212c378b2c95/src/transformers/models/gemma4/modeling_gemma4.py",
        regeneration="python3 eng/validation/generate-gemma4-audio-sscp.py",
        input_layout="planar[channel,time,frequency]", kernel_layout="[out_channel,in_channel,kernel_time,kernel_frequency]",
        projection_input_layout="[time,frequency*channels+channel]", stride=2, padding=1, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
