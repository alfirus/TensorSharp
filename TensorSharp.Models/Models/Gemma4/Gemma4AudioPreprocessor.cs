// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using NLayer;
using NVorbis;

namespace TensorSharp.Models
{
    /// <summary>Audio logarithm policy. HuggingFaceAddFloor is the model default;
    /// LlamaClampFloor is available for explicit reference comparisons.</summary>
    public enum Gemma4AudioLogPolicy { HuggingFaceAddFloor, LlamaClampFloor }

    public class Gemma4AudioPreprocessor
    {
        private const int SampleRate = 16000;
        private const int MelBins = 128;
        private const double FrameLengthMs = 20.0;
        private const double HopLengthMs = 10.0;
        private const double MinFrequency = 0.0;
        private const double MaxFrequency = 8000.0;
        private const double MelFloor = 1e-3;

        private static readonly int FrameLength = (int)Math.Round(SampleRate * FrameLengthMs / 1000.0); // 320
        private static readonly int HopLength = (int)Math.Round(SampleRate * HopLengthMs / 1000.0); // 160
        private static readonly int FftLength = ComputeFftLength();
        private static readonly int NumFreqBins = FftLength / 2 + 1;
        private static readonly Complex[][] FftTwiddles = BuildFftTwiddles();
        private static readonly double[] HannWindow = BuildWindow();
        private static readonly MelFilter[] MelFilters = BuildMelFilterBank(NumFreqBins, MelBins, MinFrequency, MaxFrequency, SampleRate);

        // Each triangular mel filter has only a few nonzero FFT bins. Store its
        // contiguous support in ascending-bin order rather than rereading a
        // mostly zero, strided 257-by-128 matrix for every analysis frame.
        private readonly record struct MelFilter(int FirstBin, double[] Weights);

        public static float[] DecodeAudioFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            float[] samples = ext switch
            {
                ".wav" => DecodeWAV(File.ReadAllBytes(path)),
                ".mp3" => DecodeMp3(path),
                ".ogg" => DecodeOgg(path),
                _ => DecodeViaPlatform(path),
            };
            ValidateSamples(samples);
            return samples;
        }

        /// <summary>
        /// Every extension the managed decoders above do not handle (.m4a / .aac / .caf from
        /// Voice Memos, .flac, ...) goes to the platform audio decoder on
        /// <see cref="TensorSharp.Models.Media.MediaCodecs.Audio"/>, which returns native-rate
        /// planar PCM. The fold to mono and the
        /// resampler are this file's own, so the mel pipeline sees exactly what it always has.
        /// With no platform decoder registered this throws the provider's
        /// <see cref="NotSupportedException"/>, which names what to register.
        /// </summary>
        private static float[] DecodeViaPlatform(string path)
        {
            TensorSharp.Models.Video.DecodedAudio raw = TensorSharp.Models.Media.MediaCodecs.Audio.Decode(path);
            if (raw?.Channels is not { Length: > 0 } || raw.Channels[0] is not { Length: > 0 } || raw.SampleRate <= 0)
                throw new InvalidDataException($"The platform audio decoder returned no samples for '{path}'.");

            int channels = raw.ChannelCount;
            int totalFrames = raw.SampleCount;
            foreach (float[] channel in raw.Channels)
            {
                if (channel == null || channel.Length != totalFrames)
                    throw new InvalidDataException("The platform audio decoder returned inconsistent channel lengths.");
                ValidateSamples(channel);
            }
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                double sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += raw.Channels[ch][i];
                mono[i] = (float)(sum / channels);
            }

            if (raw.SampleRate != SampleRate)
                mono = Gemma4AudioResampler.Resample(mono, raw.SampleRate, SampleRate);
            return mono;
        }

        private static float[] DecodeMp3(string path)
        {
            using var stream = File.OpenRead(path);
            var reader = new MpegFile(stream);
            int mpegSampleRate = reader.SampleRate;
            int channels = reader.Channels;

            var allSamples = new System.Collections.Generic.List<float>();
            float[] readBuf = new float[4096];
            int samplesRead;
            while ((samplesRead = reader.ReadSamples(readBuf, 0, readBuf.Length)) > 0)
            {
                for (int i = 0; i < samplesRead; i++)
                    allSamples.Add(readBuf[i]);
            }

            float[] interleaved = allSamples.ToArray();
            ValidateSamples(interleaved);
            int totalFrames = interleaved.Length / channels;
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += interleaved[i * channels + ch];
                mono[i] = sum / channels;
            }

            if (mpegSampleRate != SampleRate)
                mono = Gemma4AudioResampler.Resample(mono, mpegSampleRate, SampleRate);
            return mono;
        }

        private static float[] DecodeOgg(string path)
        {
            using var vorbis = new VorbisReader(path);
            int oggSampleRate = vorbis.SampleRate;
            int channels = vorbis.Channels;

            var allSamples = new System.Collections.Generic.List<float>();
            float[] readBuf = new float[4096];
            int samplesRead;
            while ((samplesRead = vorbis.ReadSamples(readBuf, 0, readBuf.Length)) > 0)
            {
                for (int i = 0; i < samplesRead; i++)
                    allSamples.Add(readBuf[i]);
            }

            float[] interleaved = allSamples.ToArray();
            ValidateSamples(interleaved);
            int totalFrames = interleaved.Length / channels;
            float[] mono = new float[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += interleaved[i * channels + ch];
                mono[i] = sum / channels;
            }

            if (oggSampleRate != SampleRate)
                mono = Gemma4AudioResampler.Resample(mono, oggSampleRate, SampleRate);
            return mono;
        }

        public static float[] DecodeWAV(byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length < 12 || System.Text.Encoding.ASCII.GetString(data, 0, 4) != "RIFF" ||
                System.Text.Encoding.ASCII.GetString(data, 8, 4) != "WAVE")
                throw new InvalidDataException("Not a WAV file.");
            uint riffLength = BitConverter.ToUInt32(data, 4);
            if (riffLength < 4 || riffLength > data.Length - 8)
                throw new InvalidDataException("WAV RIFF length exceeds the available data.");
            int riffEnd = checked((int)riffLength + 8);

            ushort audioFormat = 0;
            int numChannels = 0, sampleRate = 0, bitsPerSample = 0, blockAlign = 0;
            byte[] audioData = null;
            bool foundFmt = false;

            int offset = 12;
            while (offset < riffEnd)
            {
                if (riffEnd - offset < 8) throw new InvalidDataException("WAV chunk header is truncated.");
                string chunkId = System.Text.Encoding.ASCII.GetString(data, offset, 4);
                uint declaredSize = BitConverter.ToUInt32(data, offset + 4);
                if (declaredSize > (uint)(riffEnd - offset - 8))
                    throw new InvalidDataException("WAV chunk length exceeds the available data.");
                int chunkSize = checked((int)declaredSize);
                int chunkEnd = offset + 8 + chunkSize;

                if (chunkId == "fmt ")
                {
                    if (chunkSize < 16) throw new InvalidDataException("WAV format chunk is truncated.");
                    audioFormat = BitConverter.ToUInt16(data, offset + 8);
                    numChannels = BitConverter.ToUInt16(data, offset + 10);
                    sampleRate = BitConverter.ToInt32(data, offset + 12);
                    blockAlign = BitConverter.ToUInt16(data, offset + 20);
                    bitsPerSample = BitConverter.ToUInt16(data, offset + 22);
                    if (audioFormat == 0xFFFE)
                    {
                        if (chunkSize < 40 || BitConverter.ToUInt16(data, offset + 24) < 22)
                            throw new InvalidDataException("WAV extensible format chunk is truncated.");
                        var subtype = new Guid(data.AsSpan(offset + 32, 16));
                        if (subtype != new Guid("00000001-0000-0010-8000-00aa00389b71") &&
                            subtype != new Guid("00000003-0000-0010-8000-00aa00389b71"))
                            throw new NotSupportedException($"Unsupported WAV extensible subtype: {subtype}.");
                        audioFormat = BitConverter.ToUInt16(data, offset + 32);
                    }
                    foundFmt = true;
                }
                else if (chunkId == "data")
                {
                    audioData = new byte[chunkEnd - offset - 8];
                    Array.Copy(data, offset + 8, audioData, 0, audioData.Length);
                }

                if ((chunkSize & 1) != 0 && chunkEnd == riffEnd)
                    throw new InvalidDataException("WAV odd-length chunk is missing its padding byte.");
                offset = chunkEnd + (chunkSize & 1);
            }

            if (!foundFmt) throw new InvalidDataException("No format chunk in WAV.");
            if (audioFormat != 1 && audioFormat != 3)
                throw new NotSupportedException($"Unsupported WAV format: {audioFormat}.");
            if (sampleRate <= 0 || numChannels <= 0 ||
                (audioFormat == 1 && bitsPerSample is not (8 or 16 or 24 or 32)) ||
                (audioFormat == 3 && bitsPerSample != 32) || blockAlign != numChannels * (bitsPerSample / 8))
                throw new InvalidDataException("WAV sample rate, channels, bit depth or block alignment is invalid.");
            if (audioData == null || audioData.Length == 0 || audioData.Length % blockAlign != 0)
                throw new InvalidDataException("WAV data chunk is missing, empty or truncated.");

            float[] samples = DecodeSamples(audioData, audioFormat, bitsPerSample, numChannels);
            ValidateSamples(samples);
            if (sampleRate != SampleRate)
                samples = Gemma4AudioResampler.Resample(samples, sampleRate, SampleRate);

            return samples;
        }

        private static float[] DecodeSamples(byte[] data, ushort format, int bits, int channels)
        {
            int bytesPerSample = bits / 8;
            int totalSamples = data.Length / (bytesPerSample * channels);
            float[] mono = new float[totalSamples];

            for (int i = 0; i < totalSamples; i++)
            {
                double sum = 0;
                for (int ch = 0; ch < channels; ch++)
                {
                    int off = (i * channels + ch) * bytesPerSample;
                    if (off + bytesPerSample > data.Length) break;

                    if (format == 1 && bits == 16)
                        sum += BitConverter.ToInt16(data, off) / 32768.0;
                    else if (format == 1 && bits == 32)
                        sum += BitConverter.ToInt32(data, off) / 2147483648.0;
                    else if (format == 1 && bits == 24)
                    {
                        int v = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16);
                        if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                        sum += v / 8388608.0;
                    }
                    else if (format == 3 && bits == 32)
                        sum += BitConverter.ToSingle(data, off);
                    else if (format == 1 && bits == 8)
                        sum += (data[off] - 128.0) / 128.0;
                }
                mono[i] = (float)(sum / channels);
            }
            return mono;
        }

        public static (float[] melData, int numFrames) ComputeMelSpectrogram(float[] samples)
            => ComputeMelSpectrogram(samples, Gemma4AudioLogPolicy.HuggingFaceAddFloor);

        public static (float[] melData, int numFrames) ComputeMelSpectrogram(float[] samples, Gemma4AudioLogPolicy policy)
        {
            ArgumentNullException.ThrowIfNull(samples);
            if (!Enum.IsDefined(policy)) throw new ArgumentOutOfRangeException(nameof(policy));
            ValidateSamples(samples);
            int numFrames = ComputeMelFrameCount(samples.Length);
            if (numFrames <= 0) return (null, 0);

            float[] result = new float[numFrames * MelBins];
            if (numFrames < 8)
            {
                var fftInput = new Complex[FftLength];
                for (int f = 0; f < numFrames; f++)
                    ComputeMelFrame(samples, f, fftInput, result, policy);
            }
            else
            {
                Parallel.For(0, numFrames,
                    () => new Complex[FftLength],
                    (f, _, fftInput) =>
                    {
                        ComputeMelFrame(samples, f, fftInput, result, policy);
                        return fftInput;
                    },
                    _ => { });
            }

            return (result, numFrames);
        }

        private static void ValidateSamples(float[] samples)
        {
            foreach (float sample in samples)
                if (!float.IsFinite(sample)) throw new InvalidDataException("Audio contains a non-finite sample.");
        }

        private static MelFilter[] BuildMelFilterBank(int numFreqBins, int numMels, double fMin, double fMax, int sr)
        {
            double HzToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
            double MelToHz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);

            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);

            double[] melPts = new double[numMels + 2];
            for (int i = 0; i < melPts.Length; i++)
                melPts[i] = melMin + (double)i * (melMax - melMin) / (numMels + 1);
            double[] filterFreqs = new double[numMels + 2];
            for (int i = 0; i < melPts.Length; i++)
                filterFreqs[i] = MelToHz(melPts[i]);

            double[] fftFreqs = new double[numFreqBins];
            for (int i = 0; i < numFreqBins; i++)
                fftFreqs[i] = (double)i * sr / (2.0 * (numFreqBins - 1));

            var filters = new MelFilter[numMels];
            for (int m = 0; m < numMels; m++)
            {
                double fLeft = filterFreqs[m];
                double fCenter = filterFreqs[m + 1];
                double fRight = filterFreqs[m + 2];
                var weights = new double[numFreqBins];
                int first = numFreqBins, last = -1;
                for (int k = 0; k < numFreqBins; k++)
                {
                    double f = fftFreqs[k];
                    double v = 0;
                    if (f >= fLeft && f <= fCenter && fCenter > fLeft)
                        v = (f - fLeft) / (fCenter - fLeft);
                    else if (f > fCenter && f <= fRight && fRight > fCenter)
                        v = (fRight - f) / (fRight - fCenter);
                    if (v > 0)
                    {
                        weights[k] = v;
                        first = Math.Min(first, k);
                        last = k;
                    }
                }
                filters[m] = last < 0 ? new MelFilter(0, Array.Empty<double>()) : new MelFilter(first, weights[first..(last + 1)]);
            }
            return filters;
        }

        private static void FFT(Complex[] x)
        {
            int n = x.Length;
            if (n <= 1) return;

            int j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
                j ^= bit;
                if (i < j) { var tmp = x[i]; x[i] = x[j]; x[j] = tmp; }
            }

            int stage = 0;
            for (int size = 2; size <= n; size <<= 1, ++stage)
            {
                int halfSize = size / 2;
                Complex[] twiddles = FftTwiddles[stage];
                for (int start = 0; start < n; start += size)
                {
                    for (int k = 0; k < halfSize; k++)
                    {
                        Complex t = twiddles[k] * x[start + k + halfSize];
                        x[start + k + halfSize] = x[start + k] - t;
                        x[start + k] = x[start + k] + t;
                    }
                }
            }
        }

        private static Complex[][] BuildFftTwiddles()
        {
            var stages = new System.Collections.Generic.List<Complex[]>();
            for (int size = 2; size <= FftLength; size <<= 1)
            {
                var values = new Complex[size / 2];
                Complex step = new Complex(Math.Cos(2 * Math.PI / size), -Math.Sin(2 * Math.PI / size));
                Complex value = Complex.One;
                for (int k = 0; k < values.Length; ++k)
                {
                    values[k] = value;
                    value *= step;
                }
                stages.Add(values);
            }
            return stages.ToArray();
        }

        public static int ComputeAudioTokenCount(float[] samples, bool padToMultiple = true)
        {
            ArgumentNullException.ThrowIfNull(samples);
            int paddedLength = padToMultiple ? checked((samples.Length + 127) / 128 * 128) : samples.Length;
            int numFrames = ComputeMelFrameCount(paddedLength);
            if (numFrames <= 0) return 0;

            int tConv0 = (numFrames + 2 - 3) / 2 + 1;
            int tConv1 = (tConv0 + 2 - 3) / 2 + 1;
            return tConv1;
        }

        private static int ComputeMelFrameCount(int samples)
        {
            // HF unfolds frame_length+1 samples after prepending half a frame
            // of zeros, then drops the last sample before the FFT.
            int minimum = FrameLength + 1 - FrameLength / 2;
            return samples < minimum ? 0 : (samples - minimum) / HopLength + 1;
        }

        private static void ComputeMelFrame(float[] samples, int frameIndex, Complex[] fftInput, float[] result, Gemma4AudioLogPolicy policy)
        {
            int start = frameIndex * HopLength - FrameLength / 2;
            for (int i = 0; i < FrameLength; i++)
            {
                int sample = start + i;
                float value = sample < 0 || sample >= samples.Length ? 0 : samples[sample];
                fftInput[i] = new Complex(value * (float)HannWindow[i], 0);
            }
            for (int i = FrameLength; i < FftLength; i++)
                fftInput[i] = Complex.Zero;

            FFT(fftInput);

            // Every mel filter reads the same FFT magnitudes. Compute each
            // double-precision magnitude once without changing the filter
            // summation order or the HF logarithm policy.
            Span<double> magnitudes = stackalloc double[NumFreqBins];
            for (int k = 0; k < NumFreqBins; ++k) magnitudes[k] = fftInput[k].Magnitude;
            int dstOffset = frameIndex * MelBins;
            for (int m = 0; m < MelBins; m++)
            {
                double melVal = 0;
                var filter = MelFilters[m];
                for (int k = 0; k < filter.Weights.Length; k++)
                    melVal += magnitudes[filter.FirstBin + k] * filter.Weights[k];
                result[dstOffset + m] = (float)Math.Log(policy == Gemma4AudioLogPolicy.LlamaClampFloor
                    ? Math.Max(melVal, MelFloor) : melVal + MelFloor);
            }
        }

        private static int ComputeFftLength()
        {
            int fftLen = 1;
            while (fftLen < FrameLength) fftLen <<= 1;
            return fftLen;
        }

        private static double[] BuildWindow()
        {
            double[] window = new double[FrameLength];
            double arg = Math.PI * 2.0 / FrameLength;
            for (int i = 0; i < FrameLength; i++)
                window[i] = (float)(0.5 - 0.5 * Math.Cos(arg * i));
            return window;
        }
    }
}

