#!/usr/bin/env python3
"""Independent complete F32 video/mixed-media oracle for frozen controlled pairs.

Uses actual frozen-assembly input evidence, independently checks identity pixels
and FFmpeg video decode, reuses hash-bound independent image/audio soft rows,
and executes fresh full vision/text math. Never repairs a failed paired report.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def artifact(parent, relative):
    value = Path(relative)
    if value.is_absolute() or not (parent / value).resolve().is_relative_to(parent.resolve()):
        raise ValueError("Input proof buffers must be relative to their evidence directory")
    return parent / value


def binary_hash(pair, name):
    hashes = [row["sha256"] for row in pair["binary_artifacts"]["tensorsharp"] if Path(row["path"]).name == name]
    if len(hashes) != 1:
        raise ValueError("Require exactly one actual " + name + " identity")
    return hashes[0]


def bind_pair(pair, proof, model_hash, projector_hash):
    if (not pair.get("completed") or not pair.get("binary_artifacts_unchanged") or
            not pair.get("dependencies_unchanged_and_clean") or not proof.get("completed") or
            pair["model"]["sha256"] != model_hash or pair["projector"]["sha256"] != projector_hash or
            proof["model_sha256"] != model_hash or
            "TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp" not in pair["engines"]["tensorsharp"]["command"] or
            binary_hash(pair, "TensorSharp.Models.dll") != proof["models_assembly_sha256"] or
            binary_hash(pair, "TensorSharp.Runtime.dll") != proof["runtime_assembly_sha256"] or
            ("Video:" if pair["controlled_video_prefix"] else "") != proof["video_prefix"] or
            proof["video_soft_tokens"] != 140 or proof["include_video_timestamps"] or
            proof["video_frames_per_second"] != 1 or proof["max_video_frames"] != 32):
        raise ValueError("Frozen pair, preprocessing assembly or controlled input policy differs")
    for engine in ("tensorsharp", "llama"):
        if not pair["engines"][engine].get("completed") or pair["engines"][engine].get("server_exit_code") != 0:
            raise ValueError("Require actual cleanly completed engines")


def identity_pixels(record, proof_parent):
    width, height = record["source_width"], record["source_height"]
    if (width <= 0 or height <= 0 or width % 48 or height % 48 or
            (width, height) != (record["processed_width"], record["processed_height"]) or
            record["soft_tokens"] != width // 48 * (height // 48)):
        raise ValueError("This controlled oracle requires identity-size complete pooling grids")
    rgba_path, pixel_path = [artifact(proof_parent, record[key]) for key in ("rgba", "pixels")]
    if sha(rgba_path) != record["rgba_sha256"] or sha(pixel_path) != record["pixels_sha256"]:
        raise ValueError("Actual decoded/preprocessed evidence changed")
    rgba = np.fromfile(rgba_path, dtype=np.uint8).reshape(height, width, 4)
    pixels = np.fromfile(pixel_path, dtype="<f4").reshape(3, height, width)
    expected = rgba[:, :, :3].transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    if not np.all(rgba[:, :, 3] == 255) or not np.array_equal(pixels.view(np.uint32), expected.view(np.uint32)):
        raise ValueError("Actual opaque identity-size pixels differ from independent RGBA normalization")
    return rgba, pixels


def validate_video_indices(video):
    count, fps = video["frame_count"], video["fps"]
    if type(count) is not int or count <= 0 or count > 2147483647 or type(fps) not in (int,float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Require finite positive source FPS and a valid frame count")
    sampled = count / fps
    if not math.isfinite(sampled) or sampled > 2147483647:
        raise ValueError("Video duration exceeds the supported sampling range")
    samples=max(1,math.floor(sampled));selected=min(samples,32)
    expected=[]
    for i in range(selected):
        sample=i if samples<=32 else (0 if selected==1 else math.floor(i*(samples-1)/(selected-1)))
        expected.append(min(count-1,math.floor(sample*fps)))
    actual=video["sampled_indices"]
    if (not isinstance(actual,list) or any(type(index) is not int or index<0 or index>=count for index in actual) or
            actual!=expected or [row["index"] for row in video["frames"]]!=actual):
        raise ValueError("Observed frame indices differ from independent 1-FPS/max32 sampling")


def decoder_format(layout):
    formats={"Bgr":("bgr24",3,True),"Rgb":("rgb24",3,False),"Bgra":("bgra",4,True),"Rgba":("rgba",4,False)}
    if layout not in formats:
        raise ValueError("Unsupported observed decoder channel layout")
    return formats[layout]


def decoded_rgba(raw, count, height, width, layout):
    _,channels,bgr=decoder_format(layout)
    decoded=np.frombuffer(raw,dtype=np.uint8).reshape(count,height,width,channels)
    rgba=np.full((count,height,width,4),255,dtype=np.uint8)
    rgba[:,:,:,:3]=decoded[:,:,:,:3][:,:,:,::-1] if bgr else decoded[:,:,:,:3]
    return rgba


def compose(parts, token_ids, texts, image_rows, audio_rows, video_rows, video_prefix, token_rows):
    blocks = [token_rows([token_ids["bos"]])]
    pending = ""

    def text(value):
        if value:
            if value not in texts:
                raise ValueError("Missing observed exact text-fragment token IDs")
            blocks.append(token_rows(texts[value]))

    def media(rows, kind):
        blocks.extend((token_rows([token_ids[kind + "_start"]]), rows, token_rows([token_ids[kind + "_end"]])))

    for part in parts:
        kind, value = part["kind"], part["value"]
        if kind == "text":
            pending += value
            continue
        if kind == "video":
            pending += video_prefix
        text(pending)
        pending = ""
        if kind == "image":
            media(image_rows[value], "image")
        elif kind == "audio":
            media(audio_rows[value], "audio")
        elif kind == "video":
            for rows in video_rows[value]:
                media(rows, "image")
        else:
            raise ValueError("Unsupported input part")
    text(pending)
    blocks.append(token_rows([token_ids["eos"]]))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def controlled_parts(parts, prefix):
    result = []
    for part in parts:
        if part["kind"] == "video" and prefix:
            result.append(dict(kind="text", value=prefix))
        result.append(part)
    return result


def compare(expected, row, parts, count):
    value = np.asarray(row["embedding"], dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if (row["parts"] != parts or row["prompt_tokens"] != count or value.shape != expected.shape or
            not np.all(np.isfinite(value)) or abs(float(np.linalg.norm(value)) - 1) > 1e-5):
        raise ValueError("Actual complete embedding or input accounting differs")
    cosine = float(expected @ value / (np.linalg.norm(expected) * np.linalg.norm(value)))
    error = float(np.max(np.abs(expected - value)))
    return dict(cosine=cosine, max_absolute_error=error, passed=cosine >= .9999 and error <= .002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "projector", "image-oracle", "audio-oracle", "input-proof", "media-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pair", type=Path, action="append", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing oracle evidence; choose a new output report")
    root = Path(__file__).resolve().parents[2]
    vision_module = module("composite_vision", root / "eng/validation/embeddinggemma2-vision-reference.py")
    text_module = module("composite_text", root / "eng/embeddinggemma2-reference.py")
    image_module = module("composite_image", root / "eng/validation/embeddinggemma2-controlled-image-reference.py")
    proof = json.loads(args.input_proof.read_text())
    image_oracle, audio_oracle = [json.loads(path.read_text()) for path in (args.image_oracle, args.audio_oracle)]
    model_hash, projector_hash = sha(args.model), sha(args.projector)
    pairs = [(path, json.loads(path.read_text())) for path in args.pair]
    for _, pair in pairs:
        bind_pair(pair, proof, model_hash, projector_hash)
    if not image_oracle["completed"] or image_oracle["model_sha256"] != model_hash or image_oracle["projector_sha256"] != projector_hash:
        raise ValueError("Independent image weights changed")
    for relative in ("eng/validation/embeddinggemma2-vision-reference.py", "eng/embeddinggemma2-reference.py", "eng/diffusiongemma-vision-oracle.py"):
        if image_oracle["source_hashes"][relative] != sha(root / relative):
            raise ValueError("Preserved image/text graph arithmetic changed")
    if (audio_oracle["projector_sha256"] != projector_hash or audio_oracle["audio_log_policy"] != "llama_clamp" or
            audio_oracle["left_window_size"] != 12 or audio_oracle["past_distance_inclusive"] != 11):
        raise ValueError("Preserved independent controlled-audio policy differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, passed=False, tensorsharp_passed=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Observed frozen input pipeline; independent PNG/FFmpeg decode and F32 pixels, preserved independent soft image/audio rows, fresh full F32 video vision and complete text model",
        thresholds=dict(min_cosine=.9999, max_absolute_error=.002), model_sha256=model_hash, projector_sha256=projector_hash,
        numpy=np.__version__,python=sys.version,
        inputs={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in
                (("proof",args.input_proof),("image_oracle",args.image_oracle),("audio_oracle",args.audio_oracle))},
        source_hashes={str(path.relative_to(root)):sha(path) for path in (Path(__file__),root/"eng/validation/embeddinggemma2-vision-reference.py",root/"eng/embeddinggemma2-reference.py",root/"eng/diffusiongemma-vision-oracle.py")},
        fresh_unique_video_frames=0, fresh_complete_text_cases=0, images=[],videos=[],cases=[],
        limitations=["Only controlled recorded synthetic inputs and original Q8 model/projector; no leaderboard or default-HF/llama pipeline equivalence claim.",
            "Text-fragment token IDs are observed from the production tokenizer; their independent tokenizer regression is separate.",
            "Image/audio learned soft rows reuse preserved independent math; all composite text-model outputs and unique video-frame vision rows are computed fresh.",
            "Original cross-engine/latency failures remain unchanged. Comparison to llama uses independently verified TensorSharp decoded inputs; llama's own decoded frame bytes are not directly observed."])
    def save():
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    try:
        vision, text = vision_module.VisionEncoder(args.projector), text_module.Encoder(args.model)
        vocabulary = text.metadata("tokenizer.ggml.tokens")
        ids = dict(bos=int(text.metadata("tokenizer.ggml.bos_token_id")),eos=int(text.metadata("tokenizer.ggml.eos_token_id")),
            image_start=vocabulary.index("<|image>"),image_end=vocabulary.index("<image|>"),
            audio_start=vocabulary.index("<|audio>"),audio_end=vocabulary.index("<audio|>"))
        if proof["token_ids"] != ids:
            raise ValueError("Observed delimiter IDs differ from original GGUF metadata")
        texts = {row["text"]:row["ids"] for row in proof["texts"]}
        if len(texts) != len(proof["texts"]) or any(not all(isinstance(token,int) and 0 <= token < len(vocabulary) for token in row) for row in texts.values()):
            raise ValueError("Observed text token proof is malformed")
        media_hashes = {row["path"]:row["sha256"] for row in pairs[0][1]["media"]}
        for _, pair in pairs[1:]:
            if {row["path"]:row["sha256"] for row in pair["media"]} != media_hashes:
                raise ValueError("Controlled pair media identities differ")
        image_rows, audio_rows, video_rows = {}, {}, {}
        for record in proof["images"]:
            local = args.media_root / Path(record["path"]).name
            if sha(local) != record["sha256"] or media_hashes[record["path"]] != record["sha256"]:
                raise ValueError("Observed image source changed")
            rgba, _ = identity_pixels(record,args.input_proof.parent)
            if vision.hf.get_aspect_ratio_preserving_size(record["source_height"],record["source_width"],16,280*9,3) != (record["source_height"],record["source_width"]):
                raise ValueError("Observed image geometry differs from independent HF policy")
            with Image.open(local) as decoded:
                if decoded.format != "PNG" or not np.array_equal(np.asarray(decoded.convert("RGBA")),rgba):
                    raise ValueError("Independent PNG decoder differs")
            prior = next(row for row in image_oracle["cases"] if row["preprocessing_proofs"][0]["image_sha256"] == record["sha256"])
            if image_module.reused_pixel_hash(prior) != record["pixels_sha256"]:
                raise ValueError("Preserved independent image pixels differ")
            soft = args.image_oracle.parent / (prior["name"]+"-soft.f32")
            if sha(soft) != prior["soft_rows_sha256"]:
                raise ValueError("Preserved independent image rows changed")
            image_rows[record["path"]] = np.fromfile(soft,dtype="<f4").reshape(prior["soft_shape"])
            if image_rows[record["path"]].shape != (record["soft_tokens"],text.dim) or not np.all(np.isfinite(image_rows[record["path"]])):
                raise ValueError("Preserved independent image rows are invalid")
            report["images"].append(dict(path=record["path"],sha256=record["sha256"],pixels_sha256=record["pixels_sha256"],soft_rows_sha256=sha(soft),reused_independent_math=True))
        audio = np.asarray(audio_oracle["soft_rows"],dtype=np.float32)
        if list(audio.shape) != audio_oracle["shape"] or audio.ndim != 2 or audio.shape[1] != text.dim or not np.all(np.isfinite(audio)):
            raise ValueError("Preserved independent audio rows are invalid")
        for record in proof["audios"]:
            local = args.media_root / Path(record["path"]).name
            if sha(local) != record["sha256"] or media_hashes[record["path"]] != record["sha256"] or record["sha256"] != audio_oracle["waveform_sha256"]:
                raise ValueError("Preserved canonical waveform differs")
            audio_rows[record["path"]] = audio
        unique_frames = {}
        ffmpeg_version = subprocess.run([args.ffmpeg,"-version"],check=True,capture_output=True,text=True).stdout
        report["ffmpeg_version"] = ffmpeg_version
        ffmpeg_path=Path(shutil.which(args.ffmpeg) or args.ffmpeg).resolve()
        report["ffmpeg_binary"]=dict(path=str(ffmpeg_path),sha256=sha(ffmpeg_path))
        for video in proof["videos"]:
            validate_video_indices(video)
            local = args.media_root / Path(video["path"]).name
            if sha(local) != video["sha256"] or media_hashes[video["path"]] != video["sha256"]:
                raise ValueError("Observed clip changed")
            frames = video["frames"]
            width,height=frames[0]["source_width"],frames[0]["source_height"]
            layout=frames[0]["decoded_layout"]
            if any(frame["decoded_layout"]!=layout for frame in frames):
                raise ValueError("Require a consistent observed decoder channel layout")
            pixel_format=decoder_format(layout)[0]
            # Match the observed decoder's channel format before the separate
            # RGBA conversion. FFmpeg's RGB24 and RGBA conversion kernels can
            # round YUV colors differently even for the same encoded frame.
            raw = subprocess.run([args.ffmpeg,"-v","error","-i",str(local),"-f","rawvideo","-pix_fmt",pixel_format,"-vsync","0","pipe:1"],check=True,capture_output=True).stdout
            decoded=decoded_rgba(raw,video["frame_count"],height,width,layout)
            rows=[]; observations=[]
            for frame in frames:
                rgba,pixels=identity_pixels(frame,args.input_proof.parent)
                if vision.hf.get_aspect_ratio_preserving_size(frame["source_height"],frame["source_width"],16,140*9,3) != (frame["source_height"],frame["source_width"]):
                    raise ValueError("Observed video geometry differs from independent HF policy")
                if not np.array_equal(decoded[frame["index"]],rgba):
                    raise ValueError("Independent FFmpeg and actual production decoded frame bytes differ")
                key=frame["pixels_sha256"]
                if key not in unique_frames:
                    started=time.perf_counter(); soft=vision.encode(pixels)
                    if soft.shape != (frame["soft_tokens"],text.dim) or not np.all(np.isfinite(soft)):
                        raise ValueError("Independent video vision rows differ from observed geometry")
                    path=args.output.parent/("frame-"+key[:16]+"-soft.f32");soft.astype("<f4").tofile(path)
                    unique_frames[key]=(soft,dict(pixels_sha256=key,soft_rows_sha256=sha(path),seconds=time.perf_counter()-started))
                    report["fresh_unique_video_frames"]+=1
                rows.append(unique_frames[key][0]);observations.append(dict(index=frame["index"],independent_decoded_rgba_equal=True,**unique_frames[key][1]))
            video_rows[video["path"]]=rows;report["videos"].append(dict(path=video["path"],sha256=video["sha256"],ffmpeg_pixel_format=pixel_format,frames=observations))
            save()
        names=("video","mixed","all-modalities")
        for name in names:
            first=next(row for row in pairs[0][1]["inputs"] if row["name"]==name)
            effective_parts=controlled_parts(first["parts"],proof["video_prefix"])
            inputs=compose(effective_parts,ids,texts,image_rows,audio_rows,video_rows,"",text.token_rows)
            expected=text.encode_inputs(inputs);report["fresh_complete_text_cases"]+=1
            if expected.shape!=(768,) or not np.all(np.isfinite(expected)) or abs(float(np.linalg.norm(expected))-1)>1e-5:
                raise ValueError("Independent full composite embedding is invalid")
            output=args.output.parent/(name+"-inputs.f32");inputs.tofile(output)
            comparisons=[]
            for path,pair in pairs:
                case=next(row for row in pair["inputs"] if row["name"]==name)
                if case!=first: raise ValueError("Pair composite sequence differs")
                engines={engine:compare(expected,pair["engines"][engine]["benchmarks"][name],
                    effective_parts if engine == "tensorsharp" else case["parts"],len(inputs)) for engine in ("tensorsharp","llama")}
                comparisons.append(dict(pair=str(path.resolve()),pair_sha256=sha(path),backend=pair["expected_backend"],engines=engines,original_pair_passed=pair["passed"],original_quality_pass=pair["quality_pass"],original_performance_pass=pair["performance_pass"]))
            report["cases"].append(dict(name=name,total_input_rows=len(inputs),input_rows_sha256=sha(output),embedding=expected.tolist(),comparisons=comparisons));save()
        report["tensorsharp_passed"]=all(row["engines"]["tensorsharp"]["passed"] for case in report["cases"] for row in case["comparisons"])
        report["passed"]=all(value["passed"] for case in report["cases"] for row in case["comparisons"] for value in row["engines"].values())
        report["completed"]=True
        return 0 if report["passed"] else 1
    finally:
        save()


if __name__=="__main__":
    raise SystemExit(main())
