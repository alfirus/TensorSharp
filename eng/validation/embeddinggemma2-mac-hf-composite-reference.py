#!/usr/bin/env python3
"""Mac-specific complete default-HF learned reference.

Uses a hash-bound, separately executed AVAssetReader application for exact video
RGBA, through the same Apple codec. Original FFmpeg equality remains unqualified.
All original PNG/resize/complete F32 vision-text/protocol numerical gates remain.
Derived from unchanged default-HF reference d83acc0e; arithmetic sources unchanged.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ROOT = Path(__file__).resolve().parents[2]
base = module("hf_composite_base", Path(__file__).with_name("embeddinggemma2-controlled-composite-reference.py"))
environment = module("hf_composite_environment", Path(__file__).with_name("embeddinggemma2_http_environment.py"))
sha = base.sha
CASE_NAMES = ("text", "red", "pattern", "audio", "video", "mixed", "all-modalities")
SOURCES = ("eng/validation/embeddinggemma2-controlled-composite-reference.py",
    "eng/validation/embeddinggemma2-vision-reference.py", "eng/embeddinggemma2-reference.py",
    "eng/diffusiongemma-vision-oracle.py")


def binary_hash(report, name):
    values = [row["sha256"] for row in report["binary_artifacts"] if Path(row["path"]).name == name]
    if len(values) != 1:
        raise ValueError("Require exactly one loaded " + name + " identity")
    return values[0]


def validate_http(report, proof, model_hash, projector_hash):
    audit = report.get("environment_audit", {})
    flags = audit.get("optimization_flags", {})
    sources = audit.get("optimization_flag_sources", {})
    if (not report.get("completed") or not report.get("passed") or report.get("server_exit_code") != 0 or
            report.get("require_embedding_default") is not True or report.get("audio_log_policy") != "HuggingFaceAddFloor" or
            audit.get("require_embedding_default") is not True or audit.get("resolved_server_environment") is not True or
            audit.get("scope") != "direct host child environment" or audit.get("audio_log_policy") != "HuggingFaceAddFloor" or
            report.get("binary_artifacts_unchanged", True) is not True or
            set(flags) != set(environment.OPTIMIZATION_FLAGS) or any(value is not None for value in flags.values()) or
            set(sources) != set(flags) or any(value is not None for value in sources.values())):
        raise ValueError("Require completed clean default-HF HTTP with every arithmetic/instrumentation selector absent")
    if (report["model"]["sha256"] != model_hash or report["projector"]["sha256"] != projector_hash or
            binary_hash(report, "TensorSharp.Models.dll") != proof["models_assembly_sha256"] or
            binary_hash(report, "TensorSharp.Runtime.dll") != proof["runtime_assembly_sha256"]):
        raise ValueError("HTTP model/projector or loaded preprocessing assemblies differ")
    if tuple(row["name"] for row in report["cases"]) != CASE_NAMES:
        raise ValueError("Require all seven original default-HF input sequences")
    paths = {row["path"]: row["sha256"] for row in report["media"]}
    if len(paths) != len(report["media"]):
        raise ValueError("Duplicate HTTP media identity")
    for row in report["cases"]:
        if not row["parts"] or any(part["kind"] not in ("text", "image", "audio", "video") for part in row["parts"]):
            raise ValueError("Malformed HTTP ordered input parts")
        if any(part["value"] not in paths for part in row["parts"] if part["kind"] != "text"):
            raise ValueError("HTTP media identity missing")


def validate_proof(proof):
    defaults = proof.get("default_options_observed", {})
    expected_defaults = dict(video_prefix="",include_video_timestamps=False,image_soft_tokens=280,
        video_soft_tokens=140,video_frames_per_second=1,max_video_frames=32,audio_log_policy="HuggingFaceAddFloor")
    if (not proof.get("completed") or proof.get("source_report_kind") != "default_hf_http" or
            proof["controlled_video_prefix"] or proof["video_prefix"] != "" or proof["include_video_timestamps"] or
            proof["image_soft_tokens"] != 280 or not proof["image_reference_sizing"] or
            proof["video_soft_tokens"] != 140 or proof["video_frames_per_second"] != 1 or proof["max_video_frames"] != 32 or
            proof.get("audio_log_policy") != "HuggingFaceAddFloor" or
            any(defaults.get(key) != value for key,value in expected_defaults.items()) or
            defaults.get("type") != "TensorSharp.Models.Embeddings.EmbeddingGemma2MultimodalOptions"):
        raise ValueError("Require fresh actual default-HF input proof with original sizing/sampling policy")


def validate_runtime_guards(report, guards):
    for guard in guards:
        if (guard.get("artifacts_unchanged") is not True or not guard.get("before") or
                guard["before"] != guard.get("after")):
            raise ValueError("Require successful complete before/after frozen runtime guards")
    if not any(all(guard["before"].get(row["path"]) == row["sha256"] for row in report["binary_artifacts"])
            for guard in guards):
        raise ValueError("Actual HTTP binary identities are not bound to one unchanged runtime snapshot")


def validate_pixels(record, parent, independent_rgba, hf, max_error=2e-6):
    width, height = record["source_width"], record["source_height"]
    out_w, out_h = record["processed_width"], record["processed_height"]
    if min(width, height, out_w, out_h) <= 0 or out_w % 48 or out_h % 48:
        raise ValueError("Invalid complete pooling canvas")
    paths = [base.artifact(parent, record[key]) for key in ("rgba", "pixels")]
    if any(sha(path) != record[key + "_sha256"] for path, key in zip(paths, ("rgba", "pixels"))):
        raise ValueError("Observed pixel artifact changed")
    rgba = np.fromfile(paths[0], dtype=np.uint8).reshape(height, width, 4)
    actual = np.fromfile(paths[1], dtype="<f4").reshape(3, out_h, out_w)
    if not np.array_equal(rgba, independent_rgba) or not np.all(rgba[:, :, 3] == 255):
        raise ValueError("Independent decoded opaque RGBA bytes differ")
    if record["soft_tokens"] != out_w // 48 * (out_h // 48):
        raise ValueError("Observed soft-token count differs from canvas")
    expected = hf._resize_chw(rgba[:, :, :3].transpose(2, 0, 1), out_h, out_w, mode="float")
    expected = (expected * np.float32(1.0 / 255.0)).astype(np.float32)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("Nonfinite resize pixels")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if error > max_error:
        raise ValueError("Independent float bicubic differs: max error " + str(error))
    return expected, dict(pixels_sha256=record["pixels_sha256"],
        independent_pixels_sha256=hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
        max_absolute_error=error, max_allowed_error=max_error, decoded_rgba_equal=True)


def compare(expected, row, count):
    if row["tokens"] != count:
        raise ValueError("Complete default-HF token accounting differs")
    result = {}
    for protocol, key in (("openai", "embedding"), ("ollama", "ollama_embedding")):
        actual = np.asarray(row[key], dtype=np.float64)
        if actual.shape != (768,) or not np.all(np.isfinite(actual)) or abs(float(np.linalg.norm(actual)) - 1) > 1e-5:
            raise ValueError("Actual complete default-HF embedding invalid")
        cosine = float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))
        error = float(np.max(np.abs(expected - actual)))
        result[protocol] = dict(cosine=cosine, max_absolute_error=error, passed=cosine >= .9999 and error <= .002)
    return result


def vision_key(pixels, pixel_hash):
    return "x".join(str(value) for value in pixels.shape) + "-" + pixel_hash


def validate_decoder(decoder, parent, proof, proof_path, read_json):
    """Strict proof chain and full bytes; no relaxed FFmpeg route or observed-pixel oracle."""
    if (decoder.get("completed") is not True or decoder.get("passed") is not True or
            decoder.get("inputs_unchanged") is not True or decoder.get("aliases_unchanged") is not True or
            decoder.get("decode_exit") != 0 or decoder.get("learned_forwards") != 0 or
            decoder.get("independent_codec_implementation") is not False or
            decoder.get("ffmpeg_equivalence_qualified") is not False or
            decoder.get("input_proof_sha256") != sha(proof_path) or
            decoder.get("models_sha256") != proof["models_assembly_sha256"] or
            decoder.get("runtime_sha256") != proof["runtime_assembly_sha256"] or
            decoder.get("codec_sha256") != proof["video_codec_sha256"]):
        raise ValueError("Require exact current Mac same-codec decode proof with unchanged guards")
    paths = set()
    manifests = {}
    for name in ("before.json", "after.json", "aliases-before.json", "aliases-after.json", "generated-files.json"):
        path = parent / name; paths.add(path); manifests[name] = read_json(path)
    before = manifests["before.json"]
    if (not before or before != manifests["after.json"] or
            not manifests["aliases-before.json"] or manifests["aliases-before.json"] != manifests["aliases-after.json"] or
            set(before) != set(manifests["aliases-before.json"])):
        raise ValueError("Mac decoder complete input/alias guards differ")
    for path, expected in before.items():
        path = Path(path)
        if sha(path) != expected or str(path.resolve(strict=True)) != manifests["aliases-before.json"][str(path)]:
            raise ValueError("Mac decoder bound input or alias changed")
        paths.add(path)
    expected_proof = str(proof_path.absolute())
    if before.get(expected_proof) != decoder["input_proof_sha256"]:
        raise ValueError("Fresh Mac input proof missing from decoder guard")
    for path_key, hash_key in (("models_assembly","models_assembly_sha256"),
            ("runtime_assembly","runtime_assembly_sha256"),("video_codec_path","video_codec_sha256")):
        if before.get(proof[path_key]) != proof[hash_key]:
            raise ValueError("Actual Mac assembly/codec identity missing from decoder guard")
    for path, expected in manifests["generated-files.json"].items():
        path = Path(path)
        if not path.resolve().is_relative_to((parent / "decoded").resolve()) or sha(path) != expected:
            raise ValueError("Mac decoder produced artifact changed or escaped output")
        paths.add(path)
    raw_path = parent / "decoded/decode.json"; raw = read_json(raw_path)
    paths.add(raw_path)
    if sha(raw_path) != decoder["decode_report_sha256"] or not raw.get("completed") or not raw.get("passed"):
        raise ValueError("Raw platform decode report differs")
    if (raw.get("decoder") != "AVAssetReaderTrackOutput" or raw.get("requested_format") != "32BGRA" or
            raw.get("model_forwards") != 0 or raw.get("independent_codec_implementation") is not False or
            raw.get("ffmpeg_equivalence_qualified") is not False):
        raise ValueError("Raw platform decoder scope differs")
    if (not np.isfinite(raw.get("nominal_fps", np.nan)) or raw["nominal_fps"] <= 0 or
            type(raw.get("frame_count")) is not int):
        raise ValueError("Raw platform frame metadata is nonfinite or invalid")
    if len(proof["videos"]) != 1:
        raise ValueError("This bounded Mac proof covers exactly one video")
    video = proof["videos"][0]; base.validate_video_indices(video)
    if (raw["video"] != video["path"] or raw["frame_count"] != video["frame_count"] or
            decoder["decoded_frame_count"] != video["frame_count"] or
            abs(raw["nominal_fps"] - video["fps"]) > 1e-6 or
            len(raw["frames"]) != video["frame_count"] or
            decoder["observed_sample_indices"] != video["sampled_indices"] or
            [row["index"] for row in decoder["frames"]] != video["sampled_indices"]):
        raise ValueError("Mac frame count, rate, order or sampling differs")
    decoded = []
    width, height = video["source_width"], video["source_height"]
    for index, frame in enumerate(raw["frames"]):
        path = base.artifact(parent / "decoded", frame["rgba_file"])
        if (frame["index"] != index or (frame["width"],frame["height"]) != (width,height) or
                frame["opaque_alpha"] is not True or frame["rgba_bytes"] != width*height*4 or
                type(frame["pts_timescale"]) is not int or frame["pts_timescale"] <= 0 or
                type(frame["pts_value"]) is not int or not np.isfinite(frame["pts_seconds"]) or
                abs(frame["pts_seconds"] - frame["pts_value"]/frame["pts_timescale"]) > 1e-9 or
                abs(frame["pts_seconds"] - index/video["fps"]) > 1e-6 or
                manifests["generated-files.json"].get(str(path.absolute())) != sha(path)):
            raise ValueError("Mac decoded frame geometry/timing/byte binding differs")
        values = np.fromfile(path,dtype=np.uint8)
        if values.size != width*height*4 or not np.all(values.reshape(height,width,4)[:,:,3] == 255):
            raise ValueError("Mac decoded opaque RGBA span differs")
        decoded.append(values.reshape(height,width,4)); paths.add(path)
    for observed, row in zip(video["frames"], decoder["frames"]):
        actual = base.artifact(proof_path.parent, observed["rgba"])
        independent = base.artifact(parent / "decoded", raw["frames"][observed["index"]]["rgba_file"])
        if (not row["exact_bytes_equal"] or row["maximum_byte_difference"] != 0 or
                row["production_rgba_sha256"] != observed["rgba_sha256"] or
                row["direct_rgba_sha256"] != sha(independent) or sha(actual) != observed["rgba_sha256"] or
                actual.read_bytes() != independent.read_bytes()):
            raise ValueError("Mac complete sampled RGBA bytes differ")
    generated = manifests["generated-files.json"]
    bound = {path: before.get(str(path), generated.get(str(path), sha(path))) for path in paths}
    return {video["path"]: np.stack(decoded)}, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "projector", "input-proof", "media-root", "anchor-oracle", "hf-audio-embedding-oracle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--hf-audio-embedding-oracle-sha256", required=True)
    parser.add_argument("--http", type=Path, action="append", required=True)
    parser.add_argument("--runtime-guard", type=Path, action="append", required=True)
    parser.add_argument("--decoder-proof", type=Path, required=True)
    parser.add_argument("--decoder-proof-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a new output")
    if sha(Path(__file__).with_name("embeddinggemma2-default-hf-composite-reference.py")) != "d83acc0e55848abbc1498d6a9b3f8312d1c172b68d313a4d09d8637d868e769e":
        raise ValueError("Original default-HF reference source changed")
    parsed_hashes = {}
    def read_json(path):
        raw = path.read_bytes()
        key, digest = str(path.resolve()), hashlib.sha256(raw).hexdigest()
        if key in parsed_hashes and parsed_hashes[key] != digest:
            raise ValueError("Parsed evidence changed before guard capture")
        parsed_hashes[key] = digest
        return json.loads(raw)
    if sha(args.anchor_oracle) != args.anchor_sha256 or sha(args.hf_audio_embedding_oracle) != args.hf_audio_embedding_oracle_sha256:
        raise ValueError("Explicit independent anchor hash differs")
    anchor = read_json(args.anchor_oracle)
    if parsed_hashes[str(args.anchor_oracle.resolve())] != args.anchor_sha256:
        raise ValueError("Arithmetic anchor changed before parsing")
    if not anchor.get("completed") or not anchor.get("tensorsharp_passed") or set(anchor["source_hashes"]) != set(SOURCES):
        raise ValueError("Require completed independent F32 arithmetic anchor")
    for relative in SOURCES:
        if sha(ROOT / relative) != anchor["source_hashes"][relative]:
            raise ValueError("Independent model arithmetic changed")
    audio_embedding = read_json(args.hf_audio_embedding_oracle)
    if parsed_hashes[str(args.hf_audio_embedding_oracle.resolve())] != args.hf_audio_embedding_oracle_sha256:
        raise ValueError("HF audio anchor changed before parsing")
    audio_path = Path(audio_embedding["audio_oracle"])
    if sha(audio_path) != audio_embedding["audio_oracle_sha256"] or not audio_embedding["passed"]:
        raise ValueError("Preserved independent HF audio changed")
    audio = read_json(audio_path)
    if parsed_hashes[str(audio_path.resolve())] != audio_embedding["audio_oracle_sha256"]:
        raise ValueError("HF audio learned rows changed before parsing")
    model_hash, projector_hash = sha(args.model), sha(args.projector)
    if (anchor["model_sha256"] != model_hash or anchor["projector_sha256"] != projector_hash or
            audio_embedding["model_sha256"] != model_hash or audio_embedding["projector_sha256"] != projector_hash or
            audio_embedding["encoder_source_sha256"] != anchor["source_hashes"]["eng/embeddinggemma2-reference.py"] or
            audio["projector_sha256"] != projector_hash or audio["audio_log_policy"] != "huggingface_add" or
            audio["left_window_size"] != 12 or audio["past_distance_inclusive"] != 11 or
            audio["source_revision"] != audio_embedding["audio_source_revision"]):
        raise ValueError("Independent HF audio/model policy differs")
    proof = read_json(args.input_proof); validate_proof(proof)
    if proof["model_sha256"] != model_hash:
        raise ValueError("Observed tokenizer model differs")
    reports = [(path, read_json(path)) for path in args.http]
    guards = [(path,read_json(path)) for path in args.runtime_guard]
    if len({str(path.resolve()) for path,_ in reports}) != len(reports) or len({r["expected_backend"] for _,r in reports}) != len(reports):
        raise ValueError("Duplicate HTTP report/backend does not add validation coverage")
    if (proof["source_report_sha256"] != sha(reports[0][0]) or
            Path(proof["source_report_path"]).name != reports[0][0].name):
        raise ValueError("Fresh input proof differs from actual HTTP source")
    first = reports[0][1]
    if proof["manifest_sha256"] != first["manifest"]["sha256"] or proof["manifest_path"] != first["manifest"]["path"]:
        raise ValueError("Fresh input proof manifest differs from actual HTTP source")
    for _, value in reports:
        validate_http(value, proof, model_hash, projector_hash)
        validate_runtime_guards(value,[guard for _,guard in guards])
        if value["media"] != first["media"] or [(r["name"], r["parts"]) for r in value["cases"]] != [(r["name"], r["parts"]) for r in first["cases"]]:
            raise ValueError("Default-HF actual inputs differ across backends")
    decoder = read_json(args.decoder_proof)
    if parsed_hashes[str(args.decoder_proof.resolve())] != args.decoder_proof_sha256:
        raise ValueError("Mac decoder proof differs from its explicit hash")
    decoded_video, decoder_paths = validate_decoder(decoder, args.decoder_proof.parent, proof, args.input_proof, read_json)
    vision_module = module("default_hf_vision", ROOT / SOURCES[1])
    text_module = module("default_hf_text", ROOT / SOURCES[2])
    guard_paths = {args.model,args.projector,args.input_proof,args.anchor_oracle,args.hf_audio_embedding_oracle,
        audio_path, args.decoder_proof, *decoder_paths, Path(__file__),
        Path(__file__).with_name("embeddinggemma2-default-hf-composite-reference.py"),
        Path(__file__).with_name("embeddinggemma2_http_environment.py"),
        *(ROOT / relative for relative in SOURCES),*(path for path,_ in reports),*(path for path,_ in guards)}
    for record in proof["images"]+proof["audios"]+proof["videos"]:
        guard_paths.add(args.media_root/Path(record["path"]).name)
    for record in proof["images"]+[frame for video in proof["videos"] for frame in video["frames"]]:
        guard_paths.update(base.artifact(args.input_proof.parent,record[key]) for key in ("rgba","pixels"))
    before_hashes = {str(path.resolve()):sha(path) for path in guard_paths}
    before_hashes.update({str(path.resolve()):digest for path,digest in decoder_paths.items()})
    before_hashes.update(parsed_hashes)
    before_hashes[str(Path(__file__).with_name("embeddinggemma2-default-hf-composite-reference.py").resolve())] = "d83acc0e55848abbc1498d6a9b3f8312d1c172b68d313a4d09d8637d868e769e"
    before_hashes[str(args.model.resolve())]=model_hash
    before_hashes[str(args.projector.resolve())]=projector_hash
    for relative in SOURCES:
        before_hashes[str((ROOT/relative).resolve())]=anchor["source_hashes"][relative]
    if any(sha(path) != digest for path,digest in before_hashes.items()):
        raise ValueError("Bound input changed between parsing and full model math")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, passed=False, method="Fresh complete independent Mac default-HF vision/text math; independent PNG and separately applied Apple platform codec with exact RGBA; float bicubic and hash-bound HF audio rows",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model_sha256=model_hash, projector_sha256=projector_hash,
        thresholds=dict(min_cosine=.9999, max_absolute_error=.002), pixel_max_absolute_error=2e-6,
        source_hashes={**anchor["source_hashes"],
            "eng/validation/embeddinggemma2-default-hf-composite-reference.py":sha(Path(__file__).with_name("embeddinggemma2-default-hf-composite-reference.py")),
            str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        codec_reference_kind="mac_avfoundation_same_codec_separate_application",
        ffmpeg_equivalence_qualified=False, independent_codec_implementation=False,
        decoder_proof=dict(path=str(args.decoder_proof.resolve()),sha256=args.decoder_proof_sha256),
        runtime_guards=[dict(path=str(path.resolve()),sha256=sha(path)) for path,_ in guards],
        inputs={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in
            (("proof",args.input_proof),("arithmetic_anchor",args.anchor_oracle),("hf_audio_embedding",args.hf_audio_embedding_oracle),("hf_audio",audio_path))},
        python=sys.version,numpy=np.__version__,before_hashes=before_hashes,
        fresh_unique_vision_inputs=0,fresh_complete_text_cases=0,images=[],videos=[],cases=[],
        limitations=["Recorded seven synthetic default-HF cases only, not a retrieval leaderboard or llama pipeline equivalence.",
            "Token IDs observed from the frozen production tokenizer; separate tokenizer regression supplies independent token coverage.",
            "Float bicubic uses independent PIL F-channel resizing, a bounded reference approximation to HF torchvision float resizing.",
            "Only opaque decoded inputs qualified; no alpha compositing assertion. No service timing claim.",
            "Independent HF audio learned rows reused, not counted as fresh audio model math.",
            "Video decode uses a separate application of the same Apple platform codec, not a second codec implementation or restored FFmpeg equality.",
            "Mac codec proof and SDK interfaces/OS build do not individually hash Apple shared-cache implementation bytes."])
    def save(): args.output.write_text(json.dumps(report,indent=2)+"\n")
    try:
        vision, text = vision_module.VisionEncoder(args.projector), text_module.Encoder(args.model)
        vocabulary = text.metadata("tokenizer.ggml.tokens")
        ids = dict(bos=int(text.metadata("tokenizer.ggml.bos_token_id")),eos=int(text.metadata("tokenizer.ggml.eos_token_id")),
            image_start=vocabulary.index("<|image>"),image_end=vocabulary.index("<image|>"),audio_start=vocabulary.index("<|audio>"),audio_end=vocabulary.index("<audio|>"))
        if proof["token_ids"] != ids: raise ValueError("Observed delimiter IDs differ from checkpoint")
        texts = {row["text"]:row["ids"] for row in proof["texts"]}
        if len(texts) != len(proof["texts"]) or any(any(type(token) is not int or not 0 <= token < len(vocabulary) for token in row) for row in texts.values()):
            raise ValueError("Malformed observed text token IDs")
        media_hashes = {row["path"]:row["sha256"] for row in first["media"]}
        def media(record):
            path = args.media_root / Path(record["path"]).name
            if sha(path) != record["sha256"] or media_hashes.get(record["path"]) != record["sha256"]:
                raise ValueError("Observed media identity changed")
            return path
        unique, image_rows, video_rows = {}, {}, {}
        def encode(record, rgba, soft_tokens):
            pixels, observation = validate_pixels(record,args.input_proof.parent,rgba,vision.hf)
            if vision.hf.get_aspect_ratio_preserving_size(record["source_height"],record["source_width"],16,soft_tokens*9,3) != (record["processed_height"],record["processed_width"]):
                raise ValueError("Observed resize canvas differs from independent HF geometry")
            key = vision_key(pixels,observation["independent_pixels_sha256"])
            if key not in unique:
                soft = vision.encode(pixels)
                if soft.shape != (record["soft_tokens"],512) or not np.all(np.isfinite(soft)):
                    raise ValueError("Fresh complete vision rows invalid")
                output=args.output.parent/("vision-"+key+"-soft.f32");soft.astype("<f4").tofile(output)
                unique[key]=soft;report["fresh_unique_vision_inputs"]+=1
                observation["soft_rows_sha256"]=sha(output)
            return unique[key],observation
        for record in proof["images"]:
            with Image.open(media(record)) as image:
                if image.format != "PNG": raise ValueError("Only recorded PNG input qualified")
                rgba=np.asarray(image.convert("RGBA"))
            image_rows[record["path"]],observation=encode(record,rgba,280)
            report["images"].append(dict(path=record["path"],**observation));save()
        report["mac_video_decoder"]=dict(proof_sha256=args.decoder_proof_sha256, exact_sampled_rgba=True, same_platform_codec=True)
        for video in proof["videos"]:
            base.validate_video_indices(video);path=media(video);frames=video["frames"]
            width,height=frames[0]["source_width"],frames[0]["source_height"];layout=frames[0]["decoded_layout"]
            if any((f["source_width"],f["source_height"],f["decoded_layout"]) != (width,height,layout) for f in frames):
                raise ValueError("Changing video source layout unqualified")
            decoded=decoded_video[video["path"]]
            rows=[];observations=[]
            for frame in frames:
                soft,observation=encode(frame,decoded[frame["index"]],140);rows.append(soft)
                observations.append(dict(index=frame["index"],**observation))
            video_rows[video["path"]]=rows;report["videos"].append(dict(path=video["path"],frames=observations));save()
        audio_rows={};soft_audio=np.asarray(audio["soft_rows"],dtype=np.float32)
        if list(soft_audio.shape) != audio["shape"] or soft_audio.shape != (25,512) or not np.all(np.isfinite(soft_audio)):
            raise ValueError("Preserved HF audio rows invalid")
        for record in proof["audios"]:
            media(record)
            if record["sha256"] != audio["waveform_sha256"]: raise ValueError("Canonical HF audio waveform differs")
            audio_rows[record["path"]]=soft_audio
        for row in first["cases"]:
            inputs=base.compose(row["parts"],ids,texts,image_rows,audio_rows,video_rows,"",text.token_rows)
            expected=text.encode_inputs(inputs).astype(np.float64)
            if expected.shape != (768,) or not np.all(np.isfinite(expected)) or abs(float(np.linalg.norm(expected))-1)>1e-5:
                raise ValueError("Fresh full default-HF embedding invalid")
            report["fresh_complete_text_cases"]+=1
            path=args.output.parent/(row["name"]+"-inputs.f32");inputs.astype("<f4").tofile(path)
            comparisons=[]
            for http_path,http in reports:
                actual=next(case for case in http["cases"] if case["name"]==row["name"])
                comparisons.append(dict(http=str(http_path.resolve()),sha256=sha(http_path),backend=http["expected_backend"],protocols=compare(expected,actual,len(inputs))))
            report["cases"].append(dict(name=row["name"],total_input_rows=len(inputs),input_rows_sha256=sha(path),embedding=expected.tolist(),comparisons=comparisons));save()
        report["after_hashes"]={path:sha(path) for path in before_hashes}
        report["input_source_hashes_unchanged"]=report["after_hashes"]==before_hashes
        if not report["input_source_hashes_unchanged"]:
            raise ValueError("Bound input or independent source changed during full model math")
        report["passed"]=all(p["passed"] for case in report["cases"] for comparison in case["comparisons"] for p in comparison["protocols"].values())
        report["completed"]=True;save()
    except Exception as error:
        report["error"]=str(error);save();raise
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
