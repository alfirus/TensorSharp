#!/usr/bin/env python3
"""Independently decode media exported by the real LTX TensorAgent probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--media-root", type=Path,
                        help="Directory containing unchanged copied media, addressed by original basename; for explicit cross-host validation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    if report.get("status") != "passed":
        raise ValueError("The real TensorAgent workflow did not pass.")
    ffmpeg, ffprobe = (shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("Independent media validation requires ffmpeg and ffprobe.")
    results = []
    jobs = [("direct", item) for item in report["executions"]]
    if report.get("chat_execution") is not None:
        jobs.append(("chat", report["chat_execution"]))
    for kind, job in jobs:
        original_video, original_audio = Path(job["video"]), Path(job["audio"])
        video, audio = ((args.media_root / path.name if args.media_root else path)
                        for path in (original_video, original_audio))
        if sha(video) != job["video_sha256"] or sha(audio) != job["audio_sha256"]:
            raise ValueError("Downloaded media hash changed.")
        probe = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
                                "-show_format", "-of", "json", str(video)],
                               check=True, capture_output=True, text=True)
        metadata = json.loads(probe.stdout)
        videos = [stream for stream in metadata["streams"] if stream["codec_type"] == "video"]
        audios = [stream for stream in metadata["streams"] if stream["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise ValueError("Expected one video and one muxed audio stream.")
        stream, sound = videos[0], audios[0]
        expected = report["request"]
        if (stream["codec_name"] != "h264" or stream["width"] != expected["width"] or
                stream["height"] != expected["height"] or
                int(stream["nb_read_frames"]) != expected["frames"] or
                stream["avg_frame_rate"] != str(expected["fps"]) + "/1" or
                sound["codec_name"] != "aac" or int(sound["sample_rate"]) != 48000 or
                sound["channels"] != 2):
            raise ValueError("Muxed video/audio geometry or codec differs from the contract.")
        with wave.open(str(audio), "rb") as wav:
            params = {"channels": wav.getnchannels(), "sample_rate": wav.getframerate(),
                      "sample_width": wav.getsampwidth(), "samples": wav.getnframes()}
            if params != {"channels": 2, "sample_rate": 48000, "sample_width": 2,
                          "samples": job["audio_samples"]}:
                raise ValueError("Native PCM sidecar differs from the reported layout or length.")
        # Decode both streams fully; AAC padding is distinct from the native PCM length.
        subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(video),
                        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                       check=True, capture_output=True)
        results.append({"kind": kind, "repeat": job.get("repeat"),
                        "video": str(video), "audio": str(audio),
                        "original_video": str(original_video), "original_audio": str(original_audio),
                        "video_sha256": sha(video), "audio_sha256": sha(audio),
                        "ffprobe": metadata, "wav": params, "full_decode_passed": True})
    direct = [job for kind, job in jobs if kind == "direct"]
    output = {"status": "passed", "report": str(args.report), "report_sha256": sha(args.report),
              "media_root_override": str(args.media_root) if args.media_root else None,
              "scope": "Independent full H264/AAC decode and exact PCM sidecar layout/length; structural validity is not a perceptual quality ranking.",
              "tools": {name: {"path": path, "sha256": sha(Path(path))}
                        for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))},
              "direct_repeat_video_byte_identical": len({job["video_sha256"] for job in direct}) == 1,
              "direct_repeat_audio_byte_identical": len({job["audio_sha256"] for job in direct}) == 1,
              "media": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "fully_decoded_media": len(results)}))


if __name__ == "__main__":
    main()
