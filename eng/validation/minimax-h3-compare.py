#!/usr/bin/env python3
"""Compare saved H3 CLI outputs after verifying matched generation inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report",required=True,type=Path)
    parser.add_argument("--candidate-report",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("h3generation",Path(__file__).with_name("minimax-h3-generation.py"))
    generation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    reference=json.loads(args.reference_report.read_text())
    candidate=json.loads(args.candidate_report.read_text())
    matched=["prompt","seed","geometry","fps","steps","cfg","flow_shift","sigmas_f32"]
    mismatch={field:[reference.get(field),candidate.get(field)] for field in matched if reference.get(field) != candidate.get(field)}
    for field, default in [("checkpoint", "fl2va"), ("conditioning", [])]:
        def identity(report):
            value=report.get(field, default)
            return [{"role":item["role"],"sha256":item["sha256"]} for item in value] if field == "conditioning" else value
        if identity(reference) != identity(candidate):
            mismatch[field]=[identity(reference),identity(candidate)]
    if mismatch:
        raise SystemExit("Generation inputs differ: "+json.dumps(mismatch))
    width,height,frames=reference["geometry"]
    geometry=SimpleNamespace(width=width,height=height,frames=frames,fps=reference["fps"])
    records=[]
    for report,engine in [(reference,"sdcpp"),(candidate,"tensorsharp")]:
        selected=[record for record in report["executions"] if record["engine"] == engine]
        if len(selected) != 1 or selected[0]["exit_code"] != 0:
            raise SystemExit(f"Require exactly one successful {engine} execution")
        record=selected[0]
        media,rgb,audio=generation.validate_media(Path(record["argv"][-1]),geometry,engine)
        records.append((record,media,rgb,audio))
    metrics=generation.comparison((records[0][2],records[0][3]),(records[1][2],records[1][3]))
    matched_inputs={field:reference[field] for field in matched}
    matched_inputs.update(checkpoint=reference.get("checkpoint","fl2va"),
                          conditioning=[{"role":item["role"],"sha256":item["sha256"]}
                                        for item in reference.get("conditioning",[])])
    result={"matched_inputs": matched_inputs, "backend": candidate["backend"],
            "reference_report": str(args.reference_report.resolve()),"candidate_report": str(args.candidate_report.resolve()),
            "generation_exits": [record[0]["exit_code"] for record in records], "media_verified": True,
            "wall_seconds": {record[0]["engine"]:record[0]["wall_seconds"] for record in records},
            "media": {record[0]["engine"]:record[1] for record in records}, "metrics":metrics,
            "limitations": ["Decoded media includes different output codecs",
                            "Pixel/audio agreement is diagnostic; no quality threshold is inferred from successful output encoding",
                            "Separate new-process smoke runs do not establish formal or warm-model performance parity"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"backend":candidate["backend"],"media_verified":True,"metrics":metrics}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
