#!/usr/bin/env python3
"""Validate optimized static-video latents against official single-frame latents.

The genuine full-temporal official fixture is a separate prerequisite. This
comparison does not claim that the official full-resolution temporal graph ran.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--temporal-proof', type=Path, required=True)
    parser.add_argument('--max-absolute', type=float, default=0.0001)
    parser.add_argument('--relative-l2', type=float, default=0.00001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads((args.probe/'probe.json').read_text())
    original = json.loads((args.reference/'reference.json').read_text())
    proof = json.loads(args.temporal_proof.read_text())
    genuine = proof['static_equivalence']['official_trained64_full121_comparison']
    if not probe['completed'] or not probe['static_video_optimized'] or not original['completed'] or not genuine['completed']:
        raise ValueError('Incomplete static probe, original reference, or genuine temporal prerequisite')
    if genuine['temporal_invariance_maxabs'] != 0 or genuine['relative_l2'] > args.relative_l2:
        raise ValueError('Full-temporal official invariance prerequisite failed')
    if probe['checkpoint_sha256'] != original['checkpoint_sha256'] or probe['input_sha256'] != original['input_sha256']:
        raise ValueError('Checkpoint or source RGB differs')
    frames, height, width, channels = probe['input_shape']
    if original['input_shape'] != [1,height,width,channels] or frames < 121 or (frames-1)%8:
        raise ValueError('Static input geometry differs or is outside reference grid')
    entry = original['stages']['encoded']
    reference_file = args.reference/entry['file']
    actual_file = args.probe/'output.f32'
    if digest(reference_file) != entry['sha256'] or digest(actual_file) != probe['output_sha256']:
        raise ValueError('Latent file checksum changed')
    shape = list(entry['shape']); shape[1] = 1+(frames-1)//8
    if probe['output_shape'] != shape:
        raise ValueError('Normalized latent shape differs')
    one = np.fromfile(reference_file,dtype='<f4').reshape(entry['shape'])
    expected = np.repeat(one,shape[1],axis=1).astype(np.float64)
    actual = np.fromfile(actual_file,dtype='<f4').reshape(shape).astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite latent')
    difference = actual-expected
    maximum = float(np.max(np.abs(difference)))
    relative = float(np.linalg.norm(difference.ravel())/max(np.linalg.norm(expected.ravel()),1e-30))
    invariance = float(np.max(np.abs(actual-actual[:,0:1,:,:])))
    report = dict(completed=True,passed=maximum<=args.max_absolute and relative<=args.relative_l2 and invariance==0,
                  scope='Static specialization versus independently validated official single-frame latent; genuine full121 trained64 official temporal oracle prerequisite. Full121 actual512 official execution is not claimed.',
                  backend=probe['backend'],shape=shape,max_absolute_error=maximum,relative_l2_error=relative,
                  temporal_invariance_max_absolute=invariance,thresholds=dict(max_absolute=args.max_absolute,relative_l2=args.relative_l2),
                  times_ms=probe['times_ms'],probe=str(args.probe/'probe.json'),reference=str(args.reference/'reference.json'),
                  temporal_proof=str(args.temporal_proof),temporal_proof_sha256=digest(args.temporal_proof),
                  output_sha256=probe['output_sha256'],input_sha256=probe['input_sha256'],checkpoint_sha256=probe['checkpoint_sha256'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
