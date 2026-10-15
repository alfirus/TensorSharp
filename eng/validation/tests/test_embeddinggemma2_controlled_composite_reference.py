import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("composite_reference", Path(__file__).parents[1] / "embeddinggemma2-controlled-composite-reference.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture():
    engine = dict(completed=True, server_exit_code=0,command=["env","TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp","server"])
    pair = dict(completed=True, binary_artifacts_unchanged=True, dependencies_unchanged_and_clean=True,
        model=dict(sha256="m"), projector=dict(sha256="p"), controlled_video_prefix=True,
        binary_artifacts=dict(tensorsharp=[dict(path="TensorSharp.Models.dll",sha256="models"),dict(path="TensorSharp.Runtime.dll",sha256="runtime")]),
        engines=dict(tensorsharp=copy.deepcopy(engine),llama=copy.deepcopy(engine)))
    proof = dict(completed=True,model_sha256="m",models_assembly_sha256="models",runtime_assembly_sha256="runtime",
        video_prefix="Video:",video_soft_tokens=140,include_video_timestamps=False,video_frames_per_second=1,max_video_frames=32)
    return pair,proof


class CompositeEvidenceTests(unittest.TestCase):
    def test_exact_frozen_pair_and_policy_bind(self):
        tool.bind_pair(*fixture(), "m", "p")

    def test_failed_original_quality_is_not_converted_to_input_failure(self):
        pair,proof=fixture();pair.update(passed=False,quality_pass=False)
        tool.bind_pair(pair,proof,"m","p")

    def test_changed_assembly_runtime_policy_or_cleanup_rejected(self):
        for location,key,value in (("pair","binary_artifacts_unchanged",False),("pair","dependencies_unchanged_and_clean",False),
                ("proof","models_assembly_sha256","other"),("proof","runtime_assembly_sha256","other"),
                ("proof","model_sha256","other"),("proof","include_video_timestamps",True),("proof","video_soft_tokens",280),
                ("proof","video_prefix",""),("proof","completed",False)):
            pair,proof=fixture();(pair if location=="pair" else proof)[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):tool.bind_pair(pair,proof,"m","p")
        pair,proof=fixture();pair["engines"]["llama"]["server_exit_code"]=1
        with self.assertRaises(ValueError):tool.bind_pair(pair,proof,"m","p")
        pair,proof=fixture();pair["engines"]["tensorsharp"]["command"]=["server"]
        with self.assertRaises(ValueError):tool.bind_pair(pair,proof,"m","p")

    def test_proof_buffers_cannot_escape_evidence_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            self.assertEqual(tool.artifact(parent,"frame.f32"),parent/"frame.f32")
            for value in ("../other.f32","/tmp/other.f32"):
                with self.assertRaises(ValueError):tool.artifact(parent,value)

    def test_video_negative_wrap_indices_invalid_fps_and_sampler_changes_rejected(self):
        valid=dict(frame_count=4,fps=1.,sampled_indices=[0,1,2,3],frames=[dict(index=i) for i in range(4)])
        tool.validate_video_indices(valid)
        for update in (dict(sampled_indices=[-4,-3,-2,-1],frames=[dict(index=i) for i in range(-4,0)]),
                dict(sampled_indices=[0,1,2,4]),dict(fps=float("nan")),dict(fps=0),dict(frame_count=True),
                dict(sampled_indices=[False,1,2,3]),dict(sampled_indices=[0,2,1,3])):
            with self.subTest(update=update),self.assertRaises(ValueError):tool.validate_video_indices(dict(valid,**update))

    def test_long_clip_sampling_covers_final_source_time(self):
        video=dict(frame_count=100,fps=1.,sampled_indices=[int(i*99//31) for i in range(32)])
        video["frames"]=[dict(index=i) for i in video["sampled_indices"]]
        tool.validate_video_indices(video)

    def test_independent_decoder_uses_observed_channel_layout_before_opaque_rgba(self):
        for layout,raw in (("Bgr",bytes([3,2,1])),("Rgb",bytes([1,2,3])),("Bgra",bytes([3,2,1,19])),("Rgba",bytes([1,2,3,19]))):
            with self.subTest(layout=layout):
                self.assertEqual(tool.decoded_rgba(raw,1,1,1,layout).reshape(-1).tolist(),[1,2,3,255])
        with self.assertRaises(ValueError):tool.decoder_format("unknown")

    def test_identity_pixel_proof_checks_all_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);rgba=np.full((48,48,4),255,dtype=np.uint8)
            rgba[:,:,0]=17;pixels=rgba[:,:,:3].transpose(2,0,1).astype(np.float32)/np.float32(255)
            rgba.tofile(parent/"rgba");pixels.tofile(parent/"pixels")
            record=dict(source_width=48,source_height=48,processed_width=48,processed_height=48,soft_tokens=1,
                rgba="rgba",pixels="pixels",rgba_sha256=tool.sha(parent/"rgba"),pixels_sha256=tool.sha(parent/"pixels"))
            tool.identity_pixels(record,parent)
            pixels[0,17,17]+=1e-5;pixels.tofile(parent/"pixels");record["pixels_sha256"]=tool.sha(parent/"pixels")
            with self.assertRaises(ValueError):tool.identity_pixels(record,parent)

    def test_video_prefix_is_inserted_once_and_merged_with_adjacent_text(self):
        parts=[dict(kind="text",value="x"),dict(kind="video",value="v")]
        expanded=tool.controlled_parts(parts,"Video:")
        ids=dict(bos=1,eos=2,image_start=3,image_end=4,audio_start=5,audio_end=6)
        rows=lambda tokens:np.asarray(tokens,dtype=np.float32).reshape(-1,1)
        expected=[1,7,8,3,10,4,3,11,4,2]
        result=tool.compose(expanded,ids,{"xVideo:":[7,8]}, {},{}, {"v":[rows([10]),rows([11])]},"",rows)
        self.assertEqual(result[:,0].tolist(),expected)
        result=tool.compose(parts,ids,{"xVideo:":[7,8]}, {},{}, {"v":[rows([10]),rows([11])]},"Video:",rows)
        self.assertEqual(result[:,0].tolist(),expected)

    def test_mixed_order_and_media_boundaries_are_preserved(self):
        parts=[dict(kind="text",value="x"),dict(kind="image",value="i"),dict(kind="audio",value="a")]
        ids=dict(bos=1,eos=2,image_start=3,image_end=4,audio_start=5,audio_end=6)
        rows=lambda tokens:np.asarray(tokens,dtype=np.float32).reshape(-1,1)
        result=tool.compose(parts,ids,{"x":[7]}, {"i":rows([10,11])},{"a":rows([12])},{},"",rows)
        self.assertEqual(result[:,0].tolist(),[1,7,3,10,11,4,5,12,6,2])

    def test_numerical_failure_is_retained_and_nonfinite_count_changes_rejected(self):
        row=dict(embedding=[0.,1.],parts=[],prompt_tokens=4)
        self.assertFalse(tool.compare([1.,0.],row,[],4)["passed"])
        for values,count in (([float("nan"),1.],4),([0.,1.],5)):
            with self.assertRaises(ValueError):tool.compare([1.,0.],dict(row,embedding=values,prompt_tokens=count),[],4)


if __name__ == "__main__":
    unittest.main()
