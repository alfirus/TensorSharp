import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-mac-hf-composite-reference.py"
spec = importlib.util.spec_from_file_location("mac_hf_reference_test", PATH)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


class MacDecoderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.decoded = self.parent / "decoded"; self.decoded.mkdir()
        self.proof_path = self.parent / "input-proof.json"
        video = self.parent / "fixture.mp4"; video.write_bytes(b"synthetic evidence; never decoded")
        frames = []
        raw_frames = []
        rows = []
        for index in range(4):
            pixels = bytes([index, 7, 13, 255])
            observed = self.parent / f"observed-{index}.rgba"; observed.write_bytes(pixels)
            direct = self.decoded / f"frame-{index}.rgba"; direct.write_bytes(pixels)
            frames.append(dict(index=index,rgba=observed.name,rgba_sha256=tool.sha(observed)))
            raw_frames.append(dict(index=index,width=1,height=1,opaque_alpha=True,rgba_bytes=4,
                pts_value=index,pts_timescale=1,pts_seconds=index,rgba_file=direct.name))
            rows.append(dict(index=index,exact_bytes_equal=True,maximum_byte_difference=0,
                production_rgba_sha256=tool.sha(observed),direct_rgba_sha256=tool.sha(direct)))
        self.proof = dict(videos=[dict(path=str(video),frame_count=4,fps=1,source_width=1,source_height=1,
            sampled_indices=[0,1,2,3],frames=frames)])
        for name, path_key, hash_key in (("models","models_assembly","models_assembly_sha256"),
                ("runtime","runtime_assembly","runtime_assembly_sha256"),("codec","video_codec_path","video_codec_sha256")):
            path = self.parent / name; path.write_bytes(name.encode())
            self.proof[path_key] = str(path); self.proof[hash_key] = tool.sha(path)
        self.write(self.proof_path,self.proof)
        self.raw = dict(completed=True,passed=True,decoder="AVAssetReaderTrackOutput",requested_format="32BGRA",
            model_forwards=0,independent_codec_implementation=False,ffmpeg_equivalence_qualified=False,
            video=str(video),frame_count=4,nominal_fps=1,frames=raw_frames)
        self.write(self.decoded/"decode.json",self.raw)
        before={str(path):tool.sha(path) for path in self.parent.iterdir() if path.is_file()}
        self.write(self.parent/"before.json",before);self.write(self.parent/"after.json",before)
        aliases={path:str(Path(path).resolve()) for path in before}
        self.write(self.parent/"aliases-before.json",aliases);self.write(self.parent/"aliases-after.json",aliases)
        self.decoder=dict(completed=True,passed=True,inputs_unchanged=True,aliases_unchanged=True,decode_exit=0,
            learned_forwards=0,independent_codec_implementation=False,ffmpeg_equivalence_qualified=False,
            input_proof_sha256=tool.sha(self.proof_path),models_sha256=self.proof["models_assembly_sha256"],
            runtime_sha256=self.proof["runtime_assembly_sha256"],codec_sha256=self.proof["video_codec_sha256"],
            decode_report_sha256=tool.sha(self.decoded/"decode.json"),decoded_frame_count=4,
            observed_sample_indices=[0,1,2,3],frames=rows)
        self.refresh_raw()

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def refresh_raw(self):
        self.write(self.decoded/"decode.json",self.raw)
        self.decoder["decode_report_sha256"]=tool.sha(self.decoded/"decode.json")
        self.write(self.parent/"generated-files.json",{str(p):tool.sha(p) for p in self.decoded.iterdir()})

    def check(self,decoder=None):
        return tool.validate_decoder(decoder or self.decoder,self.parent,self.proof,self.proof_path,
            lambda path:json.loads(Path(path).read_bytes()))

    def test_complete_same_codec_proof(self):
        decoded,paths=self.check()
        self.assertEqual((4,1,1,4),next(iter(decoded.values())).shape)
        self.assertIn(self.proof_path,paths)

    def test_retains_original_guard_digest_for_outer_capture(self):
        _,bound=self.check()
        codec=Path(self.proof["video_codec_path"])
        original=tool.sha(codec);codec.write_bytes(b"changed after validation")
        self.assertEqual(original,bound[codec])
        self.assertNotEqual(tool.sha(codec),bound[codec])

    def test_refuses_wrong_scope_and_failure(self):
        for change in (dict(completed=False),dict(passed=False),dict(inputs_unchanged=False),dict(aliases_unchanged=False),
                dict(independent_codec_implementation=True),dict(ffmpeg_equivalence_qualified=True),dict(learned_forwards=1),dict(decode_exit=1)):
            d=copy.deepcopy(self.decoder);d.update(change)
            with self.assertRaises(ValueError):self.check(d)

    def test_refuses_stale_proof_assembly_and_codec(self):
        for key in ("input_proof_sha256","models_sha256","runtime_sha256","codec_sha256","decode_report_sha256"):
            d=copy.deepcopy(self.decoder);d[key]="stale"
            with self.assertRaises(ValueError):self.check(d)

    def test_refuses_changed_input_guard_and_alias(self):
        for name in ("after.json","aliases-after.json"):
            path=self.parent/name;original=path.read_bytes();self.write(path,{})
            with self.assertRaises(ValueError):self.check()
            path.write_bytes(original)

    def test_refuses_missing_codec_guard(self):
        for name in ("before.json","after.json","aliases-before.json","aliases-after.json"):
            path=self.parent/name;data=json.loads(path.read_bytes());del data[self.proof["video_codec_path"]];self.write(path,data)
        with self.assertRaises(ValueError):self.check()

    def test_refuses_changed_generated_pixels_even_when_rehashed(self):
        (self.decoded/"frame-0.rgba").write_bytes(bytes([99,7,13,255]));self.refresh_raw()
        with self.assertRaises(ValueError):self.check()

    def test_refuses_missing_or_nonopaque_pixel_bytes(self):
        for pixels in (b"",bytes([0,7,13,254])):
            (self.decoded/"frame-0.rgba").write_bytes(pixels);self.refresh_raw()
            with self.assertRaises(ValueError):self.check()

    def test_refuses_missing_frame_and_order(self):
        for value in ([0,1,2],[1,0,2,3]):
            d=copy.deepcopy(self.decoder);d["observed_sample_indices"]=value
            with self.assertRaises(ValueError):self.check(d)

    def test_refuses_nonfinite_rate_and_pts(self):
        self.raw["nominal_fps"]=float("nan");self.refresh_raw()
        with self.assertRaises(ValueError):self.check()
        self.raw["nominal_fps"]=1;self.raw["frames"][0]["pts_seconds"]=float("nan");self.refresh_raw()
        with self.assertRaises(ValueError):self.check()

    def test_refuses_changed_frame_geometry_and_cadence(self):
        for key,value in (("width",2),("pts_value",1),("pts_timescale",0),("index",1)):
            original=copy.deepcopy(self.raw);self.raw["frames"][0][key]=value;self.refresh_raw()
            with self.assertRaises(ValueError):self.check()
            self.raw=original;self.refresh_raw()

    def test_refuses_raw_decoder_policy_change(self):
        for key,value in (("decoder","FFmpeg"),("requested_format","32RGBA"),("model_forwards",1)):
            original=copy.deepcopy(self.raw);self.raw[key]=value;self.refresh_raw()
            with self.assertRaises(ValueError):self.check()
            self.raw=original;self.refresh_raw()

    def test_refuses_output_path_escape(self):
        self.raw["frames"][0]["rgba_file"]="../models";self.refresh_raw()
        with self.assertRaises(ValueError):self.check()


if __name__ == "__main__":unittest.main()
