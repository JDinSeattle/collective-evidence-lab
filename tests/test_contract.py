"""Mutation tests use saved real nccl-tests output, not hand-built happy paths."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab
from audit import audit

SNAPSHOT = lab.ROOT / "evidence/snapshot/final-single-gpu"


class ContractTests(unittest.TestCase):
    def setUp(self):
        path = SNAPSHOT / "r0-default-all_reduce-1024.native.json"
        self.payload = json.loads(path.read_text())
        self.expected = {"command": self.payload["args"], "size": 1024, "iterations": 30,
                         "warmup": 5, "devices": [0]}

    def reject(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        with self.assertRaises(ValueError):
            lab.validate_contract(payload, "all_reduce", 1, self.expected)

    def test_replay_real_completed_sweep(self):
        self.assertEqual(audit(SNAPSHOT)["process_runs"], 30)

    def test_requested_size_cannot_accept_other_valid_measurement(self):
        with self.assertRaises(ValueError):
            lab.validate_contract(self.payload, "all_reduce", 1, {**self.expected, "size": 65536})

    def test_schema_and_execution_configuration(self):
        for key, value in [("iterations", 29), ("warmup_iters", 6), ("nthreads", True),
                           ("per_iter_timing", "false"), ("graph", 1)]:
            with self.subTest(key=key):
                self.reject(lambda p: p["config"].__setitem__(key, value))
        self.reject(lambda p: p.__setitem__("version", 5))

    def test_collective_type_count_and_cardinality(self):
        for key, value in [("redop", "max"), ("type", "half"), ("count", 128), ("actual_iterations", 29)]:
            with self.subTest(key=key):
                self.reject(lambda p: p["results"][0].__setitem__(key, value))
        self.reject(lambda p: p["results"].append(p["results"][0]))

    def test_argv_and_physical_logical_mapping(self):
        self.reject(lambda p: p["args"].__setitem__(0, "/tmp/all_gather_perf"))
        self.reject(lambda p: p["env"].append("CUDA_VISIBLE_DEVICES=1"))
        self.reject(lambda p: p["config"]["devices"][0].__setitem__("device", True))

    def test_unknown_correctness_and_error_diagnostics(self):
        self.reject(lambda p: p.pop("out_of_bounds"))
        self.reject(lambda p: p["out_of_bounds"].__setitem__("count", False))
        self.reject(lambda p: p["errors"].append("asynchronous error"))
        self.payload["out_of_bounds"] = {"count": 1, "okay": "false"}
        rows = lab.validate_contract(self.payload, "all_reduce", 1, self.expected)
        self.assertEqual(lab.classify({"returncode": 1}, rows), "numerical_error")

    def test_missing_or_boolean_iteration_samples(self):
        self.reject(lambda p: p["results"][0].pop("in_place_per_iter"))
        self.reject(lambda p: p["results"][0]["out_of_place_per_iter"]["times_us"].__setitem__(0, True))

    def test_replay_detects_missing_runs_and_modified_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sweep"
            shutil.copytree(SNAPSHOT, target)
            p = target / "records.json"
            records = json.loads(p.read_text())
            p.write_text(json.dumps(records[:-1]))
            with self.assertRaisesRegex(ValueError, "incomplete sweep"):
                audit(target)
            p.write_text(json.dumps(records))
            p = target / "summary.json"
            summary = json.loads(p.read_text()); summary["statistics"][0]["median_run_mean_us"] *= 2
            p.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary"):
                audit(target)


if __name__ == "__main__":
    unittest.main()
