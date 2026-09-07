import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lab


def fixture(op="all_reduce",ranks=2):
    factor=(2 if op=="all_reduce" else 1)*(ranks-1)/ranks
    metric={"time":10.,"alg_bw":.1024,"bus_bw":.1024*factor,"nwrong":0}
    return {"end_time":"synthetic unit fixture", "config":{"validation":1,"ngpus":ranks,"nthreads":1,
            "devices":[{"rank":r} for r in range(ranks)]},
            "results":[{"size":1024,"out_of_place":dict(metric),"in_place":dict(metric)}]}


class EvidenceTests(unittest.TestCase):
    def test_bandwidth_semantics(self):
        for op in lab.OPS:
            for n in (1,2,8):
                rows=lab.normalize(fixture(op,n),op,n)
                self.assertEqual(len(rows),2)
                self.assertAlmostEqual(rows[0]["bus_factor"],(2 if op=="all_reduce" else 1)*(n-1)/n)
                self.assertIsNone(rows[0]["per_process_max_times_us"])

    def test_reject_unknown_correctness(self):
        for value in (None,-1,float("nan"),.5):
            p=fixture();p["results"][0]["in_place"]["nwrong"]=value
            with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)

    def test_reject_disabled_validation_and_empty_data(self):
        p=fixture();p["config"]["validation"]=0
        with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)
        p=fixture();p["results"][0]["size"]=0
        with self.assertRaises(ValueError):lab.normalize(p,"all_gather",2)

    def test_reject_unit_or_factor_mismatch(self):
        for key in ("time","bus_bw","alg_bw"):
            p=fixture();p["results"][0]["in_place"][key]*=1000
            with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)

    def test_truncated_run_and_rank_mapping(self):
        p=fixture();del p["end_time"]
        with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)
        p=fixture();p["config"]["devices"]=[{"rank":0},{"rank":0}]
        with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)
        p=fixture();p["config"]["ngpus"]=1
        with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)

    def test_incomplete_iteration_array(self):
        p=fixture();p["config"]["iterations"]=3;p["results"][0]["actual_iterations"]=3
        p["results"][0]["in_place_per_iter"]={"times_us":[9.,10.]}
        with self.assertRaises(ValueError):lab.normalize(p,"all_reduce",2)

    def test_error_classification(self):
        base={"returncode":0,"timeout":False,"launch_error":None}
        rows=lab.normalize(fixture(),"all_reduce",2)
        self.assertEqual(lab.classify(base,rows),"pass")
        rows[1]["nwrong"]=3
        self.assertEqual(lab.classify({**base,"returncode":1},rows),"numerical_error")
        self.assertEqual(lab.classify({**base,"timeout":True},rows),"timeout")
        self.assertEqual(lab.classify(base,[],"truncated JSON"),"invalid_evidence")
        self.assertEqual(lab.classify({**base,"returncode":127}),"runtime_failure")
        self.assertEqual(lab.classify(lab.execute(["/does/not/exist"])),"launch_failure")

    def test_real_exit_and_timeout(self):
        proc=lab.execute([sys.executable,"-c","import sys; print('partial',flush=True);sys.exit(7)"])
        self.assertEqual(proc["returncode"],7)
        proc=lab.execute([sys.executable,"-c","import time;print('partial',flush=True);time.sleep(30)"],timeout=.1)
        self.assertTrue(proc["timeout"]);self.assertIn("partial",proc["stdout"])
        self.assertLess(proc["elapsed_s"],3)

    def test_terminate_grandchild_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/"must-not-exist"
            child="import time,pathlib;time.sleep(1);pathlib.Path("+repr(str(marker))+").touch()"
            parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(30)"
            self.assertTrue(lab.execute([sys.executable,"-c",parent],timeout=.1)["timeout"])
            time.sleep(1.1)
            self.assertFalse(marker.exists())

    def test_one_factor_and_noop_guard(self):
        c=json.loads((lab.ROOT/"configs/two-gpu.json").read_text());lab.validate_config(c)
        c["cases"].append({"name":"bad","env":{"NCCL_ALGO":"Ring","NCCL_PROTO":"Simple"}})
        with self.assertRaises(ValueError):lab.validate_config(c)
        c["cases"].pop();c["sizes"][0]=8
        with self.assertRaises(ValueError):lab.validate_config(c)
        c=json.loads((lab.ROOT/"configs/two-gpu.json").read_text());c["sizes"].append(c["sizes"][0])
        with self.assertRaises(ValueError):lab.validate_config(c)

    def test_repeat_statistics_are_not_rank_percentiles(self):
        rows=lab.normalize(fixture(),"all_reduce",2)
        records=[{"case":"default","status":"pass","rows":rows} for _ in range(3)]
        r=lab.summary(records)
        self.assertEqual(r[0]["repeats"],3);self.assertEqual(r[0]["median_run_mean_us"],10)
        self.assertEqual(lab.percentile([1,2,3],.95),3)


if __name__=="__main__":unittest.main()
