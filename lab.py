#!/usr/bin/env python3
"""Evidence-oriented runner for pinned NVIDIA nccl-tests (Python stdlib only)."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OPS = {"all_reduce": "all_reduce_perf", "all_gather": "all_gather_perf"}
ENV_KEYS = {"NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_SOCKET_IFNAME"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(cmd, timeout=30, env=None):
    """Kill and reap the whole process group; keep partial logs on timeout."""
    begin = time.monotonic()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True, env=env)
    except OSError as e:
        return {"command": cmd, "returncode": None, "timeout": False, "launch_error": str(e),
                "stdout": "", "stderr": "", "elapsed_s": time.monotonic() - begin}
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGTERM)
        try:
            out, err = p.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
    return {"command": cmd, "returncode": p.returncode, "timeout": timed_out,
            "launch_error": None, "stdout": out, "stderr": err, "elapsed_s": time.monotonic() - begin}


def manifest():
    cmds = {"gpus": ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total", "--format=csv,noheader"],
            "topology": ["nvidia-smi", "topo", "-m"],
            "p2p": ["nvidia-smi", "topo", "-p2p", "r"],
            "cuda": ["nvcc", "--version"], "cpu": ["lscpu", "-J"]}
    collected = {k: execute(cmd) for k, cmd in cmds.items()}
    return {"schema_version": 1, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kernel": platform.release(), "architecture": platform.machine(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "tools": collected,
            "nccl_system_config_sha256": sha256('/etc/nccl.conf') if Path('/etc/nccl.conf').exists() else None,
            "provenance": json.loads((ROOT / "upstream.lock.json").read_text()),
            "missing": {"physical_network_counters": "single-node scope", "nvlink": "determined by topology; never assumed"}}


def percentile(xs, q):
    if not xs:
        return None
    return sorted(xs)[max(0, math.ceil(q * len(xs)) - 1)]


def normalize(payload, op, ranks):
    """Native JSON time is microseconds; busbw is collective-normalized GB/s."""
    if ranks < 1 or op not in OPS:
        raise ValueError("invalid rank count or collective")
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or not payload["results"]:
        raise ValueError("missing non-empty native results")
    if not payload.get("end_time"):
        raise ValueError("incomplete native JSON: no end_time")
    config = payload.get("config", {})
    if config.get("ngpus", 0) * config.get("nthreads", 0) != ranks:
        raise ValueError("native rank count differs from requested ranks")
    if sorted(d.get("rank", -1) for d in config.get("devices", [])) != list(range(ranks)):
        raise ValueError("missing or duplicate rank mapping")
    if payload.get("config", {}).get("validation", 0) < 1:
        raise ValueError("correctness validation was disabled")
    rows = []
    factor = (2 if op == "all_reduce" else 1) * (ranks - 1) / ranks
    for r in payload["results"]:
        size = r.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("invalid size")
        for placement in ("out_of_place", "in_place"):
            m = r.get(placement)
            if not isinstance(m, dict):
                raise ValueError(f"missing {placement}")
            for key in ("time", "alg_bw", "bus_bw", "nwrong"):
                value = m.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"missing/invalid {key}")
            if m["time"] <= 0 or int(m["nwrong"]) != m["nwrong"]:
                raise ValueError("invalid timing/error count")
            # Upstream prints JSON doubles with finite precision. Small-size bandwidth rounds to zero.
            expected_alg = size / m["time"] / 1000
            if abs(m["alg_bw"] - expected_alg) > max(0.015, expected_alg * 0.025):
                raise ValueError("algorithm bandwidth/size/time mismatch")
            if abs(m["bus_bw"] - m["alg_bw"] * factor) > max(0.02, m["alg_bw"] * factor * 0.025):
                raise ValueError("bus bandwidth normalization mismatch")
            timing = r.get(placement + "_per_iter", {})
            samples = timing.get("times_us")
            if samples is not None:
                if not isinstance(samples, list) or not samples or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError("invalid iteration samples")
                if len(samples) != r.get("actual_iterations") or len(samples) != config.get("iterations"):
                    raise ValueError("incomplete iteration sample count")
            rows.append({"op": op, "ranks": ranks, "size_bytes": size, "placement": placement,
                         "time_us": m["time"], "alg_bw_GBs": m["alg_bw"], "bus_bw_GBs": m["bus_bw"],
                         "bus_factor": factor, "nwrong": int(m["nwrong"]), "iteration_times_us": samples,
                         "aggregate_out_of_bounds": payload.get("out_of_bounds", {}).get("count"),
                         "per_process_max_times_us": timing.get("per_process_max_times_us"),
                         "rank_timing_missing_reason": "non-MPI run; GPU-rank distribution is not emitted"})
    return rows


def validate_contract(payload, op, ranks, expected):
    """Admit only the pinned JSON-v4, single-process float experiment requested.

    This is an execution contract, not authentication of an untrusted producer.
    Normalize remains available for exploratory data; sweeps use this gate first.
    """
    def same(actual, wanted, label):
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError("native/request mismatch: " + label)

    if not isinstance(payload, dict):
        raise ValueError("native payload must be an object")
    same(payload.get("version"), 4, "JSON version")
    same(payload.get("args"), expected["command"], "argv")
    if Path(expected["command"][0]).name != OPS[op]:
        raise ValueError("collective executable mismatch")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("missing native config")
    for key, value in {"nthreads": 1, "ngpus": ranks, "minimum_bytes": expected["size"],
                       "maximum_bytes": expected["size"], "iterations": expected["iterations"],
                       "warmup_iters": expected["warmup"], "aggregated_iterations": 1,
                       "validation": 1, "graph": 0, "per_iter_timing": "true", "per_iter_skip": 0}.items():
        same(config.get(key), value, key)
    devices = config.get("devices")
    if not isinstance(devices, list) or len(devices) != ranks:
        raise ValueError("incomplete device mapping")
    for rank, device in enumerate(devices):
        if not isinstance(device, dict):
            raise ValueError("invalid device mapping")
        same(device.get("rank"), rank, "rank")
        # CUDA_VISIBLE_DEVICES remaps physical indices to contiguous logical indices.
        same(device.get("device"), rank, "logical device")
    env = payload.get("env")
    if not isinstance(env, list) or any(not isinstance(e, str) or "=" not in e for e in env):
        raise ValueError("invalid environment record")
    pairs = [e.split("=", 1) for e in env]
    if len(dict(pairs)) != len(pairs):
        raise ValueError("duplicate environment record")
    if expected.get("environment") is not None:
        same(dict(pairs), expected["environment"], "environment")
    else:
        same(dict(pairs).get("CUDA_VISIBLE_DEVICES"), ",".join(map(str, expected["devices"])), "device order")
    errors = payload.get("errors")
    if not isinstance(errors, list) or any(not isinstance(e, str) or e.strip() for e in errors):
        raise ValueError("native error diagnostics")
    bounds = payload.get("out_of_bounds")
    if not isinstance(bounds, dict):
        raise ValueError("missing aggregate correctness")
    if type(bounds.get("count")) is not int or bounds["count"] < 0:
        raise ValueError("invalid aggregate error count")
    same(bounds.get("okay"), "true" if bounds["count"] == 0 else "false", "aggregate correctness")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("expected exactly one result per invocation")
    row = results[0]
    for key, value in {"size": expected["size"], "type": "float",
                       "redop": "sum" if op == "all_reduce" else "none",
                       "count": expected["size"] // (4 * (ranks if op == "all_gather" else 1)),
                       "actual_iterations": expected["iterations"]}.items():
        same(row.get(key), value, key)
    for placement in ("in_place", "out_of_place"):
        timing = row.get(placement + "_per_iter")
        if not isinstance(timing, dict):
            raise ValueError("missing per-iteration timing")
        same(timing.get("skipped_iterations"), 0, "skipped iterations")
        samples = timing.get("times_us")
        if not isinstance(samples, list) or len(samples) != expected["iterations"]:
            raise ValueError("missing iteration samples")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("invalid iteration timing")
    return normalize(payload, op, ranks)


def classify(proc, rows=None, parse_error=None):
    if proc.get("timeout"):
        return "timeout"
    if proc.get("launch_error"):
        return "launch_failure"
    if rows and any(x["nwrong"] != 0 or (x.get("aggregate_out_of_bounds") or 0) > 0 for x in rows):
        return "numerical_error"
    if proc.get("returncode") != 0:
        return "runtime_failure"
    if parse_error or not rows:
        return "invalid_evidence"
    return "pass"


def summary(records):
    groups = {}
    for r in records:
        if r["status"] != "pass":
            continue
        for row in r["rows"]:
            key = (r["case"], row["op"], row["size_bytes"], row["placement"])
            groups.setdefault(key, []).append(row["time_us"])
    return [{"case": k[0], "op": k[1], "size_bytes": k[2], "placement": k[3],
             "repeats": len(v), "median_run_mean_us": statistics.median(v),
             "p95_run_mean_us": percentile(v, .95), "min_run_mean_us": min(v), "max_run_mean_us": max(v)}
            for k, v in sorted(groups.items())]


def validate_config(c):
    if set(c) - {"devices", "sizes", "iterations", "warmup", "repeats", "timeout_s", "cases"}:
        raise ValueError("unknown configuration key")
    devices = c["devices"]
    if not isinstance(devices, list) or not devices or len(set(devices)) != len(devices) or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError("devices must be distinct nonnegative integers")
    for key in ("iterations", "warmup", "repeats", "timeout_s"):
        if type(c[key]) is not int or not 1 <= c[key] <= 3600:
            raise ValueError(f"invalid {key}")
    if not c["sizes"] or any(type(x) is not int or x < 16 * len(devices) or x % (16 * len(devices)) or x > 1024**3 for x in c["sizes"]):
        raise ValueError("sizes must be multiples of 16 * ranks, at most 1 GiB (all-gather rounds down per-rank payloads)")
    if len(set(c["sizes"])) != len(c["sizes"]):
        raise ValueError("duplicate sizes would overwrite experiment evidence")
    cases = c["cases"]
    if not cases or cases[0] != {"name": "default", "env": {}}:
        raise ValueError("first case must be an unchanged default baseline")
    seen = set()
    for case in cases:
        if set(case) - {"name", "env", "cpu_affinity", "reverse_devices", "registration"}:
            raise ValueError("unknown case field")
        if not re.fullmatch(r"[a-z0-9_-]+", case["name"]) or case["name"] in seen:
            raise ValueError("invalid/duplicate case name")
        seen.add(case["name"])
        env = case.get("env", {})
        if set(env) - ENV_KEYS or any(not isinstance(v, str) for v in env.values()):
            raise ValueError("unsupported environment override")
        changed = len(env) + sum(k in case for k in ("cpu_affinity", "reverse_devices", "registration"))
        if changed > 1:
            raise ValueError("only one experimental factor may change")
        if "registration" in case and case["registration"] not in (1, 2):
            raise ValueError("registration must be 1 or 2")
        if "cpu_affinity" in case and (not case["cpu_affinity"] or not set(case["cpu_affinity"]) <= os.sched_getaffinity(0)):
            raise ValueError("CPU affinity outside available CPUs")


def run_sweep(args):
    config = json.loads(args.config.read_text())
    validate_config(config)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hardware = manifest()
    save(out / "manifest.json", hardware)
    save(out / "config.json", config)
    devices = config["devices"]
    probe = execute([str(args.preflight.resolve())], timeout=60)
    save(out / "preflight-process.json", probe)
    try:
        baseline = json.loads(probe["stdout"])
        eligible = probe["returncode"] == 0 and baseline["correctness"] == "pass"
        eligible &= all(d < baseline["device_count"] for d in devices)
        eligible &= len(devices) >= 2 or args.single_gpu_smoke
        if len(devices) >= 2:
            eligible &= any(p["src"] == devices[0] and p["dst"] == devices[1] and p["correctness"] == "pass" for p in baseline["peer_links"])
    except (ValueError, KeyError, TypeError):
        eligible = False
    if not eligible:
        save(out / "summary.json", {"status": "blocked_preflight", "qualification": False,
                                    "reason": "required hardware or verified link baseline unavailable"})
        return 2
    records = []
    scope = "single_gpu_smoke" if len(devices) == 1 else "two_gpu_single_node"
    for rep in range(config["repeats"]):
        # Alternating order reduces a fixed order/warm-device bias.
        cases = config["cases"] if rep % 2 == 0 else list(reversed(config["cases"]))
        for case in cases:
            for op, binary in OPS.items():
                for size in config["sizes"]:
                    tag = f"r{rep}-{case['name']}-{op}-{size}"
                    native = out / (tag + ".native.json")
                    selected = list(reversed(devices)) if case.get("reverse_devices") else devices
                    env = {"PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                           "CUDA_VISIBLE_DEVICES": ",".join(map(str, selected)), "NCCL_DEBUG": "INFO",
                           "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET", **case.get("env", {})}
                    if args.library_dir:
                        env["LD_LIBRARY_PATH"] = str(args.library_dir.resolve())
                    exe = (args.bin_dir / binary).resolve()
                    cmd = [str(exe), "-b", str(size), "-e", str(size), "-g", str(len(devices)),
                           "-n", str(config["iterations"]), "-w", str(config["warmup"]), "-c", "1",
                           "-I", "1", "-T", str(config["timeout_s"]), "-J", str(native),
                           "-R", str(case.get("registration", 0))]
                    expected = {"command": list(cmd), "environment": env, "devices": selected,
                                "size": size, "iterations": config["iterations"], "warmup": config["warmup"]}
                    if "cpu_affinity" in case:
                        cmd = ["taskset", "-c", ",".join(map(str, case["cpu_affinity"]))] + cmd
                    proc = execute(cmd, timeout=config["timeout_s"] + 5, env=env)
                    save(out / (tag + ".process.json"), proc)
                    rows, error = [], None
                    try:
                        rows = validate_contract(json.loads(native.read_text()), op, len(devices), expected)
                    except (OSError, ValueError, KeyError, TypeError) as e:
                        error = str(e)
                    status = classify(proc, rows, error)
                    transport_lines = [s for s in (proc["stdout"] + proc["stderr"]).splitlines()
                                       if "Channel" in s and ("via" in s or "->" in s)]
                    records.append({"case": case["name"], "repeat": rep, "op": op, "size": size,
                                    "status": status, "parse_error": error, "rows": rows,
                                    "physical_device_order": selected, "binary_sha256": sha256(exe) if exe.exists() else None,
                                    "actual_transport_evidence": transport_lines or None,
                                    "transport_missing_reason": None if transport_lines else "no channel transport line emitted",
                                    "raw": native.name, "sha256": sha256(native) if native.exists() else None})
                    save(out / "records.json", records)
                    if status != "pass":
                        save(out / "summary.json", {"status": status, "scope": scope, "qualification": False})
                        return 1
    save(out / "summary.json", {"status": "pass", "scope": scope,
                                "qualification": False, "collective_sweep_verified": len(devices) >= 2,
                                "remaining": "two-GPU hypothesis and fault experiments require review before qualification",
                                "statistics": summary(records),
                                "metric_note": "bus_bw is an operation normalization, not per-link telemetry; times are us"})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, default=ROOT / "configs/two-gpu.json")
    r.add_argument("--bin-dir", type=Path, required=True)
    r.add_argument("--library-dir", type=Path)
    r.add_argument("--preflight", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--single-gpu-smoke", action="store_true")
    a = p.parse_args()
    if a.command == "manifest":
        save(a.out, manifest()); return 0
    return run_sweep(a)


if __name__ == "__main__":
    raise SystemExit(main())
