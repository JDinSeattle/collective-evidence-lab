#!/usr/bin/env python3
"""Replay a completed sweep against its requested configuration, without a GPU."""
import argparse
import json
from pathlib import Path
from lab import OPS, classify, summary, validate_contract


def audit(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    records = json.loads((directory / "records.json").read_text())
    wanted = {(rep, case["name"], op, size) for rep in range(config["repeats"])
              for case in config["cases"] for op in OPS for size in config["sizes"]}
    seen = set()
    replay = []
    for record in records:
        identity = (record["repeat"], record["case"], record["op"], record["size"])
        if identity not in wanted or identity in seen:
            raise ValueError("unexpected or duplicate experiment")
        seen.add(identity)
        native = directory / record["raw"]
        proc = json.loads(native.with_name(native.name.replace(".native.json", ".process.json")).read_text())
        command = proc["command"]
        if command[0] == "taskset":
            command = command[3:]
        case = next(c for c in config["cases"] if c["name"] == record["case"])
        devices = list(reversed(config["devices"])) if case.get("reverse_devices") else config["devices"]
        payload = json.loads(native.read_text())
        requested_command = [command[0], "-b", str(record["size"]), "-e", str(record["size"]),
                             "-g", str(len(devices)), "-n", str(config["iterations"]),
                             "-w", str(config["warmup"]), "-c", "1", "-I", "1",
                             "-T", str(config["timeout_s"]), "-J", command[command.index("-J")+1],
                             "-R", str(case.get("registration", 0))]
        if command != requested_command:
            raise ValueError("process command differs from configured experiment")
        environment = {"PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                       "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)), "NCCL_DEBUG": "INFO",
                       "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET", **case.get("env", {})}
        # Library location is machine-specific; the recorded binary/library hashes
        # belong to build provenance. All experimental environment keys are checked.
        for entry in payload.get("env", []):
            if entry.startswith("LD_LIBRARY_PATH="):
                environment["LD_LIBRARY_PATH"] = entry.split("=", 1)[1]
        rows = validate_contract(payload, record["op"], len(devices),
                                 {"command": command, "devices": devices, "environment": environment, "size": record["size"],
                                  "iterations": config["iterations"], "warmup": config["warmup"]})
        if classify(proc, rows) != "pass" or record["status"] != "pass" or rows != record["rows"]:
            raise ValueError("stored results differ from native execution")
        replay.append({**record, "rows": rows})
    if seen != wanted:
        raise ValueError("incomplete sweep")
    stored = json.loads((directory / "summary.json").read_text())
    if stored["status"] != "pass" or stored["statistics"] != summary(replay):
        raise ValueError("stored summary differs from replay")
    return {"status": "pass", "process_runs": len(records), "placements": 2 * len(records),
            "scope": stored["scope"], "qualification": False,
            "note": "Replay of saved execution data; not a fresh GPU run or source authentication."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
