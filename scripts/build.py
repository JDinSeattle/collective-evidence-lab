#!/usr/bin/env python3
"""Build pinned NCCL and nccl-tests in a whitespace-free temporary directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir",type=Path,default=Path(tempfile.gettempdir())/("collective-lab-"+hashlib.sha256(str(ROOT).encode()).hexdigest()[:8]))
    p.add_argument("--arch",default="89",choices=["80","86","89","90","100","120"])
    p.add_argument("--cuda",type=Path,default=Path("/usr/local/cuda"))
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args();work=a.work_dir.resolve()
    if any(c.isspace() for c in str(work)):p.error("upstream Makefiles require a whitespace-free build path")
    work.mkdir(parents=True,exist_ok=True);out=ROOT/"evidence/local";out.mkdir(parents=True,exist_ok=True)
    with (out/"build.log").open("w") as log:
        def run(cmd,cwd=work):
            log.write("$ "+repr(list(map(str,cmd)))+"\n");log.flush()
            subprocess.run(list(map(str,cmd)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        lock=json.loads((ROOT/"upstream.lock.json").read_text())
        for name,item in lock.items():
            src=work/name
            if not src.exists():
                run(["git","init",src]);run(["git","fetch","--depth=1",item["repository"],item["revision"]],src)
                run(["git","checkout","--detach","FETCH_HEAD"],src)
            actual=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
            if actual!=item["revision"]:raise RuntimeError("revision mismatch: "+name)
        gencode=f"NVCC_GENCODE=-gencode=arch=compute_{a.arch},code=sm_{a.arch}"
        run(["make",f"-j{a.jobs}","src.build",f"CUDA_HOME={a.cuda}",gencode],work/"nccl")
        run(["make",f"-j{a.jobs}",f"CUDA_HOME={a.cuda}",f"NCCL_HOME={work/'nccl/build'}",gencode],work/"nccl-tests")
        (ROOT/"build").mkdir(exist_ok=True)
        run([a.cuda/"bin/nvcc","-std=c++17","-O3",f"-arch=sm_{a.arch}",ROOT/"src/preflight.cu","-o",ROOT/"build/preflight"])
    paths={"bin_dir":str(work/"nccl-tests/build"),"library_dir":str(work/"nccl/build/lib"),"preflight":str(ROOT/"build/preflight"),"arch":a.arch,"lock":lock}
    (out/"build.json").write_text(json.dumps(paths,indent=2)+"\n");print(json.dumps(paths,indent=2))


if __name__=="__main__":main()
