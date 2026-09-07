#!/usr/bin/env python3
"""Export sanitized text evidence with original/shared SHA-256 receipts."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"evidence/local")
    p.add_argument("--out",type=Path,default=ROOT/"evidence/snapshot")
    p.add_argument("--verify",type=Path)
    a=p.parse_args()
    if a.verify:
        receipt=json.loads((a.verify/"HASHES.json").read_text())
        for item in receipt["files"]:
            if digest((a.verify/item["path"]).read_bytes())!=item["exported_sha256"]:
                raise SystemExit("hash mismatch: "+item["path"])
        print(f"Verified {len(receipt['files'])} exported evidence files");return
    a.out.mkdir(parents=True,exist_ok=False)
    substitutions=[(str(ROOT),"<PROJECT_ROOT>"),(str(ROOT.parent),"<WORKSPACE>"),
                   (str(Path.home()),"<HOME>"),(socket.gethostname(),"<HOST>")]
    def redact(text):
        for original,replacement in substitutions:text=text.replace(original,replacement)
        return text
    def walk(value):
        if isinstance(value,str):return redact(value)
        if isinstance(value,list):return [walk(v) for v in value]
        if isinstance(value,dict):return {k:walk(v) for k,v in value.items()}
        return value
    files=[]
    for source in sorted(a.source.rglob("*")):
        if not source.is_file() or source.is_symlink():continue
        raw=source.read_bytes()
        try:text=raw.decode("utf-8")
        except UnicodeDecodeError:continue
        if source.suffix==".json":
            try:text=json.dumps(walk(json.loads(text)),indent=2,allow_nan=False)+"\n"
            except ValueError:text=redact(text)
        else:text=redact(text)
        relative=source.relative_to(a.source)
        output=a.out/relative;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
        files.append({"path":str(relative),"original_sha256":digest(raw),"exported_sha256":digest(output.read_bytes())})
    code={str(path.relative_to(ROOT)):digest(path.read_bytes()) for path in ROOT.rglob("*")
          if path.is_file() and path.suffix in {".py",".c",".cu",".h",".patch"}
          and not any(part in {".git","build","evidence","__pycache__"} for part in path.relative_to(ROOT).parts)}
    receipt={"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"files":files,"export_time_source_sha256":code,
             "note":"Run metadata hashes refer to original local bytes. This receipt maps originals to redacted exports. Export-time source hashes are not a claim that every historical run used the final source.",
             "redactions":["project directory","workspace directory","home directory","local hostname"]}
    (a.out/"HASHES.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(f"Exported {len(files)} files to {a.out}")


if __name__=="__main__":main()
