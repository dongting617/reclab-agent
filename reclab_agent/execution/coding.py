"""Fixed, offline CPU container execution for small functional coding checks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

DRIVER = """import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('candidate','/workspace/src/solution.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=json.load(sys.stdin)
print(json.dumps([module.solve(case) for case in cases],allow_nan=False))
"""


class CodingContainer:
    def __init__(self, image, *, docker="docker", timeout=10, output_limit=131072):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("coding execution requires a locally verified immutable image ID")
        if os.name != "posix":
            raise ValueError("run coding containers from Linux/WSL")
        self.image, self.docker = image, docker
        self.timeout, self.output_limit = timeout, output_limit

    def run(self, workspace, inputs):
        root = Path(workspace).absolute()
        if any(c in str(root) for c in ",\n\r"):
            raise ValueError("invalid container mount path")
        name = "reclab-coding-" + uuid.uuid4().hex
        argv = [self.docker, "create", "--pull=never", "--name", name, "--interactive",
                "--network=none", "--ipc=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=128m", "--cpus=1",
                "--user=65532:65532", "--mount", f"type=bind,src={root},dst=/workspace,readonly",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m", "--workdir", "/workspace",
                "--entrypoint", "/usr/local/bin/python", self.image, "-E", "-s", "-B", "-c", DRIVER]
        started = time.monotonic()
        failure = None
        with tempfile.TemporaryDirectory(prefix="reclab-coding-log-") as directory:
            stdout, stderr = Path(directory)/"stdout", Path(directory)/"stderr"
            input_file = Path(directory)/"input.json"
            payload = json.dumps(inputs, allow_nan=False).encode()
            if len(payload) > 65536:
                raise ValueError("coding input exceeds its bounded envelope")
            input_file.write_bytes(payload)
            created = subprocess.run(argv,capture_output=True,timeout=15)
            if created.returncode:
                return {"outputs":None,"error":"container_create","exit_code":created.returncode,
                        "elapsed_ms":round((time.monotonic()-started)*1000,3),"stderr":created.stderr[:2000].decode(errors="replace")}
            # This separate host process survives a controller crash. Create the
            # container before starting it so a late start cannot miss cleanup.
            watchdog_code = "import subprocess,sys,time;time.sleep(float(sys.argv[1]));subprocess.run([sys.argv[2],'rm','-f',sys.argv[3]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)"
            watchdog = subprocess.Popen([sys.executable,"-I","-B","-c",watchdog_code,
                str(self.timeout+2),self.docker,name],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            def cap_logs():
                import resource
                resource.setrlimit(resource.RLIMIT_FSIZE,(self.output_limit,self.output_limit))
            with input_file.open("rb") as inp, stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen([self.docker,"start","--attach","--interactive",name],
                    stdin=inp,stdout=out,stderr=err,preexec_fn=cap_logs)
                try:
                    while process.poll() is None:
                        if time.monotonic()-started > self.timeout:
                            failure = "timeout";break
                        if stdout.stat().st_size+stderr.stat().st_size > self.output_limit:
                            failure = "output_limit";break
                        time.sleep(.03)
                except BrokenPipeError:
                    failure = "container_exit"
                finally:
                    subprocess.run([self.docker,"rm","-f",name],capture_output=True,timeout=10)
                    if process.poll() is None:process.kill()
                    process.wait(timeout=10)
                    if watchdog.poll() is None:watchdog.terminate()
                    watchdog.wait(timeout=3)
            if stdout.stat().st_size+stderr.stat().st_size > self.output_limit:
                failure = "output_limit"
            if process.returncode and failure is None:
                failure = "container_exit"
            outputs = None
            if failure is None:
                try:
                    outputs = json.loads(stdout.read_bytes())
                    if not isinstance(outputs,list) or len(outputs)!=len(inputs):
                        failure = "invalid_output"
                except (ValueError,UnicodeError):
                    failure = "invalid_output"
            return {"outputs": outputs, "error": failure, "exit_code": process.returncode,
                    "elapsed_ms": round((time.monotonic()-started)*1000,3),
                    "stderr": stderr.read_bytes()[:2000].decode("utf-8",errors="replace")}
