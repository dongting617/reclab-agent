"""SSH function-tool schemas and registration over the transport client."""
from __future__ import annotations
from typing import Any, Callable
from reclab_agent.execution.ssh import SSHClient
from reclab_agent.tools.contracts import ToolError, ToolSpec
from reclab_agent.tools.registry import ToolRegistry

class RemoteTools:
    def __init__(self, *args, **kwargs):
        self.client = SSHClient(*args, **kwargs)
        self._tools = ToolRegistry()
        string = {"type": "string"}
        self._register("upload", "Upload one registered source or public-data file and verify its SHA-256.", {
            "source_root": {"type": "string", "enum": sorted(self.upload_roots)},
            "source": string, "destination": string,
        }, self.upload)
        self._register("download", "Download one bounded artifact and verify its SHA-256 before saving it.", {
            "source": string, "destination": string,
        }, self.download)
        self._register("run_remote", "Start or recover one stable tmux job; use job_status to observe its explicit exit code.", {
            "job_id": string, "argv": {"type": "array", "items": string, "minItems": 2},
            "cwd": string, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": self.max_timeout_seconds},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2_147_483_647},
        }, self.run_remote)
        self._register("job_status", "Read a persisted job status; tmux absence alone does not mean success.",
                       {"job_id": string}, self.job_status)
        self._register("tail_log", "Read a bounded diagnostic tail of the job log.", {
            "job_id": string, "lines": {"type": "integer", "minimum": 1, "maximum": 1000},
        }, self.tail_log)
        self._register("gpu_status", "Read current GPU availability from nvidia-smi.", {}, self.gpu_status)
        self._register("stop_job", "Stop the matching job process group and retain its final status.",
                       {"job_id": string}, self.stop_job)

    def _register(self, name: str, description: str, properties: dict[str, Any], handler: Callable) -> None:
        def invoke(**arguments: Any) -> dict[str, Any]:
            try:
                return handler(**arguments)
            except OSError as error:
                raise ToolError("LOCAL_IO_ERROR", str(error)) from error

        self._tools.register_function(ToolSpec(name, description, {
            "type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False,
        }), invoke)

    @property
    def specs(self) -> list[ToolSpec]:
        return self._tools.specs

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._tools.invoke(name, arguments)


    def __getattr__(self, name):
        return getattr(self.client, name)

    def __setattr__(self, name, value):
        if name not in {"client", "_tools"} and "client" in self.__dict__ and hasattr(self.client, name):
            setattr(self.client, name, value)
        else:
            object.__setattr__(self, name, value)
