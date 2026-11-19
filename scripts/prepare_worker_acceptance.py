"""Validate a preloaded Docker digest and record non-secret acceptance provenance.

This trusted preparation command never pulls/builds an image or runs a candidate.
It is deliberately limited to the official Python stdlib acceptance image. A real
model image needs its own administrator review, not a different CLI argument here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


OFFICIAL_PYTHON_DIGEST = re.compile(
    r"(?:(?:docker\.io/library/|library/)?python|public\.ecr\.aws/docker/library/python)@sha256:[a-f0-9]{64}\Z"
)


def approved_digest(value: str) -> str:
    if not OFFICIAL_PYTHON_DIGEST.fullmatch(value):
        raise ValueError("acceptance requires an explicit official Python repository@sha256 digest")
    return value


def docker_json(arguments: list[str], *, config: Path) -> object:
    executable = shutil.which("docker")
    if sys.platform != "linux" or executable is None:
        raise RuntimeError("real acceptance requires Linux and a local Docker CLI; no host fallback")
    result = subprocess.run(
        [executable, "--host", "unix:///var/run/docker.sock", *arguments],
        capture_output=True, text=True, timeout=30, check=False, shell=False,
        env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
             "LANG": "C.UTF-8", "DOCKER_CONFIG": str(config)},
    )
    if result.returncode:
        raise RuntimeError("Docker preparation failed: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def prepare(image: str, evidence: Path) -> dict:
    image = approved_digest(image)
    evidence.mkdir(parents=True, exist_ok=True)
    config = evidence / "docker-client"
    config.mkdir(exist_ok=True)
    (config / "config.json").write_text('{"auths":{}}', encoding="utf-8")
    images = docker_json(["image", "inspect", "--", image], config=config)
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("expected exactly one preloaded image")
    actual = images[0]
    if image not in actual.get("RepoDigests", []) or actual.get("Config", {}).get("Volumes"):
        raise RuntimeError("preloaded image digest differs or declares writable volumes")
    info = docker_json(["info", "--format", "{{json .}}"], config=config)
    if info.get("OSType") != "linux":
        raise RuntimeError("a local Linux Docker daemon is required")
    files = ["reclab_agent/execution.py", "reclab_agent/watchdog.py", "reclab_agent/grading.py", "tests/test_real_execution.py",
             "scripts/prepare_worker_acceptance.py"]
    repository = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": 1, "prepared_at": datetime.now(timezone.utc).isoformat(),
        "scope": "reviewed stdlib synthetic CPU execution acceptance; not GPU or model improvement",
        "requested_image": image, "image_id": actual.get("Id"),
        "repo_digests": actual.get("RepoDigests", []),
        "image_os": actual.get("Os"), "image_architecture": actual.get("Architecture"),
        # Preserve the real repository URI. ECR publication is official, but
        # an ECR digest must never be relabelled as a Docker Hub RepoDigest.
        "image_repository": image.split("@", 1)[0],
        "image_registry": "public.ecr.aws" if image.startswith("public.ecr.aws/") else "docker.io",
        "image_source": ("https://gallery.ecr.aws/docker/library/python"
                         if image.startswith("public.ecr.aws/") else "https://hub.docker.com/_/python"),
        "official_distribution_reference": ("https://www.docker.com/blog/news-from-aws-reinvent-docker-official-images-on-amazon-ecr-public/"
                                            if image.startswith("public.ecr.aws/") else "https://hub.docker.com/_/python"),
        "daemon": {key: info.get(key) for key in (
            "ServerVersion", "OSType", "Architecture", "KernelVersion", "CgroupVersion",
            "SecurityOptions", "MemoryLimit", "SwapLimit", "CpuCfsPeriod", "CpuCfsQuota", "PidsLimit")},
        "source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in files},
        "ci": {key: os.environ[key] for key in (
            "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY") if key in os.environ},
        "real_container_tested": False,
    }
    (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="already pulled official Python repository digest")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.image, args.evidence)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2
    print(json.dumps({"ok": True, "image": result["requested_image"], "real_container_tested": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
