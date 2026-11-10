"""Stage the linux wheels the API image needs into wheelhouse/, using the laptop's own (fast) network.

Docker Desktop on Windows sends build-time downloads through its desktop proxy, which on a slow link
crawls: the 195 MB torch CPU wheel measured 16 kB/s inside `docker build` and 3 MB/s from the host, so
the image could not be built in under three hours. `docker build` then installs from wheelhouse/ with
`--find-links`, which needs no network at all.

    python tools/fetch_api_wheels.py            # idempotent, skips torch if it is already there
    docker compose build                        # picks wheelhouse/ up as build context

The wheels are linux-only on purpose: they are the *image's* wheels, not the laptop's.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELHOUSE = ROOT / "wheelhouse"
REQUIREMENTS = ROOT / "requirements-api.txt"
TORCH = ("torch==2.4.1", "torchvision==0.19.1")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# torch's CPU wheels are tagged linux_x86_64; everything on PyPI is manylinux2014
PYPI_TARGET = ["--only-binary=:all:", "--platform", "manylinux2014_x86_64",
               "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311"]
TORCH_TARGET = ["--only-binary=:all:", "--platform", "linux_x86_64",
                "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311"]


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "download", "--dest", str(WHEELHOUSE), *args]
    print("+", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    WHEELHOUSE.mkdir(exist_ok=True)
    if not any(WHEELHOUSE.glob("torch-*.whl")):
        pip("--no-deps", *TORCH_TARGET, "--index-url", TORCH_INDEX, *TORCH)
    rest = [line for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith(("torch", "#"))]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(rest) + "\n")
        tmp = fh.name
    try:
        pip(*PYPI_TARGET, "-r", tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    total = sum(p.stat().st_size for p in WHEELHOUSE.glob("*.whl"))
    print(f"{len(list(WHEELHOUSE.glob('*.whl')))} wheels, {total / 2**20:.0f} MiB in {WHEELHOUSE}")


if __name__ == "__main__":
    main()
