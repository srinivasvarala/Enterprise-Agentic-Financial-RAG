"""Resource guard for `make eval-index`: refuse to start when the Docker VM has too little memory.

Docling and the embedding step on a real 10-K can exhaust a small Docker Desktop VM and be killed
without a useful message. The check runs on the host in the Make target, before `docker compose
run`, so it never needs the stack.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence

# Decimal 8 GB, not 8 GiB: Docker Desktop reports a VM total slightly below the slider value, so a
# binary threshold would refuse a VM that was set to 8 GB.
MIN_DOCKER_MEMORY_BYTES = 8 * 10**9


class DockerMemoryError(Exception):
    """The Docker VM has too little memory, or its size could not be read."""


def docker_vm_memory_bytes() -> int:
    """Total memory of the Docker VM, from `docker info` (read-only)."""
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        return int(out.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise DockerMemoryError(
            "could not read the Docker VM memory (is Docker running?); "
            "start Docker Desktop and run `make eval-index` again."
        ) from exc


def require_docker_memory(
    memory_source: Callable[[], int] = docker_vm_memory_bytes,
    *,
    minimum_bytes: int = MIN_DOCKER_MEMORY_BYTES,
) -> None:
    """Raise `DockerMemoryError` naming the Docker Desktop setting, below the minimum."""
    available = memory_source()
    if available < minimum_bytes:
        raise DockerMemoryError(
            f"the Docker VM has {available / 10**9:.1f} GB of memory; the eval index build needs "
            f"at least {minimum_bytes / 10**9:.0f} GB. In Docker Desktop open "
            "Settings > Resources > Memory limit, raise it to 8 GB or more, apply, "
            "then run `make eval-index` again."
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    memory_source: Callable[[], int] = docker_vm_memory_bytes,
) -> int:
    del argv
    try:
        require_docker_memory(memory_source)
    except DockerMemoryError as exc:
        print(f"eval-index refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
