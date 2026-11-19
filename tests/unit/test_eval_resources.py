"""Seam: `make eval-index` refuses to start when the Docker VM has too little memory."""

import pytest

from finrag.evaluation.resources import (
    MIN_DOCKER_MEMORY_BYTES,
    DockerMemoryError,
    main,
    require_docker_memory,
)

GB = 10**9


def test_enough_memory_passes() -> None:
    require_docker_memory(lambda: 16 * GB)


def test_exactly_the_minimum_passes() -> None:
    require_docker_memory(lambda: MIN_DOCKER_MEMORY_BYTES)


def test_too_little_memory_names_the_docker_desktop_setting() -> None:
    with pytest.raises(DockerMemoryError) as excinfo:
        require_docker_memory(lambda: 4 * GB)

    message = str(excinfo.value)
    assert "Docker Desktop" in message
    assert "Settings > Resources > Memory limit" in message
    assert "8 GB" in message
    assert "4.0 GB" in message


def test_the_threshold_is_injectable() -> None:
    require_docker_memory(lambda: 2 * GB, minimum_bytes=2 * GB)
    with pytest.raises(DockerMemoryError):
        require_docker_memory(lambda: 2 * GB - 1, minimum_bytes=2 * GB)


def test_the_command_exits_nonzero_with_the_message_when_short(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(memory_source=lambda: 4 * GB)

    assert code == 1
    assert "Memory limit" in capsys.readouterr().err


def test_the_command_exits_zero_when_enough() -> None:
    assert main(memory_source=lambda: 16 * GB) == 0


def test_an_unreadable_docker_refuses_rather_than_guessing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken() -> int:
        raise DockerMemoryError("could not read the Docker VM memory (is Docker running?)")

    assert main(memory_source=broken) == 1
    assert "Docker running" in capsys.readouterr().err
