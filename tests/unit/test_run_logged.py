"""``scripts/run_logged.py``: refuses unknown or disallowed machines, redacts secrets, and logs.

No experiment is run: the one subprocess these tests start is a one-line ``python -c`` that prints
to both streams, to exercise the log writer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rl() -> Any:
    return _load("smc_run_logged", "scripts/run_logged.py")


MACHINES = {
    "box": {"id": "box", "platform": "linux", "requires_gpu": True, "runs_allowed": True},
    "here": {
        "id": "here",
        "platform": "win32",
        "requires_gpu": False,
        "runs_allowed": False,
        "not_allowed_reason": "not until the owner says so",
    },
}


def test_refuses_a_machine_not_in_the_registry(rl: Any) -> None:
    with pytest.raises(rl.RefusedError, match="not in registry"):
        rl.check_machine("elsewhere", MACHINES, platform_name="linux", has_nvidia_smi=True)


def test_refuses_a_machine_whose_runs_are_not_allowed(rl: Any) -> None:
    with pytest.raises(rl.RefusedError, match="not until the owner says so"):
        rl.check_machine("here", MACHINES, platform_name="win32", has_nvidia_smi=False)


def test_refuses_a_platform_mismatch_and_a_missing_gpu(rl: Any) -> None:
    with pytest.raises(rl.RefusedError, match="wrong --machine"):
        rl.check_machine("box", MACHINES, platform_name="win32", has_nvidia_smi=True)
    with pytest.raises(rl.RefusedError, match="nvidia-smi"):
        rl.check_machine("box", MACHINES, platform_name="linux", has_nvidia_smi=False)
    assert rl.check_machine("box", MACHINES, platform_name="linux", has_nvidia_smi=True)


def test_registry_refuses_this_workstation(rl: Any) -> None:
    """registry/machines.yaml keeps the local workstation closed to runs."""
    machines = rl.load_machines()
    assert machines["local-windows"]["runs_allowed"] is False
    with pytest.raises(rl.RefusedError):
        rl.check_machine("local-windows", machines, platform_name="win32", has_nvidia_smi=False)


def test_redactor_removes_credential_values_and_token_shapes(rl: Any) -> None:
    redact = rl.Redactor({"HF_TOKEN": "abcdefgh12345678", "PATH": "/usr/bin/secretish"})
    out = redact(
        "token=abcdefgh12345678 key=sk-ABCDEFGHIJKLMNOPQRSTUVWX hf=hf_ABCDEFGHIJKLMNOPQRSTUV"
    )
    assert "abcdefgh12345678" not in out and "[REDACTED:HF_TOKEN]" in out
    assert "[REDACTED:openai-key]" in out and "[REDACTED:hf-token]" in out
    assert redact("/usr/bin/secretish") == "/usr/bin/secretish"  # PATH is not a credential name


def test_collapse_carriage_returns_keeps_the_final_redraw(rl: Any) -> None:
    assert rl.collapse_carriage_returns(" 10%|#\r 50%|###\r100%|#####") == "100%|#####"
    assert rl.collapse_carriage_returns("plain line") == "plain line"


def test_run_command_logs_both_streams_redacted(rl: Any, tmp_path: Path) -> None:
    redact = rl.Redactor({"API_KEY": "s3cr3tvalue99"})
    code = (
        "import sys; print('step 1 s3cr3tvalue99'); "
        "print('Warning: slow', file=sys.stderr); sys.exit(4)"
    )
    log = tmp_path / "log.txt"
    exit_code, counts, _seconds = rl.run_command([sys.executable, "-c", code], log, redact)
    text = log.read_text(encoding="utf-8")
    assert exit_code == 4
    assert "s3cr3tvalue99" not in text and "[REDACTED:API_KEY]" in text
    assert " out] step 1" in text and " err] Warning: slow" in text
    assert counts == {"lines": 2, "warning_lines": 1, "error_lines": 0}
    assert text.rstrip().endswith("# exit code: 4")


def test_environment_id_matches_the_validator(rl: Any) -> None:
    """The wrapper and scripts/validate.py must compute the same content address."""
    v = _load("smc_validate_for_runs", "scripts/validate.py")
    record, _private = rl.capture_environment("gpu-box", rl.Redactor({}))
    assert v.environment_id(record) == record["env_id"]
    assert json.loads(json.dumps(record)) == record


def test_modal_runs_only_inside_a_modal_container(rl: Any) -> None:
    """`--machine modal` from the launching machine would log the launcher's hardware."""
    machines = rl.load_machines()
    with pytest.raises(rl.RefusedError, match="MODAL_IS_REMOTE"):
        rl.check_machine("modal", machines, platform_name="linux", has_nvidia_smi=True, environ={})
    assert rl.check_machine(
        "modal",
        machines,
        platform_name="linux",
        has_nvidia_smi=True,
        environ={"MODAL_IS_REMOTE": "1"},
    )


def test_gpu_box_refuses_inside_a_modal_container(rl: Any) -> None:
    machines = rl.load_machines()
    with pytest.raises(rl.RefusedError, match="MODAL_TASK_ID"):
        rl.check_machine(
            "gpu-box",
            machines,
            platform_name="linux",
            has_nvidia_smi=True,
            environ={"MODAL_TASK_ID": "ta-123"},
        )
    assert rl.check_machine(
        "gpu-box", machines, platform_name="linux", has_nvidia_smi=True, environ={}
    )
