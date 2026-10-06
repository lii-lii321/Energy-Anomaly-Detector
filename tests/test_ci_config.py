from pathlib import Path

import pytest
from conftest import REPO_ROOT

WORKFLOW_PATH = Path(REPO_ROOT) / ".github" / "workflows" / "ci.yml"


def _load_workflow():
    yaml = pytest.importorskip("yaml")
    assert WORKFLOW_PATH.is_file(), f"missing CI workflow at {WORKFLOW_PATH}"
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict), "ci.yml must parse to a mapping"
    return workflow


def _step_commands(workflow):
    commands = []
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            for key in ("uses", "run"):
                command = step.get(key)
                if command:
                    commands.append(command)
    return commands


def test_ci_workflow_exists_and_is_named_ci():
    assert _load_workflow().get("name") == "CI"


def test_ci_steps_install_requirements_and_run_pytest():
    commands = "\n".join(_step_commands(_load_workflow()))
    assert "pip install -r requirements.txt" in commands
    assert "pytest" in commands


def test_ci_steps_run_ruff_check_before_pytest():
    commands = _step_commands(_load_workflow())
    lint_positions = [i for i, cmd in enumerate(commands) if "ruff check" in cmd]
    test_positions = [i for i, cmd in enumerate(commands) if "pytest" in cmd]

    assert lint_positions, "ci.yml must contain a ruff check step"
    assert test_positions, "ci.yml must contain a pytest step"
    assert min(lint_positions) < max(test_positions), (
        "ruff check step must run before the pytest step"
    )
    assert "python -m ruff check ." in "\n".join(commands)
