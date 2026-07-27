from __future__ import annotations

import tomllib
from pathlib import Path

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment

ROOT = Path(__file__).resolve().parents[2]


def test_robocasa_production_code_has_one_action_owner() -> None:
    source_files = list((ROOT / "src" / "hey_robot").rglob("*.py"))
    callers = [
        path
        for path in source_files
        if "manager.step" in path.read_text(encoding="utf-8")
    ]
    assert callers == [
        ROOT / "src" / "hey_robot" / "robocasa_backend" / "runtime_server.py"
    ]


def test_foundation_policy_does_not_import_environment_owner() -> None:
    source = (
        ROOT
        / "src"
        / "hey_robot"
        / "foundation"
        / "backends"
        / "lerobot"
        / "executor.py"
    ).read_text(encoding="utf-8")
    assert "from hey_robot.robocasa_backend.episode_manager" not in source
    assert "import EpisodeManager" not in source


def test_agent_surface_has_only_generic_manipulate() -> None:
    config = DeploymentConfig.from_yaml(
        ROOT / "configs" / "evaluation" / "robocasa365.yaml"
    )
    assert config.skills.tools == ("inspect_scene", "manipulate")
    assert all(
        "robocasa_option" not in spec.provides
        for spec in config.model_services.values()
    )
    assert config.model_services["robocasa365"].settings["prompt_mode"] == (
        "environment_root"
    )
    assert not [
        issue for issue in validate_deployment(config) if issue.level == "error"
    ]


def test_lerobot_has_one_production_executor() -> None:
    root = ROOT / "src" / "hey_robot" / "foundation" / "backends" / "lerobot"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert source.count("class LeRobotPolicyExecutor:") == 1
    assert "LeRobotVLAExecutor" not in source
    assert "LeRobotVLAPolicyExecutor" not in source
    assert "RoboCasaLeRobotPolicyExecutor" not in source


def test_lerobot_is_a_runtime_backend_not_a_vla_subtype() -> None:
    backends = ROOT / "src" / "hey_robot" / "foundation" / "backends"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (backends / "lerobot").glob("*.py")
    )

    assert not list((backends / "vla").rglob("*.py"))
    assert "EXPECTED_INPUTS" not in source
    assert "SUPPORTED_STATE_SHAPES" not in source
    assert "fastwam" not in source


def test_benchmark_selects_the_generic_lerobot_service() -> None:
    source = (ROOT / "evaluation/robocasa365/full_system_benchmark.py").read_text(
        encoding="utf-8"
    )

    assert 'spec.type == "robot_policy"' in source
    assert 'spec.settings.get("runtime")' in source
    assert 'spec.settings.get("embodiment")' in source
    assert "robocasa_lerobot_policy" not in source


def test_production_never_imports_evaluation_worker() -> None:
    roots = [ROOT / "src" / "hey_robot", ROOT / "docker"]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "evaluation.robocasa365.worker" not in text
    assert not (ROOT / "evaluation" / "robocasa365" / "worker").exists()


def test_robocasa_dependencies_have_one_locked_group() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["dependency-groups"]["robocasa365"]
    text = "\n".join(dependencies)
    assert "cb73cf3ffa1cec60640a06b924c2174548ae2b1b" in text
    assert "56e355ccc64389dfc1b8a61a33b9127b975ba681" in text
    assert "aaa8b9b214ce8e77e82926d677b4d61d55e577ab" in text
    assert "torch==2.7.1" in text
    assert "mujoco==3.3.1" in text
    assert "grpcio==1.73.1" in text
    assert "tianshou" not in text
    for policy_or_dataset_dependency in (
        "transformers",
        "sentencepiece",
        "datasets",
        "pandas",
        "pyarrow",
        "jsonlines",
        "av>=",
    ):
        assert policy_or_dataset_dependency in text


def test_local_and_docker_backends_consume_the_locked_group() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.robocasa365").read_text(
        encoding="utf-8"
    )
    setup = (ROOT / "scripts" / "evaluation" / "setup_robocasa365_env.sh").read_text(
        encoding="utf-8"
    )
    command = "uv sync --frozen --only-group robocasa365 --no-install-project"
    assert command in dockerfile
    assert command in setup.replace("\\\n", "")
    assert "uv pip install" not in dockerfile
    assert not (ROOT / "docker" / "requirements.robocasa365.txt").exists()
