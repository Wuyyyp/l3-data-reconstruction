import importlib
from pathlib import Path

import tomllib


def test_project_script_entrypoints_are_importable():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"].get("scripts", {})

    missing = []
    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing.append(f"{name}: {module_name} ({error})")
            continue
        if attribute and not hasattr(module, attribute):
            missing.append(f"{name}: missing attribute {attribute}")

    assert missing == []
