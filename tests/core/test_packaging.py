import ast
import pathlib
import re


def test_amdgpu_docker_torch_version_is_supported():
    module_root_dir = pathlib.Path(__file__).parents[2]
    dockerfile = (module_root_dir / "docker" / "Dockerfile.amdgpu").read_text()
    genesis_init = ast.parse((module_root_dir / "genesis" / "__init__.py").read_text())

    active_from = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
    version_match = re.fullmatch(
        r"FROM rocm/pytorch:.*_pytorch_release_(\d+)\.(\d+)\.(\d+)",
        active_from,
    )
    assert version_match is not None
    image_version = tuple(map(int, version_match.groups()))

    for node in genesis_init.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_IS_OLD_TORCH" for target in node.targets
        ):
            torch_floor = ast.literal_eval(node.value.comparators[0])
            break
    else:
        raise AssertionError("Genesis PyTorch version threshold not found")

    assert image_version[: len(torch_floor)] >= torch_floor
