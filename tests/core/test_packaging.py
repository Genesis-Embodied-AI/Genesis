import ast
import pathlib
import re


def test_amdgpu_docker_torch_version_is_supported():
    module_root_dir = pathlib.Path(__file__).parents[2]
    dockerfile = (module_root_dir / "docker" / "Dockerfile.amdgpu").read_text()
    genesis_init = ast.parse((module_root_dir / "genesis" / "__init__.py").read_text())

    active_from = next(
        (line for line in dockerfile.splitlines() if line.startswith("FROM ")),
        None,
    )
    assert active_from is not None, "Dockerfile.amdgpu has no active FROM instruction"
    version_match = re.fullmatch(
        r"FROM rocm/pytorch:rocm7\.2\.4_ubuntu22\.04_py3\.10_pytorch_release_"
        r"(\d+)\.(\d+)\.(\d+)@sha256:"
        r"880e126d83370e3502b069a39f85cbd7b1f6f7dbfbceb00b6fefbac03c5da091",
        active_from,
    )
    assert version_match is not None, "AMD Docker image tag or digest is not the registered image"
    image_version = tuple(map(int, version_match.groups()))

    old_torch_assign = next(
        (
            node
            for node in genesis_init.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_IS_OLD_TORCH"
                for target in node.targets
            )
        ),
        None,
    )
    assert isinstance(old_torch_assign, ast.Assign), "Genesis PyTorch version threshold assignment not found"
    comparison = old_torch_assign.value
    assert isinstance(comparison, ast.Compare), "_IS_OLD_TORCH must be a comparison"
    assert len(comparison.comparators) == 1, "_IS_OLD_TORCH must have one comparator"
    assert len(comparison.ops) == 1, "_IS_OLD_TORCH must have one comparison operator"
    assert isinstance(comparison.ops[0], ast.Lt), "_IS_OLD_TORCH must use a less-than comparison"
    torch_floor = ast.literal_eval(comparison.comparators[0])

    assert image_version[: len(torch_floor)] >= torch_floor
