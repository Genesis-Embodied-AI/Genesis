import subprocess


def test_gs_clean() -> None:
    # at least check it runs
    subprocess.check_output(["gs", "clean"])
