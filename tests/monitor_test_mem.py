from collections import defaultdict
import subprocess
import time
import os
import argparse
import psutil
import re


CHECK_INTERVAL = 2.0


def grep(contents: list[str], target):
    return [l for l in contents if target in l]


def parse_test_name(test_name: str) -> dict[str, str]:
    """
    Expected format: test_speed[env-constraint_solver-gjk_collision-batch_size-backend]
    Example: test_speed[franka-None-True-30000-cuda]
    Returns:
        dict: Parsed parameters
    """
    match = re.search(r"\[(.*?)\]", test_name)
    if not match:
        return {}

    parts = match.group(1).split("-")
    if len(parts) < 5:
        return {}

    params = {
        "env": parts[0],
        "constraint_solver": parts[1],
        "gjk_collision": parts[2],
        "batch_size": parts[3],
        "backend": parts[4],
        "dtype": parts[5],
    }

    # Remove "None" values for consistency
    filtered_params = {}
    for k, v in params.items():
        if v != "None" and v is not None:
            filtered_params[k] = v

    return filtered_params


def get_cuda_usage() -> dict[int, int]:
    output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    section = 0
    subsec = 0
    res = {}
    for line in output.split("\n"):
        if line.startswith("|============"):
            section += 1
            subsec = 0
            continue
        if line.startswith("+-------"):
            subsec += 1
            continue
        if section == 2 and subsec == 0:
            if "No running processes" in line:
                continue
            split_line = line.split()
            pid = int(split_line[4])
            mem = int(split_line[-2].split("MiB")[0])
            res[pid] = mem
    return res


def get_test_name_by_pid() -> dict[int, str]:
    test_by_psid = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline is None:
                continue
            # Join cmdline to get full command string
            cmd_str = " ".join(cmdline)
            if "pytest: tests" in cmd_str:
                # Find the test name after "::"
                if "::" in cmd_str:
                    test_name = cmd_str.partition("::")[2]
                    if test_name.strip() != "":
                        test_by_psid[proc.info["pid"]] = test_name
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process may have terminated or we don't have permission
            pass
    return test_by_psid


def format_result_line(test_name: str, max_mem_mb: int) -> str:
    """Format a result line in pipe-delimited format."""
    params = parse_test_name(test_name)
    params["max_mem_mb"] = str(max_mem_mb)

    line_parts = [f"{k}={v}" for k, v in params.items()]
    return " \t| ".join(line_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--die-with-parent", action="store_true")
    args = parser.parse_args()

    max_mem_by_test = defaultdict(int)

    f = open(args.out_file, "w")
    old_mem_by_test = {}
    num_results_written = 0
    last_output_line = None
    while not args.die_with_parent or os.getppid() != 1:
        mem_by_pid = get_cuda_usage()
        test_by_psid = get_test_name_by_pid()
        num_tests = len(test_by_psid)
        _mem_by_test = {}
        for psid, test in test_by_psid.items():
            if psid not in mem_by_pid:
                continue
            if test.strip() == "":
                continue
            _mem = mem_by_pid[psid]
            _mem_by_test[test] = _mem
        for test, _mem in _mem_by_test.items():
            max_mem_by_test[test] = max(_mem, max_mem_by_test[test])
        for _test, _mem in old_mem_by_test.items():
            if _test not in _mem_by_test:
                result_line = format_result_line(_test, max_mem_by_test[_test])
                f.write(result_line + "\n")
                f.flush()
                num_results_written += 1
        potential_output_line = (
            f"{num_tests} tests running, of which {len(_mem_by_test)} on gpu. "
            f"Num results written: {num_results_written} [updating]         "
        )
        if potential_output_line != last_output_line:
            print(potential_output_line, end="\r", flush=True)
            last_output_line = potential_output_line
        old_mem_by_test = _mem_by_test
        time.sleep(CHECK_INTERVAL)
    print("Test monitor exiting")


if __name__ == "__main__":
    main()
