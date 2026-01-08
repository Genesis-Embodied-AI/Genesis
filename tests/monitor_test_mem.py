from collections import defaultdict
import csv
import subprocess
import time
import os
import argparse
import psutil


CHECK_INTERVAL = 2.0


def grep(contents: list[str], target):
    return [l for l in contents if target in l]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv-filepath", type=str, required=True)
    parser.add_argument("--die-with-parent", action="store_true")
    args = parser.parse_args()

    max_mem_by_test = defaultdict(int)

    f = open(args.out_csv_filepath, "w")
    dict_writer = csv.DictWriter(f, fieldnames=["test", "max_mem_mb"])
    dict_writer.writeheader()
    old_mem_by_test = {}
    num_results_written = 0
    disp = False
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
                dict_writer.writerow({"test": _test, "max_mem_mb": max_mem_by_test[_test]})
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
        disp = not disp
        time.sleep(CHECK_INTERVAL)
    print("Test monitor exiting")


if __name__ == "__main__":
    main()
