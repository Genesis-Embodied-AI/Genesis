from collections import defaultdict
import csv
import subprocess
import time
import os
import argparse


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
    ps_ef = subprocess.check_output(["ps", "-ef"]).decode("utf-8").split("\n")
    test_lines = grep(ps_ef, "pytest-xdist")
    tests = [line.partition("::")[2] for line in test_lines]
    psids = [int(line.split()[1]) for line in test_lines]
    test_by_psid = {psid: test for test, psid in zip(tests, psids) if test.strip() != ""}
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
        spinny = "x" if disp else "+"
        print(
            num_tests,
            "tests running, of which",
            len(_mem_by_test),
            "on gpu. Num results written: ",
            num_results_written,
            "[updating]",
            "       ",
            end="\r",
            flush=True,
        )
        old_mem_by_test = _mem_by_test
        disp = not disp
        time.sleep(1.0)
    print("Test monitor exiting")


if __name__ == "__main__":
    main()
