"""
This script runs from alarm.yml

Terminology/variable names:
- benchmark suite results: the results of running all benchmark tests once, for a specific code base
- the code base could be conceptually:
    - the current code under test
    - some past revision of the code, described by a git commit hash
- there are actually multiple benchmark test suites, identified by a suite_id
    - in this script, we are only interested in the rigid benchmark suite
- metric: the string name of something we are measuring, such as 'runtime_fps'
- configuration parameter: something we vary/control, such as batch_size, or env
- config_params_str: a string like "backend=cpu-n_envs=64", which specifies specific configuration
  parameters, in string format
      - note that, two config_params_str might represent the same configuration, but be different strings,
        because ordering of configuration parameters might be differnet
- config_params_fdict: a frozen dict that represents a specific set of configuration parameters
      - by comparison with config_str, two identical config_params_fdict's always represent the same configuration
      - note that config_params_fdict's are hashable
      - (fdict is an abbreviation for 'frozendict')
- config_param_names: ordered list of the config parameter names, that we have almost certainly derived
  from a config_params_fdict, by simply returning the ordered list of keys (though we may have merged
  such a list over multiple config_params_fdict's)
  - we are prefixing with 'config' to make explicit that this does not include the names of metrics
- 'pipeline format':
   a string having format like:
   "solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5"
"""

from collections import defaultdict
import argparse
import dataclasses
import os, sys, json, math, statistics
from typing import Callable, Iterable, Any
import wandb
from wandb.apis.public import Run
from frozendict import frozendict
from pathlib import Path
import csv


# def parse_kv_pairs_str(kv_pairs_str: str) -> dict[str, str]:
#     kv_pairs_str_l = kv_pairs_str.split("-")
#     kv_pairs = {}
#     for kv_pair_str in kv_pairs_str_l:
#         k, _, v = kv_pair_str.partition("=")
#         kv_pairs[k] = v
#     return kv_pairs


def config_params_str_to_fdict(config_params_str: str) -> frozendict[str, str]:
    """
    Expects a config_params_str in the string format like:
    solver=PBD-backend=cpu-n_envs=128

    Returns this as a frozen dict of key value pairs.

    Note that the values are strings, not converted into numbers.
    """
    kv = {}
    if config_params_str:
        for token in config_params_str.split("-"):
            token = token.strip()
            if token and "=" in token:
                k, v = token.split("=", 1)
                kv[k.strip()] = v.strip()
    return frozendict(kv)


# def normalize_benchmark_id(benchmark_id: str) -> frozendict[str, str]:
#     """
#     Converts a string benchmark id into a frozendict benchmark id, which is
#     hashable.

#     Questoin: why do we do this?
#     """
#     return frozendict(parse_benchmark_id_to_kv_pairs(benchmark_id))


def dicts_to_key_names(dicts: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """
    Merge a tuple of dicts (could be frozen) into a single tuple of keys that:
    - Preserves the relative order of keys within each tuple
    - Gives precedence to later tuples when conflicts arise
    """
    merged_keys: list[str] = list(dicts[-1])
    merged_keys_set = set(merged_keys)
    for dict_ in dicts[:-1]:
        for key in dict_:
            if key not in merged_keys_set:
                merged_keys.append(key)
                merged_keys_set.add(key)
    return tuple(merged_keys)


def build_sort_key(config_param_names: Iterable[str]) -> Callable:
    """
    Builds a sort key function that can be used to order
    dictionaries of values. The sort key function returns
    a list of tuples of (0|1, value | None), where the sequence of
    (0|1, value) matches that of config_param_names and:
    - only keys in config_param_names are considered in the sorting
      (in the context of this script, this lets us ignore the values of
      metrics during sorting)
    - when a param_name is present in the dictionary, the tuple
      contains (0, value), otherwise (1, None)
    
    Since the resulting tuples will be used for sorting, the result
    is that we will first sort the incoming dictionaries by the first param_name,
    then the second, etc
    - for a particular param_name, the dicts without that param_name will
      be placed after the dicts with that param name, since 1 is after 0.
    """
    def sort_key(d: dict[str, Any]):
        nonlocal config_param_names
        key_list = []
        for col in config_param_names:
            if col in d:
                val = d[col]
                key_list.append((0, val))
            else:
                key_list.append((1, None))
        return key_list
    return sort_key


def parse_results_file(results_file_path: Path, metric_keys: Iterable[str]) -> dict[frozendict[str, str], dict[str, float]]:
    """
    results file path should have lines in pipeline format, like:
    solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5
    solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=985.0 | realtime_factor=49.3
    solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=988.0 | realtime_factor=49.4

    This function returns a dict of dicts, something like:
    {
        FrozenDict({"solver": "PBD", "backend": "cpu"}): {
            "compile_time": 2.52,
            "runtime_fps": 990.0,
        }
    }
    So:
    - the keys of the top level dict are frozendict's representing all the key value pairs in a results row
      EXCEPT the metric key value pairs
    - the values are dicts where the keys are names of the metrics in metric_keys, and the values are
      the measured value of that metric

    Conceptually the keys are config_param_fdict's, and the values are a dictionary of metric names and
    values.
    """
    results = {}
    for line in results_file_path.read_text().splitlines():
        config_param_dict = dict(map(str.strip, p.split("=", 1)) for p in line.split("|") if "=" in p)
        metrics = {}
        for k in metric_keys:
            try:
                # removes metric keys from the config param dict, and adds
                # to the metric kv dict
                metrics[k] = float(config_param_dict.pop(k))
            except (ValueError, TypeError, KeyError):
                pass
        config_param_fdict = frozendict(config_param_dict)
        results[config_param_fdict] = metrics
    return results


def fmt_num(v, is_int: bool):
    """
    converts number to string where:
    - ints => displays as int
    - floats => displays to 2 decimal places
    """
    return f"{int(v):,}" if is_int else f"{v:.2f}"


class BenchmarkRunUnderTest:
    """
    This class contains the data about the benchmark run under test, which we will then
    compare with historical data. This data is loaded from text files in pipe format.
    | foo=123 | bar=456 | ...

    Note: currently this class is kind of a mess, but we will make it contain what the previous
    paragraph just described.
    """
    def __init__(self, artifacts_dir: Path, metric_keys: Iterable[str], filename_glob: str) -> None:
        """
        metric_keys: the keys corresponding to values being measured, such as runtime_fps
        filename_glob: how to locate the data files with the data for the benchmark run
        under test.
        """
        self.result_file_paths = list(artifacts_dir.rglob(filename_glob))
        # make sure we do actually have some current benchmark data to read
        assert self.result_file_paths

        self.metric_keys = metric_keys

        # self.results is a dictionary where the keys are config_param_fdict's, and the values
        # are dicts of metric names and values
        self.results: dict[frozendict[str, str], dict[str, float]] = {}
        for self.result_file_path in self.result_file_paths:
            self.results |= parse_results_file(self.result_file_path, self.metric_keys)
        self.config_param_fdict_set = frozenset(self.results.keys())
        assert self.config_param_fdict_set

        # ordered list of the config parameter names
        self.config_param_names = dicts_to_key_names(tuple((tuple(kv.keys())) for kv in self.results.keys()))

    # def ingest_records_by_commit_hash(self, records_by_commit_hash):

    def get_config_param_names(self):
        """
        Returns an ordered list of the config param names (i.e. not including metric names)
        """
        return dicts_to_key_names(tuple((tuple(kv.keys())) for kv in self.results.keys()))


@dataclasses.dataclass
class Table:
    markdown_rows: list[str]
    # rows: list[dict[str, Any]]


class Alarm:
    def __init__(self, args: argparse.Namespace) -> None:
        self.MAX_VALID_REVISIONS = args.max_valid_revisions
        self.MAX_FETCH_REVISIONS = args.max_fetch_revisions

        self.METRICS_TOL = {
            "runtime_fps": args.runtime_fps_regression_tolerance_pct,
            "compile_time": args.compile_time_regression_tolerance_pct,
            "max_mem_mb": args.mem_regression_tolerance_pct,
        }

        self.speed_artifacts_dir = Path(args.speed_artifacts_dir).expanduser().resolve()
        self.mem_artifacts_dir = Path(args.mem_artifacts_dir).expanduser().resolve()
        self.check_body_path = Path(args.check_body_path).expanduser()

        self.csv_out_file_by_metric_name = {
            "runtime_fps": Path(args.csv_runtime_fps_path).expanduser().resolve(),
            "compile_time": Path(args.csv_compile_time_path).expanduser().resolve(),
            "max_mem_mb": Path(args.csv_mem_path).expanduser().resolve(),
        }

        self.SPEED_METRIC_KEYS = ("compile_time", "runtime_fps", "realtime_factor")
        self.MEM_METRIC_KEYS = ("max_mem_mb")

        self.dev_skip_speed = args.dev_skip_speed
        self.dev_allow_all_branches = args.dev_allow_all_branches

        assert "WANDB_API_KEY" in os.environ

        self.ENTITY = os.environ["WANDB_ENTITY"]
        self.PROJECT_OLD = os.environ["WANDB_PROJECT_OLD_FORMAT"]
        self.PROJECT_NEW = os.environ["WANDB_PROJECT_NEW_FORMAT"]

    def run(self) -> int:
        results_under_test_speed = BenchmarkRunUnderTest(artifacts_dir=self.speed_artifacts_dir, metric_keys=self.SPEED_METRIC_KEYS, filename_glob="speed_test*.txt")
        results_under_test_mem = BenchmarkRunUnderTest(artifacts_dir=self.mem_artifacts_dir, metric_keys=self.MEM_METRIC_KEYS, filename_glob="mem_test*.txt")

        # ----- W&B baselines -----

        speed_records_by_commit_hash = {}
        if not self.dev_skip_speed:
            speed_records_by_commit_hash = self.fetch_wandb_data_old_format(benchmark_under_test=results_under_test_speed)
        print('speed_records_by_commit_hash', speed_records_by_commit_hash)

        mem_records_by_commit_hash = self.fetch_wandb_data_new_format(benchmark_under_test=results_under_test_mem)

        # ----- build TWO tables -----

        # Parse benchmark IDs into key-value dicts while preserving order

        reg_found, alert_found = False, False
        table_by_metric_name: dict[str, Table] = {}
        rows_for_csv_by_metric_name = {"runtime_fps": [], "compile_time": [], "max_mem_mb": []}
        info = {}
        reg_found, alert_found = False, False
        for metric, alias, sign, results_under_test_ in (
            ("runtime_fps", "FPS", 1, results_under_test_speed),
            ("compile_time", "compile", -1, results_under_test_speed),
            ("max_mem_mb", "memory", -1, results_under_test_mem),
        ):
            table_by_metric_name[metric], rows_for_csv_by_metric_name[metric], reg_found_, alert_found_ = self.build_table(
                config_param_names=results_under_test_.get_config_param_names(),
                alias=alias,
                metric=metric,
                sign=sign,
                benchmark_run_under_test=results_under_test_,
                records_by_commit_hash=)
            reg_found |= reg_found_
            alert_found |= alert_found_

        # ----- baseline commit list (MD) -----

        # ----- CHECK body (always) -----

        thr_repr = ", ".join(
            f"{alias} ± {self.METRICS_TOL[metric]:.0f}%"
            for metric, alias in (("runtime_fps", "runtime"), ("compile_time", "compile"))
        )

        check_body = "\n".join(
            [
                f"Thresholds: {thr_repr}",
                "",
                "### Runtime FPS",
                *table_by_metric_name["runtime_fps"].markdown_rows,
                "",
                "### Compile Time",
                *table_by_metric_name["compile_time"].markdown_rows,
                "",
                "### Memory usage",
                *table_by_metric_name["max_mem_mb"].markdown_rows,
                "",
                f"- (*1) last: last commit on main, mean/std: stats over commit hashes {self.MAX_VALID_REVISIONS} commits if available.",
                f"- (*2) Δ: relative difference between PR and last commit on main, i.e. (PR - main) / main * 100%.",
            ]
        )

        for metric_ in ("runtime_fps", "compile_time", "max_mem_mb"):
            with self.csv_out_file_by_metric_name[metric_].open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=info.keys())
                w.writeheader()
                for rec in rows_for_csv_by_metric_name[metric_]:
                    w.writerow(rec)

        self.check_body_path.write_text(check_body + "\n", encoding="utf-8")

        if reg_found:
            return int(os.environ["EXIT_CODE_REGRESSION"])

        if alert_found:
            return int(os.environ["EXIT_CODE_ALERT"])

        return 0

    def fetch_wandb_data_loop_common(
        self,
        i: int,
        run: Run,
        commit_hashes: set[str],
    ) -> bool:
        """
        The common part of the loop over runs, that is the same for both
        old and new format

        If returns False, then the calling for loop should do 'continue'
        """
        print("i", i, "run", run)
        # Abort if still not complete after checking enough runs.
        # This would happen if a new benchmark has been added, and not enough past data is available yet.
        if len(commit_hashes) == self.MAX_FETCH_REVISIONS:
            return False

        # Early return if enough complete records have been collected
        records_is_complete = [benchmark_under_test.benchmark_ids_set.issubset(record.keys()) for record in records_by_commit_hash.values()]
        if sum(records_is_complete) == self.MAX_VALID_REVISIONS:
            return False

        # Load config and summary, with support of legacy runs
        config, summary = run.config, run.summary
        if isinstance(config, str):
            config = {k: v["value"] for k, v in json.loads(config).items() if not k.startswith("_")}
        if isinstance(summary._json_dict, str):
            summary = json.loads(summary._json_dict)

        # Extract revision commit and branch
        try:
            commit_hash, branch = config["revision"].split("@", 1)
            commit_hashes.add(commit_hash)
        except ValueError:
            # Ignore this run if the revision has been corrupted for some unknown reason
            return False

        # Ignore runs associated with a commit that is not part of the official repository
        if not branch.startswith('Genesis-Embodied-AI/') and not args.dev_allow_all_branches:
            return False

        # Skip runs did not finish for some reason
        if run.state != "finished":
            return False

        # Do not store new records if the desired number of revision is already reached
        if len(records_by_commit_hash) == self.MAX_VALID_REVISIONS and commit_hash not in records_by_commit_hash:
            return False

        # Extract benchmark ID and normalize it to make sure it does not depends on key ordering.
        # Note that the rigid body benchmark suite is the only one being supported for now.
        suite_id, benchmark_id = config["benchmark_id"].split("-", 1)
        if suite_id != "rigid_body":
            return False

        return True

    def fetch_wandb_data_old_format(
        self,
        benchmark_under_test: BenchmarkRunUnderTest,
    ):
        print("fetch_wandb_data_old_format")
        api = wandb.Api()
        runs_iter = api.runs(f"{self.ENTITY}/{self.PROJECT_OLD}", order="-created_at")
        print('got runs_iter')

        commit_hashes = set()
        records_by_commit_hash = {}
        for i, run in enumerate(runs_iter):
            # Make sure that stats are valid
            try:
                is_valid = True
                for k in self.SPEED_METRIC_KEYS:
                    v = summary[k]
                    if not isinstance(v, (float, int)) or math.isnan(v):
                        is_valid = False
                        break
                if not is_valid:
                    continue
            except KeyError:
                continue

            # Store all the records into a dict
            nbid = normalize_benchmark_id(benchmark_id)
            records_by_commit_hash.setdefault(commit_hash, {})[nbid] = {
                metric: summary[metric] for metric in self.SPEED_METRIC_KEYS
            }
            return records_by_commit_hash

    def fetch_wandb_data_new_format(
        self,
        benchmark_under_test: BenchmarkRunUnderTest,
    ):
        print("fetch_wandb_data_new_format")
        api = wandb.Api()
        runs_iter = api.runs(f"{self.ENTITY}/{self.PROJECT_NEW}", order="-created_at")
        print('got runs_iter')

        commit_hashes = set()
        records_by_commit_hash = defaultdict(lambda: defaultdict(dict))
        for i, run in enumerate(runs_iter):
            print("i", i, "run", run)
            # Abort if still not complete after checking enough runs.
            # This would happen if a new benchmark has been added, and not enough past data is available yet.
            if len(commit_hashes) == self.MAX_FETCH_REVISIONS:
                break

            # Early return if enough complete records have been collected
            records_is_complete = [benchmark_under_test.benchmark_ids_set.issubset(record.keys()) for record in records_by_commit_hash.values()]
            if sum(records_is_complete) == self.MAX_VALID_REVISIONS:
                break

            # Load config and summary, with support of legacy runs
            config, summary = run.config, run.summary
            if isinstance(config, str):
                config = {k: v["value"] for k, v in json.loads(run.config).items() if not k.startswith("_")}
            if isinstance(summary._json_dict, str):
                summary = json.loads(summary._json_dict)

            # Extract revision commit and branch
            try:
                commit_hash, branch = config["revision"].split("@", 1)
                commit_hashes.add(commit_hash)
            except ValueError:
                print('didnt find rev')
                # Ignore this run if the revision has been corrupted for some unknown reason
                continue
            print("commit_hash", commit_hash, "branch", branch)
            # Ignore runs associated with a commit that is not part of the official repository
            if not branch.startswith('Genesis-Embodied-AI/') and not self.dev_allow_all_branches:
                print('branch didnt start with Genesis-Embodied-AI')
                continue

            # Skip runs did not finish for some reason
            if run.state != "finished":
                continue

            # Do not store new records if the desired number of revision is already reached
            if len(records_by_commit_hash) == self.MAX_VALID_REVISIONS and commit_hash not in records_by_commit_hash:
                continue

            for k, v in summary.items():
                if k.startswith("_"):
                    continue
                metric_name, _, kv_pairs_str = k.partition("-")
                records_by_commit_hash[commit_hash][kv_pairs_str][metric_name] = v                

            print('records_by_commit_hash', records_by_commit_hash)
            for commit_hash, records in records_by_commit_hash.items():
                print('commit_hash')
                for k, v in records.items():
                    print("- ", "record", k, v)
            # adsfasdf
            return records_by_commit_hash

    def build_table(
        self,
        config_param_names: str,
        alias: str,
        metric: str,
        benchmark_run_under_test: BenchmarkRunUnderTest,
        records_by_commit_hash: dict[str, Any],
        sign: int,
    ) -> tuple[Table, list[dict[str, Any]], bool, bool]:
        # together these rows contain the text of the markdwon
        markdown_rows = []
        rows = []
        alert_found, reg_found = False, False

        # the labels in the header row of the table
        header_cells = (
            "status",
            *config_param_names,
            f"current {alias}",
            f"baseline {alias} [last (mean ± std)] (*1)",
            f"Δ {alias} (*2)"
        )
        header = "| " + " | ".join(header_cells) + " |"
        align  = "|:------:|" + "|".join([":---" for _ in config_param_names]) + "|---:|---:|---:|"

        for benchmark_id in sorted(benchmark_run_under_test.current_bm.keys(), key=sort_key):
            value_cur = benchmark_run_under_test.current_bm[benchmark_id][metric]
            is_int = isinstance(value_cur, int) or value_cur.is_integer()
            value_repr = fmt_num(value_cur, is_int)

            params_repr = [benchmark_id.get(k, "-") for k in config_param_names]
            info = {
                **dict(zip(config_param_names, params_repr)),
                "current": value_cur,
                "baseline_last": None,
                "baseline_mean": None,
                "baseline_min": None,
                "baseline_max": None,
                "status": None,
            }

            values_prev = [
                record[benchmark_id][metric]
                for record in records_by_commit_hash.values()
                if benchmark_id in record
            ]
            if values_prev:
                value_last = values_prev[0]
                value_ref = statistics.fmean(values_prev)
                delta = (value_cur - value_last) / value_last * 100.0

                info["baseline_last"] = int(value_last) if is_int else float(value_last)

                stats_repr = f"{fmt_num(value_last, is_int)}"
                delta_repr = f"{delta:+.1f}%"
                if len(values_prev) == self.MAX_VALID_REVISIONS:
                    info["baseline_mean"] = int(value_ref) if is_int else float(value_ref)
                    info["baseline_min"] = int(min(values_prev)) if is_int else float(min(values_prev))
                    info["baseline_max"] = int(max(values_prev)) if is_int else float(max(values_prev))

                    value_std = statistics.stdev(values_prev)
                    stats_repr += f" ({fmt_num(value_ref, is_int)} ± {fmt_num(value_std, is_int)})"
                    if sign * delta < - self.METRICS_TOL[metric]:
                        info["status"] = "regression"

                        delta_repr = f"**{delta_repr}**"
                        picto = "🔴"
                        reg_found = True
                    elif sign * delta > self.METRICS_TOL[metric]:
                        info["status"] = "alert"

                        delta_repr = f"**{delta_repr}**"
                        picto = "⚠️"
                        alert_found = True
                    else:
                        info["status"] = "ok"

                        picto = "✅"
                else:
                    info["status"] = "n/a"

                    picto = "ℹ️"
            else:
                picto, stats_repr, delta_repr = "ℹ️", "---", "---"

            markdown_rows.append("| " + " | ".join((picto, *params_repr, value_repr, stats_repr, delta_repr)) + " |")
            rows.append(info)

        blist = [f"- Commit {i}: {sha}" for i, sha in enumerate(records_by_commit_hash.keys(), 1)]
        baseline_block = ["**Baselines considered:** " + f"**{len(records_by_commit_hash)}** commits"] + blist

        # return [header, align] + markdown_rows, rows
        return Table(markdown_rows=[header, align] + markdown_rows + baseline_block), rows, reg_found, alert_found
        # tables[metric] = [header, align] + rows_md


def main() -> None:
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-artifacts-dir", type=str, required=True)
    parser.add_argument("--mem-artifacts-dir", type=str, required=True)
    parser.add_argument("--max-valid-revisions", type=int, default=10, help="limits how many git commits are used to build the baseline statistics")
    parser.add_argument("--max-fetch-revisions", type=int, default=10)
    parser.add_argument("--runtime-fps-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--compile-time-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--mem-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--check-body-path", type=str, required=True)
    parser.add_argument("--csv-runtime-fps-path", type=str, required=True)
    parser.add_argument("--csv-compile-time-path", type=str, required=True)
    parser.add_argument("--csv-mem-path", type=str, required=True)
    parser.add_argument("--exit-code-regression", type=int, default=42)
    parser.add_argument("--exit-code-alert", type=int, default=43)
    parser.add_argument("--dev-skip-speed", action="store_true")
    parser.add_argument("--dev-allow-all-branches", action="store_true")
    args = parser.parse_args()

    alarm = Alarm(args=args)
    exit_code = alarm.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
