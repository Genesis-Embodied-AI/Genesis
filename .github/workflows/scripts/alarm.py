# runs from alarm.yml

from collections import defaultdict
import argparse
import os, sys, json, math, statistics
from typing import Iterable
import wandb
from frozendict import frozendict
from pathlib import Path
import csv


def parse_kv_pairs_str(kv_pairs_str: str) -> dict[str, str]:
    kv_pairs_str_l = kv_pairs_str.split("-")
    kv_pairs = {}
    for kv_pair_str in kv_pairs_str_l:
        k, _, v = kv_pair_str.partition("=")
        kv_pairs[k] = v
    return kv_pairs


def parse_benchmark_id(bid: str) -> dict:
    kv = {}
    if bid:
        for token in bid.split("-"):
            token = token.strip()
            if token and "=" in token:
                k, v = token.split("=", 1)
                kv[k.strip()] = v.strip()
    return kv

def normalize_benchmark_id(bid: str) -> frozendict[str, str]:
    return frozendict(parse_benchmark_id(bid))

def get_param_names(bids: tuple[frozendict]) -> tuple[str, ...]:
    """
    Merge a list of tuples into a single tuple of keys that:
    - Preserves the relative order of keys within each tuple
    - Gives precedence to later tuples when conflicts arise
    """
    merged = list(bids[-1])
    merged_set = set(merged)
    for tup in bids[:-1]:
        for key in tup:
            if key not in merged_set:
                merged.append(key)
                merged_set.add(key)
    return tuple(merged)


def build_sort_key(params_name: Iterable[str]) -> Callable:
    def sort_key(d):
        nonlocal params_name
        key_list = []
        for col in params_name:
            if col in d:
                val = d[col]
                key_list.append((0, val))
            else:
                key_list.append((1, None))
        return key_list
    return sort_key


def artifacts_parse_csv_summary(current_txt_path: Path, metric_keys: Iterable[str]):
    out = {}
    for line in current_txt_path.read_text().splitlines():
        kv = dict(map(str.strip, p.split("=", 1)) for p in line.split("|") if "=" in p)
        record = {}
        for k in metric_keys:
            try:
                record[k] = float(kv.pop(k))
            except (ValueError, TypeError, KeyError):
                pass
        nbid = frozendict(kv)
        out[nbid] = record
    return out

def fmt_num(v, is_int: bool):
    return f"{int(v):,}" if is_int else f"{v:.2f}"


class CsvInfo:
    def __init__(self, artifacts_dir: Path, metric_keys: Iterable[str], filename_glob: str) -> None:
        self.current_csv_paths = list(artifacts_dir.rglob(filename_glob))
        assert self.current_csv_paths

        self.metric_keys = metric_keys

        self.current_bm = {}
        for self.csv_path in self.current_csv_paths:
            self.current_bm |= artifacts_parse_csv_summary(self.csv_path, self.metric_keys)
        self.bids_set = frozenset(self.current_bm.keys())
        assert self.bids_set

        self.params_name = get_param_names(tuple((tuple(kv.keys())) for kv in self.current_bm.keys()))

    def ingest_records_by_rev(self, records_by_rev):
        self.blist = [f"- Commit {i}: {sha}" for i, sha in enumerate(records_by_rev.keys(), 1)]
        self.baseline_block = ["**Baselines considered:** " + f"**{len(self.records_by_rev)}** commits"] + blist


    def get_params_name(self):
        return get_param_names(tuple((tuple(kv.keys())) for kv in self.current_bm.keys()))

def build_table(params_name: str, alias: str, csv_info: CsvInfo, records_by_rev) -> None:
    rows_md = []
    rows = []

    header_cells = (
        "status",
        *params_name,
        f"current {alias}",
        f"baseline {alias} [last (mean ± std)] (*1)",
        f"Δ {alias} (*2)"
    )
    header = "| " + " | ".join(header_cells) + " |"
    align  = "|:------:|" + "|".join([":---" for _ in params_name]) + "|---:|---:|---:|"

    for bid in sorted(csv_info.current_bm.keys(), key=sort_key):
        value_cur = csv_info.current_bm[bid][metric]
        is_int = isinstance(value_cur, int) or value_cur.is_integer()
        value_repr = fmt_num(value_cur, is_int)

        params_repr = [bid.get(k, "-") for k in params_name]
        info = {
            **dict(zip(params_name, params_repr)),
            "current": value_cur,
            "baseline_last": None,
            "baseline_mean": None,
            "baseline_min": None,
            "baseline_max": None,
            "status": None,
        }

        values_prev = [
            record[bid][metric]
            for record in records_by_rev.values()
            if bid in record
        ]
        if values_prev:
            value_last = values_prev[0]
            value_ref = statistics.fmean(values_prev)
            delta = (value_cur - value_last) / value_last * 100.0

            info["baseline_last"] = int(value_last) if is_int else float(value_last)

            stats_repr = f"{fmt_num(value_last, is_int)}"
            delta_repr = f"{delta:+.1f}%"
            if len(values_prev) == MAX_VALID_REVISIONS:
                info["baseline_mean"] = int(value_ref) if is_int else float(value_ref)
                info["baseline_min"] = int(min(values_prev)) if is_int else float(min(values_prev))
                info["baseline_max"] = int(max(values_prev)) if is_int else float(max(values_prev))

                value_std = statistics.stdev(values_prev)
                stats_repr += f" ({fmt_num(value_ref, is_int)} ± {fmt_num(value_std, is_int)})"
                if sign * delta < - METRICS_TOL[metric]:
                    info["status"] = "regression"

                    delta_repr = f"**{delta_repr}**"
                    picto = "🔴"
                    reg_found = True
                elif sign * delta > METRICS_TOL[metric]:
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

        rows_md.append("| " + " | ".join((picto, *params_repr, value_repr, stats_repr, delta_repr)) + " |")
        rows.append(info)

    return [header, align] + rows_md, rows
    # tables[metric] = [header, align] + rows_md


def main() -> None:
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-artifacts-dir", type=str, required=True)
    parser.add_argument("--mem-artifacts-dir", type=str, required=True)
    parser.add_argument("--max-valid-revisions", type=int, default=10)
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

    MAX_VALID_REVISIONS = args.max_valid_revisions
    MAX_FETCH_REVISIONS = args.max_fetch_revisions

    METRICS_TOL = {
        "runtime_fps": args.runtime_fps_regression_tolerance_pct,
        "compile_time": args.compile_time_regression_tolerance_pct,
        "max_mem_mb": args.mem_regression_tolerance_pct,
    }

    speed_artifacts_dir = Path(args.speed_artifacts_dir).expanduser().resolve()
    mem_artifacts_dir = Path(args.mem_artifacts_dir).expanduser().resolve()
    check_body_path = Path(args.check_body_path).expanduser()

    speed_csv_files = {
        "runtime_fps": Path(args.csv_runtime_fps_path).expanduser().resolve(),
        "compile_time": Path(args.csv_compile_time_path).expanduser().resolve(),
    }
    mem_csv_files = {
        "mem": Path(args.csv_mem_path).expanduser().resolve(),
    }

    SPEED_METRIC_KEYS = ("compile_time", "runtime_fps", "realtime_factor")
    MEM_METRIC_KEYS = ("max_mem_mb")

    csv_info_speed = CsvInfo(artifacts_dir=speed_artifacts_dir, metric_keys=SPEED_METRIC_KEYS, filename_glob="speed_test*.txt")
    csv_info_mem = CsvInfo(artifacts_dir=mem_artifacts_dir, metric_keys=MEM_METRIC_KEYS, filename_glob="mem_test*.txt")

    # ----- W&B baselines -----

    assert "WANDB_API_KEY" in os.environ

    ENTITY = os.environ["WANDB_ENTITY"]
    PROJECT_OLD = os.environ["WANDB_PROJECT_OLD_FORMAT"]
    PROJECT_NEW = os.environ["WANDB_PROJECT_NEW_FORMAT"]

    def fetch_wandb_data_old_format(csv_info: CsvInfo):
        print("fetch_wandb_data_old_format")
        api = wandb.Api()
        runs_iter = api.runs(f"{ENTITY}/{PROJECT_OLD}", order="-created_at")
        print('got runs_iter')

        revs = set()
        records_by_rev = {}
        for i, run in enumerate(runs_iter):
            print("i", i, "run", run)
            # Abort if still not complete after checking enough runs.
            # This would happen if a new benchmark has been added, and not enough past data is available yet.
            if len(revs) == MAX_FETCH_REVISIONS:
                break

            # Early return if enough complete records have been collected
            records_is_complete = [csv_info.bids_set.issubset(record.keys()) for record in records_by_rev.values()]
            if sum(records_is_complete) == MAX_VALID_REVISIONS:
                break

            # Load config and summary, with support of legacy runs
            config, summary = run.config, run.summary
            if isinstance(config, str):
                config = {k: v["value"] for k, v in json.loads(run.config).items() if not k.startswith("_")}
            if isinstance(summary._json_dict, str):
                summary = json.loads(summary._json_dict)

            # Extract revision commit and branch
            try:
                rev, branch = config["revision"].split("@", 1)
                revs.add(rev)
            except ValueError:
                # Ignore this run if the revision has been corrupted for some unknown reason
                continue
            # Ignore runs associated with a commit that is not part of the official repository
            if not branch.startswith('Genesis-Embodied-AI/') and not args.dev_allow_all_branches:
                continue

            # Skip runs did not finish for some reason
            if run.state != "finished":
                continue

            # Do not store new records if the desired number of revision is already reached
            if len(records_by_rev) == MAX_VALID_REVISIONS and rev not in records_by_rev:
                continue

            # Extract benchmark ID and normalize it to make sure it does not depends on key ordering.
            # Note that the rigid body benchmark suite is the only one being supported for now.
            sid, bid = config["benchmark_id"].split("-", 1)
            if sid != "rigid_body":
                continue

            # Make sure that stats are valid
            try:
                is_valid = True
                for k in SPEED_METRIC_KEYS:
                    v = summary[k]
                    if not isinstance(v, (float, int)) or math.isnan(v):
                        is_valid = False
                        break
                if not is_valid:
                    continue
            except KeyError:
                continue

            # Store all the records into a dict
            nbid = normalize_benchmark_id(bid)
            records_by_rev.setdefault(rev, {})[nbid] = {
                metric: summary[metric] for metric in SPEED_METRIC_KEYS
            }
            return records_by_rev

    def fetch_wandb_data_new_format(csv_info: CsvInfo):
        print("fetch_wandb_data_new_format")
        api = wandb.Api()
        runs_iter = api.runs(f"{ENTITY}/{PROJECT_NEW}", order="-created_at")
        print('got runs_iter')

        revs = set()
        records_by_rev = defaultdict(lambda: defaultdict(dict))
        for i, run in enumerate(runs_iter):
            print("i", i, "run", run)
            # Abort if still not complete after checking enough runs.
            # This would happen if a new benchmark has been added, and not enough past data is available yet.
            if len(revs) == MAX_FETCH_REVISIONS:
                break

            # Early return if enough complete records have been collected
            records_is_complete = [csv_info.bids_set.issubset(record.keys()) for record in records_by_rev.values()]
            if sum(records_is_complete) == MAX_VALID_REVISIONS:
                break

            # Load config and summary, with support of legacy runs
            config, summary = run.config, run.summary
            if isinstance(config, str):
                config = {k: v["value"] for k, v in json.loads(run.config).items() if not k.startswith("_")}
            if isinstance(summary._json_dict, str):
                summary = json.loads(summary._json_dict)

            # Extract revision commit and branch
            try:
                rev, branch = config["revision"].split("@", 1)
                revs.add(rev)
            except ValueError:
                print('didnt find rev')
                # Ignore this run if the revision has been corrupted for some unknown reason
                continue
            print("rev", rev, "branch", branch)
            # Ignore runs associated with a commit that is not part of the official repository
            if not branch.startswith('Genesis-Embodied-AI/') and not args.dev_allow_all_branches:
                print('branch didnt start with Genesis-Embodied-AI')
                continue

            # Skip runs did not finish for some reason
            if run.state != "finished":
                continue

            # Do not store new records if the desired number of revision is already reached
            if len(records_by_rev) == MAX_VALID_REVISIONS and rev not in records_by_rev:
                continue

            for k, v in summary.items():
                if k.startswith("_"):
                    continue
                metric_name, _, kv_pairs_str = k.partition("-")
                records_by_rev[rev][kv_pairs_str][metric_name] = v                

            print('records_by_rev', records_by_rev)
            for rev, records in records_by_rev.items():
                print('rev')
                for k, v in records.items():
                    print("- ", "record", k, v)
            # adsfasdf
            return records_by_rev

    speed_records_by_rev = {}
    if not args.dev_skip_speed:
        speed_records_by_rev = fetch_wandb_data_old_format(csv_info=csv_info_speed)
    print('speed_records_by_rev', speed_records_by_rev)

    mem_records_by_rev = fetch_wandb_data_new_format(csv_info=csv_info_mem)

    # ----- build TWO tables -----

    # Parse benchmark IDs into key-value dicts while preserving order

    reg_found, alert_found = False, False
    tables = {}
    rows_for_csv = {"runtime_fps": [], "compile_time": [], "mem": []}
    info = {}
    for metric, alias, sign in (("runtime_fps", "FPS", 1), ("compile_time", "compile", -1)):
        tables[metric], rows_for_csv[metric] = build_table(
            params_name=csv_info.get_params_name())

    # ----- baseline commit list (MD) -----

    # ----- CHECK body (always) -----

    thr_repr = ", ".join(
        f"{alias} ± {METRICS_TOL[metric]:.0f}%"
        for metric, alias in (("runtime_fps", "runtime"), ("compile_time", "compile"))
    )

    check_body = "\n".join(
        [
            *baseline_block,
            "",
            f"Thresholds: {thr_repr}",
            "",
            "### Runtime FPS",
            *tables["runtime_fps"],
            "",
            "### Compile Time",
            *tables["compile_time"],
            "",
            f"- (*1) last: last commit on main, mean/std: stats over revs {MAX_VALID_REVISIONS} commits if available.",
            f"- (*2) Δ: relative difference between PR and last commit on main, i.e. (PR - main) / main * 100%.",
        ]
    )

    # ----- COMMENT body (only if regressions) -----

    if reg_found:
        comment_body = "\n".join([":warning: **Benchmark Regression Detected**", *check_body])
    else:
        comment_body = ""

    for metric in ("runtime_fps", "compile_time"):
        with csv_files[metric].open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=info.keys())
            w.writeheader()
            for rec in rows_for_csv[metric]:
                w.writerow(rec)

    # write md results
    check_body_path.write_text(check_body + "\n", encoding="utf-8")

    if reg_found:
        exit_code = int(os.environ["EXIT_CODE_REGRESSION"])
    elif alert_found:
        exit_code = int(os.environ["EXIT_CODE_ALERT"])
    else:
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
