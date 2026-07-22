import os
from pathlib import Path
from typing import Any

import pytest

import genesis as gs

from ..utils import pprint_oneline


@pytest.fixture(scope="session")
def stream_writers(printer_session, request):
    report_path = Path(request.config.getoption("--speed-test-filepath"))

    # Delete old unrelated worker-specific reports
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id == "gw0":
        worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])

        for path in report_path.parent.glob("-".join((report_path.stem, "*.txt"))):
            _, worker_id_ = path.stem.rsplit("-", 1)
            worker_num = int(worker_id_[2:])
            if worker_num >= worker_count:
                path.unlink()

    # Create new empty worker-specific report
    report_name = "-".join(filter(None, (report_path.stem, worker_id)))
    report_path = report_path.with_name(f"{report_name}.txt")
    if report_path.exists():
        report_path.unlink()
    fd = open(report_path, "w")

    yield (lambda msg: print(msg, file=fd, flush=True), printer_session)

    fd.close()


@pytest.fixture(scope="function")
def factory_logger(stream_writers):
    class Logger:
        def __init__(self, hparams: dict[str, Any]):
            self.hparams = {
                **hparams,
                "dtype": "ndarray" if gs.use_ndarray else "field",
                "backend": str(gs.backend.name),
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def write(self, items):
            nonlocal stream_writers

            if stream_writers:
                msg = (
                    pprint_oneline(self.hparams, delimiter=" \t| ")
                    + " \t| "
                    + pprint_oneline(items, delimiter=" \t| ", digits=1)
                )
                for writer in stream_writers:
                    writer(msg)

    return Logger
