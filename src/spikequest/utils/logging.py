import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO


class ExperimentLogger:
    """Simple CSV logger for experiment results.

    Logs per-episode metrics and optionally prints to stdout.

    Args:
        log_dir: directory for log files (created if missing)
        experiment_name: name prefix for log file
        stdout: if True, also print each row to stdout
    """

    def __init__(
        self,
        log_dir: str = "experiments/outputs",
        experiment_name: str = "run",
        stdout: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.stdout = stdout
        self._file: Optional[TextIO] = None
        self._writer = None
        self._headers_written = False

    @property
    def path(self) -> Path:
        return self.log_path

    def log(self, row: dict):
        if not self._headers_written:
            self._file = open(self.log_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._headers_written = True
        self._writer.writerow(row)
        self._file.flush()
        if self.stdout:
            print(row)

    def close(self):
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()