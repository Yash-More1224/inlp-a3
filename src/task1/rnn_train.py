from __future__ import annotations

from src.task1.core import run_task1


def main(config_path: str, mode: str) -> None:
    run_task1(config_path=config_path, mode=mode, cell_type="rnn")
