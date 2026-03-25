from __future__ import annotations

from src.task2.core import run_task2


def main(config_path: str, mode: str) -> None:
    run_task2(config_path=config_path, mode=mode, model_type="ssm")
