"""Utilities of model training callbacks."""

from collections.abc import Callable
from typing import Any
import pathlib
import numpy as np
import torch
import mlflow


class CheckpointCallback:
    """Callback of model checkpointing."""

    def __init__(self, ckptdir: pathlib.Path, ckptname: str) -> None:
        self.ckptdir = ckptdir
        self.ckptname = ckptname  # could be a factory method in the future
        self.best_metric = -np.inf
        self._best_ckptname = None

    @property
    def best_ckptpath(self) -> pathlib.Path:
        """Return filepath to the best checkpoint."""
        if self._best_ckptname is None:
            raise RuntimeError("No checkpoint has been saved")
        else:
            return self.ckptdir / self._best_ckptname

    def save(self, model: torch.nn.Module) -> None:
        """Save the model."""
        self._best_ckptname = self.ckptname
        torch.save(model.state_dict(), self.best_ckptpath)

    def update_save(
        self, metric: float, model: torch.nn.Module, negate_metric: bool = False
    ) -> None:
        """Update best metric and save model if it has a larger metric."""
        _metric = -metric if negate_metric else metric
        if _metric > self.best_metric:
            self.best_metric = _metric
            self.save(model)


class MlflowLogger:
    """Callback of MLflow logger."""

    def __init__(
        self, server_uri: str, expname: str, runname: str, rundesc: str | None = None
    ) -> None:
        self.server_uri = server_uri
        self.expname = expname
        self.runname = runname
        self.rundesc = rundesc

    def start_run(self) -> mlflow.ActiveRun:
        """Return context manager of the newly started active run."""
        mlflow.set_tracking_uri(self.server_uri)
        if mlflow.get_experiment_by_name(self.expname) is None:
            mlflow.create_experiment(self.expname)
        mlflow.set_experiment(experiment_name=self.expname)
        return mlflow.start_run(run_name=self.runname, description=self.rundesc)

    def log_metrics_callback(
        self, start_step: int
    ) -> Callable[[dict[str, Any], int], None]:
        """Return callback function that logs the metrics."""

        def logcb(metrics: dict[str, Any], step: int) -> None:
            mlflow.log_metrics(metrics=metrics, step=start_step + step)

        return logcb


class BatchLossAveraging:
    """Averaging accumulator of loss values of batches."""

    def __init__(self) -> None:
        self.numerator = 0
        self.denominator = 0

    def reset(self) -> None:
        self.numerator = 0
        self.denominator = 0

    def update(self, loss: float, weight: float) -> None:
        self.numerator += loss * weight
        self.denominator += weight

    def compute(self) -> float:
        return self.numerator / self.denominator if self.denominator != 0 else 0


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
