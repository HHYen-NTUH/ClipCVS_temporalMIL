"""Evaluate CVS prediction."""

import argparse
from collections import namedtuple
import pathlib
import torch
import pandas as pd
import torchmetrics as tm
from data import CvsScoreItem


CLASSES = [item.name for item in CvsScoreItem]


DATACOLS = ["Video", "Frames"]
TRGTCOLSUFFIX = "Trgt"
PREDCOLSUFFIX = "Prob"
TRGTCOLS = [f"{item.name}{TRGTCOLSUFFIX}" for item in CvsScoreItem]
PREDCOLS = [f"{item.name}{PREDCOLSUFFIX}" for item in CvsScoreItem]


def combine_prediction(predpath: list[pathlib.Path]) -> pd.DataFrame:
    assert len(predpath) > 0
    data = pd.read_csv(predpath[0])
    for _predpath in predpath[1:]:
        data = data.merge(pd.read_csv(_predpath), on=DATACOLS)
    assert set(TRGTCOLS + PREDCOLS).issubset(data.columns), (
        "missing columns in prediction "
        f"{list(set(TRGTCOLS + PREDCOLS) - set(data.columns))}"
    )
    return data


def main(
    predpath: list[pathlib.Path],
    thrshs: list[float] | None,
    evaldir: pathlib.Path,
) -> None:
    """Save evaluation results of CVS prediction."""
    # Instantiate evaluation metrics
    metric_kwargs: dict = {
        "task": "multilabel" if len(CLASSES) else "binary",
    }
    confmat_kwargs = metric_kwargs.copy()
    if len(CLASSES) > 1:
        metric_kwargs["num_labels"] = len(CLASSES)
        metric_kwargs["average"] = "none"
        confmat_kwargs["num_labels"] = len(CLASSES)
    thrsh_metrics = tm.MetricCollection(  # metrics requiring thresholding
        {
            "Recall": tm.Recall(**metric_kwargs),
            "Specificity": tm.Specificity(**metric_kwargs),
            "Precision": tm.Precision(**metric_kwargs),
            "F1Score": tm.F1Score(**metric_kwargs),
            "ConfusionMatrix": tm.ConfusionMatrix(**confmat_kwargs),
        }
    )
    nonthrsh_metrics = tm.MetricCollection(  # metrics not requiring thresholding
        {
            "AveragePrecision": tm.AveragePrecision(**metric_kwargs),
            "AUROC": tm.AUROC(**metric_kwargs),
        }
    )
    if thrshs is not None:
        _thrshs = torch.tensor(thrshs)
        assert len(thrshs) == len(CLASSES), (
            f"Classification target consists of {len(CLASSES)} items but "
            f"{len(thrshs)} thresholds are given"
        )
    else:
        _thrshs = torch.ones(len(CLASSES)) * 0.5
    # Compute evaluation results
    data = combine_prediction(predpath)
    trgts = torch.tensor(data.loc[:, TRGTCOLS].to_numpy())
    probs = torch.tensor(data.loc[:, PREDCOLS].to_numpy())
    thrsh_metrics.update((probs >= _thrshs).float(), trgts)
    nonthrsh_metrics.update(probs, trgts)
    result = thrsh_metrics.compute() | nonthrsh_metrics.compute()
    EvaluationRow = namedtuple(
        "EvaluationRow",
        [
            "Recall",
            "Specificity",
            "Precision",
            "BalancedAccuracy",
            "F1Score",
            "AveragePrecision",
            "AUROC",
        ],
    )
    evaluation = pd.DataFrame(
        (
            [
                EvaluationRow(
                    result["Recall"][i].item(),
                    result["Specificity"][i].item(),
                    result["Precision"][i].item(),
                    (result["Recall"][i].item() + result["Specificity"][i].item()) / 2,
                    result["F1Score"][i].item(),
                    result["AveragePrecision"][i].item(),
                    result["AUROC"][i].item(),
                )
                for i in range(len(CLASSES))
            ]
            if len(CLASSES) > 1
            else [
                EvaluationRow(
                    result["Recall"].item(),
                    result["Specificity"].item(),
                    result["Precision"].item(),
                    (result["Recall"].item() + result["Specificity"].item()) / 2,
                    result["F1Score"].item(),
                    result["AveragePrecision"].item(),
                    result["AUROC"].item(),
                )
            ]
        ),
        index=CLASSES,
    ).round(decimals=4)
    ConfMatRow = namedtuple(
        "ConfMatRow",
        ["TruePositives", "FalsePositives", "FalseNegatives", "TrueNegatives"],
    )
    confmat = pd.DataFrame(
        (
            [
                ConfMatRow(
                    result["ConfusionMatrix"][i, 1, 1].item(),
                    result["ConfusionMatrix"][i, 0, 1].item(),
                    result["ConfusionMatrix"][i, 1, 0].item(),
                    result["ConfusionMatrix"][i, 0, 0].item(),
                )
                for i in range(len(CLASSES))
            ]
            if len(CLASSES) > 1
            else [
                ConfMatRow(
                    result["ConfusionMatrix"][1, 1].item(),
                    result["ConfusionMatrix"][0, 1].item(),
                    result["ConfusionMatrix"][1, 0].item(),
                    result["ConfusionMatrix"][0, 0].item(),
                )
            ]
        ),
        index=CLASSES,
    )
    # Add a row for the macro-averaging metrics
    evaluation = pd.concat(
        [evaluation, evaluation.mean().to_frame(name="macro").T.round(decimals=4)],
        axis=0,
    )
    # Display and save evaluation results
    evaluation.to_csv(evaldir / "metrics.csv")
    confmat.to_csv(evaldir / "confusion_matrix.csv")
    # Display and save evaluation results
    print(evaluation)
    print(confmat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predpath",
        type=str,
        required=True,
        nargs="+",
        help="File path of CVS prediction",
    )
    parser.add_argument(
        "--thrshs",
        type=float,
        default=None,
        nargs="+",
        help="Classification thresholds of the CVS score items respectively",
    )
    parser.add_argument(
        "--evaldir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )
    args = parser.parse_args()
    pathlib.Path(args.evaldir).mkdir(parents=True, exist_ok=True)
    main(
        predpath=[pathlib.Path(predpath) for predpath in args.predpath],
        thrshs=args.thrshs,
        evaldir=pathlib.Path(args.evaldir),
    )
