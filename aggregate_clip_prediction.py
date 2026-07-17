"""Aggregate prediction of short clips into those of long clips."""

import argparse
from typing import Literal
import pathlib
import itertools
import pandas as pd
import scipy as sp


def mean_logit_pooling(probs: pd.Series) -> float:
    return sp.special.expit(sp.special.logit(probs.to_numpy()).mean())


def main(
    srcpredpath: pathlib.Path,
    dstpredpath: pathlib.Path,
    aggmethod: Literal["mean_logit", "max"],
    aggpredpath: pathlib.Path,
) -> None:
    """Save aggregated prediction."""
    srcpred = pd.read_csv(srcpredpath)
    dstpred = pd.read_csv(dstpredpath)
    # Partition destination clips into source clips
    prtnlen = len(srcpred["Frames"].iloc[0].split("_"))
    aggpred = (
        dstpred.groupby(["Video", "Frames"])
        .apply(
            lambda grp: pd.DataFrame(
                {
                    "FramePartition": [
                        "_".join(frames)
                        for frames in itertools.batched(
                            grp["Frames"].iloc[0].split("_"), prtnlen
                        )
                    ],
                }
            ),
            include_groups=True,
        )
        .droplevel(level=2, axis=0)
        .reset_index()
    )
    # Aggregate predicted probabilities
    match aggmethod:
        case "mean_logit":
            aggfn = mean_logit_pooling
        case _:  # max pooling
            aggfn = "max"
    aggpred = (
        aggpred.merge(
            srcpred.loc[:, ["Video", "Frames", "C1Prob", "C2Prob", "C3Prob"]].rename(
                columns={"Frames": "FramePartition"}
            ),
            on=["Video", "FramePartition"],
        )
        .groupby(["Video", "Frames"])
        .agg({"C1Prob": aggfn, "C2Prob": aggfn, "C3Prob": aggfn})
        .reset_index()
    )
    # Save aggregated prediction
    aggpred = aggpred.merge(  # combine with target labels of destination clips
        dstpred.loc[:, ["Video", "Frames", "C1Trgt", "C2Trgt", "C3Trgt"]],
        on=["Video", "Frames"],
    ).loc[
        :,
        ["Video", "Frames", "C1Trgt", "C2Trgt", "C3Trgt", "C1Prob", "C2Prob", "C3Prob"],
    ]
    aggpred.to_csv(aggpredpath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcpredpath", type=str, required=True)
    parser.add_argument("--dstpredpath", type=str, required=True)
    parser.add_argument("--aggmethod", choices=["mean_logit", "max"], required=True)
    parser.add_argument("--aggpredpath", type=str, required=True)
    args = parser.parse_args()
    pathlib.Path(args.aggpredpath).parent.mkdir(parents=True, exist_ok=True)
    main(
        srcpredpath=pathlib.Path(args.srcpredpath),
        dstpredpath=pathlib.Path(args.dstpredpath),
        aggmethod=args.aggmethod,
        aggpredpath=pathlib.Path(args.aggpredpath),
    )
