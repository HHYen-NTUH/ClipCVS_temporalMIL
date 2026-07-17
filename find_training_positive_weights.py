"""Display positive weights of CVS scores for class-balanced training."""

import pathlib
import pandas as pd
from parser import ntucholecvs_parser
from _analyze_data_distribution import (
    read_ntucholecvs,
    SPLITCOL,
    TRAINGRP,
    C1COL,
    C2COL,
    C3COL,
    ICGCOL,
)


if __name__ == "__main__":
    parser = ntucholecvs_parser()
    args = parser.parse_args()
    metadata = read_ntucholecvs(
        pathlib.Path(args.metadatadir), pathlib.Path(args.videometapath)
    )
    # All training set
    _metadata = metadata.query(f"{SPLITCOL} == @TRAINGRP")
    _count = pd.DataFrame(
        {
            "C1": _metadata[C1COL].value_counts(),
            "C2": _metadata[C2COL].value_counts(),
            "C3": _metadata[C3COL].value_counts(),
        }
    )
    print("------------")
    print("Training set")
    print("------------")
    print("Class distribution:")
    print(_count)
    print("Positive weights:")
    print((_count.loc[0, :] / _count.loc[1, :]).round(4))
    # Training set without ICG
    _metadata = metadata.query(f"{SPLITCOL} == @TRAINGRP & {ICGCOL} == False")
    _count = pd.DataFrame(
        {
            "C1": _metadata[C1COL].value_counts(),
            "C2": _metadata[C2COL].value_counts(),
            "C3": _metadata[C3COL].value_counts(),
        }
    )
    print("------------------------")
    print("Training set with no ICG")
    print("------------------------")
    print("Class distribution:")
    print(_count)
    print("Positive weights:")
    print((_count.loc[0, :] / _count.loc[1, :]).round(4))
