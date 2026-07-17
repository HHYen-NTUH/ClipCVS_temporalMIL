"""Analyze data distribution of the Endoscapes dataset and the NTUcholeCVS dataset."""

import pathlib
import pandas as pd


VIDEOCOL = "video"
SGMTCOL = "segment"
C1COL = "two_structures"
C2COL = "hepatocystic_triangle"
C3COL = "cystic_plate"

SPLITCOL = "split"
TRAINGRP = "training"
VALGRP = "validation"
TESTGRP = "testing"

ICGCOL = "is_icg"


def read_endoscapes(datadir: pathlib.Path) -> pd.DataFrame:
    metadata = (
        pd.read_csv(datadir / "all_metadata.csv")
        .query("cvs_annotator_1.notnull()")
        .loc[:, ["vid", "frame", "C1", "C2", "C3"]]
        .rename(
            columns={
                "vid": VIDEOCOL,
                "frame": SGMTCOL,
                "C1": C1COL,
                "C2": C2COL,
                "C3": C3COL,
            }
        )
    )
    metadata[[C1COL, C2COL, C3COL]] = (
        metadata[[C1COL, C2COL, C3COL]].round().astype(int)
    )

    metadata[ICGCOL] = False
    split_metadata = pd.concat(
        [
            pd.read_csv(datadir / "train_vids.txt", header=None, names=[VIDEOCOL])
            .astype(int)
            .assign(**{SPLITCOL: TRAINGRP}),
            pd.read_csv(datadir / "val_vids.txt", header=None, names=[VIDEOCOL])
            .astype(int)
            .assign(**{SPLITCOL: VALGRP}),
            pd.read_csv(datadir / "test_vids.txt", header=None, names=[VIDEOCOL])
            .astype(int)
            .assign(**{SPLITCOL: TESTGRP}),
        ],
        axis=0,
    )
    metadata = metadata.merge(split_metadata, on=VIDEOCOL)
    return metadata


def read_ntucholecvs(
    metadatadir: pathlib.Path, videometapath: pathlib.Path
) -> pd.DataFrame:
    video_metadata = (
        pd.read_csv(videometapath)
        .loc[:, ["video", "ICG", "Group_20250619"]]
        .replace(
            {
                "Group_20250619": {
                    "Train": TRAINGRP,
                    "Validation": VALGRP,
                    "Test": TESTGRP,
                }
            }
        )
        .rename(columns={"video": VIDEOCOL, "Group_20250619": SPLITCOL, "ICG": ICGCOL})
    )
    video_metadata[ICGCOL] = video_metadata[ICGCOL].astype(bool)
    frame_metadata = pd.concat(
        [
            pd.read_csv(metadatadir / f"{video}.csv")
            .loc[
                :,
                [
                    "tiff_name",
                    "two_structures",
                    "hepatocystic_triangle",
                    "cystic_plate",
                ],
            ]
            .rename(
                columns={
                    "tiff_name": SGMTCOL,
                    "two_structures": C1COL,
                    "hepatocystic_triangle": C2COL,
                    "cystic_plate": C3COL,
                }
            )
            .assign(**{VIDEOCOL: video})
            for video in video_metadata[VIDEOCOL]
        ],
        axis=0,
    )
    metadata = frame_metadata.merge(video_metadata, on=VIDEOCOL)
    return metadata


def filter_no_icg(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata.query(f"{ICGCOL} == False")


def filter_positive(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata.query(f"{C1COL} == 1 | {C2COL} == 1 | {C3COL} == 1")


def _display_class_distribution(data: pd.DataFrame) -> None:
    multilabel = pd.Series(
        list(data[[C1COL, C2COL, C3COL]].itertuples(index=False, name="CVS")),
    )
    distribution = multilabel.value_counts().sort_index().to_frame()
    distribution["proportion"] = (distribution["count"] / len(multilabel)).round(4)
    print(distribution)


def main(
    ntucholecvs_metadatadir: pathlib.Path,
    ntucholecvs_videometapath: pathlib.Path,
    endoscapes_datadir: pathlib.Path,
) -> None:
    """
    Read the metadata of Endoscapes and NTUcholeCVS datasets and display their class
    distribution respectively.
    """
    endoscapes_metadata = read_endoscapes(endoscapes_datadir)
    ntucholecvs_metadata = read_ntucholecvs(
        metadatadir=ntucholecvs_metadatadir, videometapath=ntucholecvs_videometapath
    )
    print("CVS distribution of the Endoscapes dataset")
    _display_class_distribution(endoscapes_metadata)
    print("CVS distribution of the NTUcholeCVS dataset")
    _display_class_distribution(ntucholecvs_metadata)
    print("CVS distribution of the NTUcholeCVS dataset (training)")
    _display_class_distribution(ntucholecvs_metadata.query(f"{SPLITCOL} == @TRAINGRP"))
    print("CVS distribution of the NTUcholeCVS dataset (validation)")
    _display_class_distribution(ntucholecvs_metadata.query(f"{SPLITCOL} == @VALGRP"))
    print("CVS distribution of the NTUcholeCVS dataset (testing)")
    _display_class_distribution(ntucholecvs_metadata.query(f"{SPLITCOL} == @TESTGRP"))
    ntucholecvs_metadata_no_icg = filter_no_icg(ntucholecvs_metadata)
    print("CVS distribution of the NTUcholeCVS dataset (no ICG)")
    _display_class_distribution(ntucholecvs_metadata_no_icg)
    print("CVS distribution of the NTUcholeCVS dataset (no ICG; training)")
    _display_class_distribution(
        ntucholecvs_metadata_no_icg.query(f"{SPLITCOL} == @TRAINGRP")
    )
    print("CVS distribution of the NTUcholeCVS dataset (no ICG; validation)")
    _display_class_distribution(
        ntucholecvs_metadata_no_icg.query(f"{SPLITCOL} == @VALGRP")
    )
    print("CVS distribution of the NTUcholeCVS dataset (no ICG; testing)")
    _display_class_distribution(
        ntucholecvs_metadata_no_icg.query(f"{SPLITCOL} == @TESTGRP")
    )


if __name__ == "__main__":
    ntucholecvs_metadatadir = pathlib.Path("../../data/NTUcholeCVS/artifact/metadata")
    ntucholecvs_videometapath = pathlib.Path(
        "../../data/NTUcholeCVS/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv"
    )
    endoscapes_datadir = pathlib.Path("../../data/public/endoscapes")
    main(
        ntucholecvs_metadatadir=ntucholecvs_metadatadir,
        ntucholecvs_videometapath=ntucholecvs_videometapath,
        endoscapes_datadir=endoscapes_datadir,
    )
