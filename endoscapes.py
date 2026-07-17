"""Utilities to load the Endoscapes dataset."""

from collections import namedtuple
from collections.abc import Callable
from typing import Literal
import pathlib
from glob import glob
import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as tvtrnsfrm
from tqdm import tqdm
from swinv2 import Swinv2SpatialCvsModel
from _analyze_data_distribution import (
    VIDEOCOL,
    SGMTCOL,
    C1COL,
    C2COL,
    C3COL,
    SPLITCOL,
    TRAINGRP,
    VALGRP,
    TESTGRP,
)
from data import CvsScoreItem, cvs_classification_target, cvs_multiclass_target
from augmentation import AugmentFlag, zoom_augment, color_jitter_augment


def _process_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    _metadata = metadata.rename(
        columns={
            VIDEOCOL: "vid",
            SGMTCOL: "frame",
            C1COL: "C1",
            C2COL: "C2",
            C3COL: "C3",
        }
    )
    return _metadata


# NOTE: Adapted from source
def generate_path(row, image_folder: str) -> str:
    vid = int(row["vid"])
    frame = int(row["frame"])
    filepath = str(pathlib.Path(image_folder) / f"{vid}_{frame}.jpg")
    return filepath


def _framepaths_of_videos(
    metadata: pd.DataFrame, datadir: pathlib.Path
) -> list[pathlib.Path]:
    return [
        pathlib.Path(framepath)
        for vid in metadata["vid"].unique()
        for framepath in glob(str(datadir / f"{vid}_*.jpg"))
    ]


UnlabelledRow = namedtuple("UnlabelledRow", ["vid", "frame", "C1", "C2", "C3"])


# NOTE: Adapted from source
def add_unlabelled_imgs(
    list_of_selected_images: list[pathlib.Path], selected_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Given existing splits dataframes, in correct otder, append images that are unlabelled. Output dataframe has columns:
    idx | vid | frame | C1 | C2 | C3
    where C1-3 is unlabelled it has a value '-1.0'
    """
    rows = []
    for image in list_of_selected_images:
        vid, frame = map(int, image.stem.split("_"))
        rows.append(UnlabelledRow(vid=vid, frame=frame, C1=-1, C2=-1, C3=-1))

    df = pd.DataFrame(rows)

    combined_df = pd.merge(
        df,
        selected_dataframe,
        on=["vid", "frame"],
        how="left",
        suffixes=("_new", "_lbld"),
    )
    combined_df["C1"] = combined_df["C1_lbld"].combine_first(combined_df["C1_new"])
    combined_df["C2"] = combined_df["C2_lbld"].combine_first(combined_df["C2_new"])
    combined_df["C3"] = combined_df["C3_lbld"].combine_first(combined_df["C3_new"])

    # Drop the redundant columns from df1
    final_df = combined_df[["vid", "frame", "C1", "C2", "C3"]]

    final_df = final_df.sort_values(by=["vid", "frame"])
    final_df = final_df.reset_index(drop=True)

    return final_df


# NOTE: Adapted from source
def update_dataframe(
    dataframe: pd.DataFrame,
    image_folder: str,
    cvs_score_items: CvsScoreItem,
    multiclass: bool = False,
):
    """
    Function only for creation of dataframes when training backbone - SwinV2. It changes the structure of the dataframe from:
    idx | vid | frame | C1 | C2 | C3
    to:
    idx | path | classification
    where path is a path to a given image and classification is a list of ground truth values for C1-3 as, [C1, C2, C3] e.g. [0.0, 0.0, 1.0]
    """
    dataframe["path"] = dataframe.apply(
        lambda row: generate_path(row, image_folder), axis=1
    )
    dataframe["classification"] = dataframe.apply(
        lambda row: (
            cvs_multiclass_target(row, items=cvs_score_items)
            if multiclass
            else cvs_classification_target(row, items=cvs_score_items)
        ),
        axis=1,
    )
    dataframe.drop(columns=["vid", "frame", "C1", "C2", "C3"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def endoscapes_transform(
    image_size: int = 384,
    augment_flags: AugmentFlag = AugmentFlag.NONE,
) -> Callable[[torch.Tensor], torch.Tensor]:
    transforms: list = [
        tvtrnsfrm.Resize(480),  # resize to 480p
    ]
    if AugmentFlag.ZOOM in augment_flags:
        transforms.append(zoom_augment(base_size=480, ratio_range=(1, 1.2)))
    transforms.extend(
        [tvtrnsfrm.CenterCrop(480), tvtrnsfrm.Resize((image_size, image_size))]
    )
    if AugmentFlag.COLOR in augment_flags:
        transforms.append(
            color_jitter_augment(
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_range=(-0.05, 0.05),
            )
        )
    transforms.extend(
        [
            tvtrnsfrm.ToDtype(torch.float32, scale=True),
            tvtrnsfrm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tvtrnsfrm.Compose(transforms)


class FrameDataset(Dataset):
    """
    Dataset creator only for backbone - SwinV2 training.
    """

    def __init__(
        self,
        image_dataframe: pd.DataFrame,
        transform: Callable[[torch.Tensor], torch.Tensor],
        minmaxscale: bool = True,
        multiclass: bool = False,
    ) -> None:
        self.image_dataframe = image_dataframe
        self.transform = transform
        self.minmaxscale = minmaxscale
        self.multiclass = multiclass

    def __len__(self) -> int:
        return len(self.image_dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_info = self.image_dataframe.iloc[idx]
        image_path = image_info["path"]
        image = tvtrnsfrm.functional.to_image(PIL.Image.open(image_path))
        image = self.transform(image)
        if self.minmaxscale:  # wierd choice by the original repo
            image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image))
        label = self.get_label(idx)
        return image, label

    def get_label(self, idx: int) -> torch.Tensor:
        image_info = self.image_dataframe.iloc[idx]
        label = torch.tensor(image_info["classification"])
        if self.multiclass:
            label = label.long()
        return label


def extract_sequence_dataframe(
    dataframe: pd.DataFrame, seqlen: int
) -> pd.api.typing.DataFrameGroupBy:
    # `dataframe` has columns of vid | frame | C1 | C2 | C3
    # For each labelled frame of each video, find prior frames that are less than
    # `seqlen` away
    def _prior_frames_of_labelled_frames(group: pd.DataFrame) -> pd.DataFrame:
        group["pos"] = np.arange(len(group))
        labelled = group.loc[
            (group["C1"] != -1) & (group["C2"] != -1) & (group["C3"] != -1),
            ["frame", "pos"],
        ]
        pairs = group.merge(
            labelled.rename(columns={"frame": "endframe", "pos": "endpos"}), how="cross"
        )
        prior_frames = pairs.loc[
            (pairs["endpos"] >= pairs["pos"])
            & (pairs["endpos"] - pairs["pos"] < seqlen),
            ["endframe", "frame"],
        ]
        return prior_frames

    sequences = (
        dataframe.groupby("vid")
        .apply(_prior_frames_of_labelled_frames, include_groups=False)
        .droplevel(level=None, axis=0)
        .reset_index()
        .sort_values(["vid", "endframe", "frame"])
    )
    # Exclude incomplete sequences
    sequences = sequences.groupby(["vid", "endframe"]).filter(
        lambda grp: len(grp) == seqlen
    )
    # Merge `dataframe` with pairs of seqnence-end frames and prior frames
    sequences = sequences.merge(dataframe, on=["vid", "frame"], how="left")
    # Return the grouped dataframe by video and sequence-end frames
    return sequences.groupby(["vid", "endframe"])


class FlexibleFrameSequenceDataset(Dataset):
    """Dataset of frame sequences with flexible configuration."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        seqlen: int,
        image_folder: str,
        cvs_score_items: CvsScoreItem,
        transform: Callable[[torch.Tensor], torch.Tensor],
        minmaxscale: bool = True,
        label_method: Literal["last", "union", "more_than_50%"] = "last",
    ) -> None:
        super().__init__()
        self.sequences = extract_sequence_dataframe(dataframe, seqlen)
        self.groups = list(self.sequences.groups.keys())
        self.image_folder = image_folder
        self.cvs_score_items = cvs_score_items
        self.transform = transform
        self.minmaxscale = minmaxscale
        self.label_method = label_method

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, ind: int) -> tuple[torch.Tensor, torch.Tensor]:
        df = update_dataframe(
            self.sequences.get_group(self.groups[ind]).copy(),
            self.image_folder,
            self.cvs_score_items,
        )
        frames = []
        for framepath in df["path"]:
            frame = tvtrnsfrm.functional.to_image(PIL.Image.open(framepath))
            frames.append(frame)
        frames = torch.stack(frames)
        frames = self.transform(frames)
        if self.minmaxscale:  # wierd choice by the original repo
            frames_min = frames.flatten(start_dim=1).amin(dim=1).view(-1, 1, 1, 1)
            frames_max = frames.flatten(start_dim=1).amax(dim=1).view(-1, 1, 1, 1)
            frames = (frames - frames_min) / (frames_max - frames_min)
        label = self.get_label(ind)
        return frames, label

    def get_label(self, ind: int) -> torch.Tensor:
        df = update_dataframe(
            self.sequences.get_group(self.groups[ind]).copy(),
            self.image_folder,
            self.cvs_score_items,
        )
        if self.label_method == "union":
            _labels = np.array(df["classification"].tolist())
            _labels = _labels[((_labels == 0) | (_labels == 1)).all(axis=1), :]
            label = torch.tensor(_labels.astype(bool).any(axis=0).astype(float))
        elif self.label_method == "more_than_50%":
            _labels = np.array(df["classification"].tolist())
            _labels = _labels[((_labels == 0) | (_labels == 1)).all(axis=1), :]
            label = torch.tensor((_labels.mean(axis=0) >= 0.5).astype(float))
        else:  # use label of the last frame
            label = torch.tensor(df["classification"].iloc[-1])
        return label


SPLITDIRNAME = {TRAINGRP: "train", VALGRP: "val", TESTGRP: "test"}


def trainvaltest_datasets(
    metadata: pd.DataFrame,
    datadir: pathlib.Path,
    cvs_score_items: CvsScoreItem = CvsScoreItem.C1 | CvsScoreItem.C2 | CvsScoreItem.C3,
    multiclass: bool = False,
    image_size: int = 384,
    disable_minmaxscale: bool = False,
    augment: AugmentFlag = AugmentFlag.NONE,
    include_unlabelled_frames: bool = False,
) -> tuple[FrameDataset, FrameDataset, FrameDataset]:
    metadata = _process_metadata(metadata)
    trainmeta = metadata.loc[metadata[SPLITCOL] == TRAINGRP, :].copy()
    valmeta = metadata.loc[metadata[SPLITCOL] == VALGRP, :].copy()
    testmeta = metadata.loc[metadata[SPLITCOL] == TESTGRP, :].copy()
    trainset = FrameDataset(
        update_dataframe(
            (
                trainmeta
                if not include_unlabelled_frames
                else add_unlabelled_imgs(
                    _framepaths_of_videos(trainmeta, datadir / SPLITDIRNAME[TRAINGRP]),
                    trainmeta,
                )
            ),
            str(datadir / SPLITDIRNAME[TRAINGRP]),
            cvs_score_items,
            multiclass=multiclass,
        ),
        endoscapes_transform(image_size=image_size, augment_flags=augment),
        minmaxscale=not disable_minmaxscale,
        multiclass=multiclass,
    )
    valset = FrameDataset(
        update_dataframe(
            (
                valmeta
                if not include_unlabelled_frames
                else add_unlabelled_imgs(
                    _framepaths_of_videos(valmeta, datadir / SPLITDIRNAME[VALGRP]),
                    valmeta,
                )
            ),
            str(datadir / SPLITDIRNAME[VALGRP]),
            cvs_score_items,
            multiclass=multiclass,
        ),
        endoscapes_transform(image_size=image_size, augment_flags=AugmentFlag.NONE),
        minmaxscale=not disable_minmaxscale,
        multiclass=multiclass,
    )
    testset = FrameDataset(
        update_dataframe(
            (
                testmeta
                if not include_unlabelled_frames
                else add_unlabelled_imgs(
                    _framepaths_of_videos(testmeta, datadir / SPLITDIRNAME[TESTGRP]),
                    testmeta,
                )
            ),
            str(datadir / SPLITDIRNAME[TESTGRP]),
            cvs_score_items,
            multiclass=multiclass,
        ),
        endoscapes_transform(image_size=image_size, augment_flags=AugmentFlag.NONE),
        minmaxscale=not disable_minmaxscale,
        multiclass=multiclass,
    )
    return trainset, valset, testset


def trainvaltest_flexible_sequence_datasets(
    metadata: pd.DataFrame,
    datadir: pathlib.Path,
    seqlen: int = 5,
    cvs_score_items: CvsScoreItem = CvsScoreItem.C1 | CvsScoreItem.C2 | CvsScoreItem.C3,
    image_size: int = 384,
    disable_minmaxscale: bool = False,
    augment: AugmentFlag = AugmentFlag.NONE,
    label_method: Literal["last", "union", "more_than_50%"] = "last",
) -> tuple[
    FlexibleFrameSequenceDataset,
    FlexibleFrameSequenceDataset,
    FlexibleFrameSequenceDataset,
]:
    metadata = _process_metadata(metadata)
    trainmeta = metadata.loc[metadata[SPLITCOL] == TRAINGRP, :].copy()
    valmeta = metadata.loc[metadata[SPLITCOL] == VALGRP, :].copy()
    testmeta = metadata.loc[metadata[SPLITCOL] == TESTGRP, :].copy()
    trainset = FlexibleFrameSequenceDataset(
        add_unlabelled_imgs(
            _framepaths_of_videos(trainmeta, datadir / SPLITDIRNAME[TRAINGRP]),
            trainmeta,
        ),
        seqlen,
        str(datadir / SPLITDIRNAME[TRAINGRP]),
        cvs_score_items,
        endoscapes_transform(image_size=image_size, augment_flags=augment),
        minmaxscale=not disable_minmaxscale,
        label_method=label_method,
    )
    valset = FlexibleFrameSequenceDataset(
        add_unlabelled_imgs(
            _framepaths_of_videos(valmeta, datadir / SPLITDIRNAME[VALGRP]), valmeta
        ),
        seqlen,
        str(datadir / SPLITDIRNAME[VALGRP]),
        cvs_score_items,
        endoscapes_transform(image_size=image_size, augment_flags=AugmentFlag.NONE),
        minmaxscale=not disable_minmaxscale,
        label_method=label_method,
    )
    testset = FlexibleFrameSequenceDataset(
        add_unlabelled_imgs(
            _framepaths_of_videos(testmeta, datadir / SPLITDIRNAME[TESTGRP]),
            testmeta,
        ),
        seqlen,
        str(datadir / SPLITDIRNAME[TESTGRP]),
        cvs_score_items,
        endoscapes_transform(image_size=image_size, augment_flags=AugmentFlag.NONE),
        minmaxscale=not disable_minmaxscale,
        label_method=label_method,
    )
    return trainset, valset, testset


class FrameFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of features extracted from frames."""

    def __init__(self, dataset: FrameDataset, featdir: pathlib.Path) -> None:
        super().__init__()
        self.dataset = dataset
        self.featdir = featdir
        self.dirfilenames = FrameFeatureDataset._dirnames_filenames(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dirname, filename = self.dirfilenames[index]
        feats = torch.load(
            self.featdir / dirname / f"{filename}.pt", map_location="cpu"
        )
        return feats, self.dataset.get_label(index)

    @staticmethod
    def _dirnames_filenames(dataset: FrameDataset) -> list[tuple[str, str]]:
        videos: list[str] = []
        frame_inds: list[str] = []
        for ind in range(len(dataset)):
            framepath = pathlib.Path(str(dataset.image_dataframe["path"].iloc[ind]))
            videos.append(framepath.stem.split("_")[0])
            frame_inds.append(framepath.stem.split("_")[-1])
        return list(zip(videos, frame_inds))

    @staticmethod
    def prepare(
        dataset: FrameDataset,
        model: Swinv2SpatialCvsModel,
        device: str,
        num_workers: int,
        featdir: pathlib.Path,
    ) -> None:
        """Extract and save features of frames."""
        loader = DataLoader(
            dataset, batch_size=1, num_workers=num_workers - 1, shuffle=False
        )
        model = model.to(device)
        model.eval()
        with torch.inference_mode():
            for (inputs, _), (dirname, filename) in zip(
                tqdm(loader, desc="Extract"),
                FrameFeatureDataset._dirnames_filenames(dataset),
            ):
                inputs = inputs.to(device)
                feats = model.forward_features(inputs).cpu().squeeze(dim=0)
                savedir = featdir / dirname
                savedir.mkdir(parents=True, exist_ok=True)
                torch.save(feats, savedir / f"{filename}.pt")


class FlexibleFrameSequnceFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of features extracted from video-frame sequences."""

    def __init__(
        self,
        dataset: FlexibleFrameSequenceDataset,
        featdir: pathlib.Path,
        stack_frame_feats: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.featdir = featdir
        self.dirfilenames = FlexibleFrameSequnceFeatureDataset._dirnames_filenames(
            dataset
        )
        self.stack_frame_feats = stack_frame_feats

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dirname, filename = self.dirfilenames[index]
        if self.stack_frame_feats:
            feats = torch.stack(
                [
                    torch.load(
                        self.featdir / dirname / f"{frame_ind}.pt", map_location="cpu"
                    )
                    for frame_ind in filename.split("_")
                ],
                dim=0,
            )
        else:
            feats = torch.load(
                self.featdir / dirname / f"{filename}.pt", map_location="cpu"
            )
        return feats, self.dataset.get_label(index)

    @staticmethod
    def _dirnames_filenames(
        dataset: FlexibleFrameSequenceDataset,
    ) -> list[tuple[str, str]]:
        videos: list[str] = []
        frame_inds: list[list[str]] = []
        for ind in range(len(dataset)):
            df = update_dataframe(
                dataset.sequences.get_group(dataset.groups[ind]).copy(),
                dataset.image_folder,
                dataset.cvs_score_items,
            )
            paths = [pathlib.Path(str(framepath)) for framepath in df["path"]]
            videos.append(paths[-1].stem.split("_")[0])
            frame_inds.append([path.stem.split("_")[-1] for path in paths])
        return list(zip(videos, map(lambda inds: "_".join(inds), frame_inds)))


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
