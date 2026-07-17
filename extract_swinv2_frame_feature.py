"""Extract features of SwinV2 model from video frames."""

import argparse
import pathlib
from _analyze_data_distribution import read_ntucholecvs, filter_no_icg, read_endoscapes
from data import (
    trainvaltest_datasets as ntucholecvs_trainvaltest_datasets,
    FrameFeatureDataset as NtucholecvsFrameFeatureDataset,
    CvsScoreItem,
    cvs_score_items as _cvs_score_items,
)
from endoscapes import (
    trainvaltest_datasets as endoscapes_trainvaltest_datasets,
    FrameFeatureDataset as EndoscapesFrameFeatureDataset,
)
from augmentation import AugmentFlag
from model import load_model_checkpoint
from swinv2 import Swinv2SpatialCvsModel
from lora import LoraTarget, lora_targets as _lora_targets, lora_model_factory
from parser import dataset_parser, cvsmodel_parser, lora_parser, compute_parser


def main(
    datasetname: str,
    datadir: pathlib.Path,
    metadatadir: pathlib.Path | None,
    videometapath: pathlib.Path | None,
    no_icg: bool,
    disable_minmaxscale: bool,
    cvs_score_items: CvsScoreItem,
    swinv2_ckptpath: pathlib.Path | None,
    ckptpath: pathlib.Path | None,
    lora: bool,
    lora_targets: LoraTarget,
    lora_rank: int,
    featdir: pathlib.Path,
    num_workers: int,
    device: str,
) -> None:
    """Extract and save features of sequences of video frames."""
    # Instantiate model
    lora_factory = (
        lora_model_factory(targets=lora_targets, rank=lora_rank) if lora else None
    )
    model = Swinv2SpatialCvsModel(
        num_classes=len(cvs_score_items),
        swinv2_ckptpath=swinv2_ckptpath,
        lora_factory=lora_factory,
    )
    if ckptpath is not None:
        load_model_checkpoint(model, ckptpath)
    # Instantiate data loaders and save the extracted features
    if datasetname == "endoscapes":
        metadata = read_endoscapes(datadir)
        trainset, valset, testset = endoscapes_trainvaltest_datasets(
            metadata,
            datadir,
            image_size=384,
            disable_minmaxscale=disable_minmaxscale,
            augment=AugmentFlag.NONE,  # no augmentation
            cvs_score_items=cvs_score_items,
            include_unlabelled_frames=True,
        )
        EndoscapesFrameFeatureDataset.prepare(
            trainset, model, device, num_workers, featdir
        )
        EndoscapesFrameFeatureDataset.prepare(
            valset, model, device, num_workers, featdir
        )
        EndoscapesFrameFeatureDataset.prepare(
            testset, model, device, num_workers, featdir
        )
    else:
        assert (metadatadir is not None) and (videometapath is not None)
        metadata = read_ntucholecvs(metadatadir, videometapath)
        if no_icg:
            metadata = filter_no_icg(metadata)
        trainset, valset, testset = ntucholecvs_trainvaltest_datasets(
            metadata,
            datadir,
            image_size=384,
            disable_minmaxscale=disable_minmaxscale,
            augment=AugmentFlag.NONE,  # no augmentation
            cvs_score_items=cvs_score_items,
            include_unlabelled_frames=True,
        )
        NtucholecvsFrameFeatureDataset.prepare(
            trainset, model, device, num_workers, featdir
        )
        NtucholecvsFrameFeatureDataset.prepare(
            valset, model, device, num_workers, featdir
        )
        NtucholecvsFrameFeatureDataset.prepare(
            testset, model, device, num_workers, featdir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[
            dataset_parser(),
            cvsmodel_parser(),
            lora_parser(),
            compute_parser(),
        ]
    )
    parser.add_argument(
        "--swinv2_ckptpath",
        type=str,
        default=None,
        help="File path of pre-trained SwinV2 checkpoint",
    )
    parser.add_argument(
        "--ckptpath",
        type=str,
        default=None,
        help="File path of model checkpoint",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="Whether to use LoRA finetuning",
    )
    parser.add_argument(
        "--featdir",
        type=str,
        required=True,
        help="Directory containing features of video-frame sequences",
    )
    args = parser.parse_args()
    pathlib.Path(args.featdir).mkdir(parents=True, exist_ok=True)
    main(
        datasetname=args.dataset,
        datadir=pathlib.Path(args.datadir),
        metadatadir=(
            pathlib.Path(args.metadatadir) if args.metadatadir is not None else None
        ),
        videometapath=(
            pathlib.Path(args.videometapath) if args.videometapath is not None else None
        ),
        no_icg=args.no_icg,
        disable_minmaxscale=args.disable_minmaxscale,
        cvs_score_items=_cvs_score_items(args.clstrgt),
        swinv2_ckptpath=(
            pathlib.Path(args.swinv2_ckptpath)
            if args.swinv2_ckptpath is not None
            else None
        ),
        ckptpath=pathlib.Path(args.ckptpath) if args.ckptpath is not None else None,
        lora=args.lora,
        lora_targets=_lora_targets(args.lora_targets),
        lora_rank=args.lora_rank,
        num_workers=args.num_workers,
        device=args.device,
        featdir=pathlib.Path(args.featdir),
    )
