"""Predict CVS scores by a LSTM model."""

import argparse
from collections import namedtuple
from typing import Literal
import pathlib
import itertools
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from parser import compute_parser, dataset_parser, cvsmodel_parser
from _analyze_data_distribution import read_ntucholecvs, filter_no_icg, read_endoscapes
from data import (
    trainvaltest_flexible_sequence_datasets as ntucholecvs_trainvaltest_flexible_sequence_datasets,
    FlexibleFrameSequnceFeatureDataset as NtucholecvsSequenceFeatureDataset,
    CvsScoreItem,
    cvs_score_items as _cvs_score_items,
)
from endoscapes import (
    trainvaltest_flexible_sequence_datasets as endoscapes_trainvaltest_flexible_sequence_datasets,
    FlexibleFrameSequnceFeatureDataset as EndoscapesSequenceFeatureDataset,
)
from model import LstmTemporalCvsModel, load_model_checkpoint
from evaluate_prediction import DATACOLS, TRGTCOLSUFFIX, PREDCOLSUFFIX


def main(
    datasetname: str,
    datadir: pathlib.Path,
    metadatadir: pathlib.Path | None,
    videometapath: pathlib.Path | None,
    no_icg: bool,
    cvs_score_items: CvsScoreItem,
    seqlen: int,
    lblagg_method: Literal["union", "more_than_50%", "last"],
    featdir: pathlib.Path,
    stack_frame_feats: bool,
    input_size: int,
    num_layers: int,
    hidden_size: int,
    bias: bool,
    bidirectional: bool,
    ckptpath: pathlib.Path,
    batchsize: int,
    num_workers: int,
    device: str,
    predpath: pathlib.Path,
) -> None:
    """Save CVS prediction results along with the groundtruth."""
    # Instantiate data loaders
    if datasetname == "endoscapes":
        metadata = read_endoscapes(datadir)
        _, _, testset = endoscapes_trainvaltest_flexible_sequence_datasets(
            metadata,
            datadir,
            image_size=224,
            seqlen=seqlen,
            cvs_score_items=cvs_score_items,
            label_method=lblagg_method,
        )
        testset = EndoscapesSequenceFeatureDataset(testset, featdir, stack_frame_feats)
        testloader = DataLoader(
            testset, batch_size=batchsize, num_workers=num_workers - 1, shuffle=False
        )
    else:
        assert (metadatadir is not None) and (videometapath is not None)
        metadata = read_ntucholecvs(metadatadir, videometapath)
        if no_icg:
            metadata = filter_no_icg(metadata)
        _, _, testset = ntucholecvs_trainvaltest_flexible_sequence_datasets(
            metadata,
            datadir,
            image_size=224,
            seqlen=seqlen,
            cvs_score_items=cvs_score_items,
            label_method=lblagg_method,
        )
        testset = NtucholecvsSequenceFeatureDataset(testset, featdir, stack_frame_feats)
        testloader = DataLoader(
            testset, batch_size=batchsize, num_workers=num_workers - 1, shuffle=False
        )
    assert testloader.batch_sampler is not None
    # Instantiate model
    model = LstmTemporalCvsModel(
        input_size=input_size,
        num_classes=len(cvs_score_items),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        bidirectional=bidirectional,
    )
    model.eval()
    load_model_checkpoint(model, ckptpath)
    model = model.to(device)
    # Generate prediction
    PredRow = namedtuple(
        "PredRow",
        itertools.chain(
            DATACOLS,
            [f"{item.name}{TRGTCOLSUFFIX}" for item in cvs_score_items],
            [f"{item.name}{PREDCOLSUFFIX}" for item in cvs_score_items],
        ),
    )
    prediction = []
    video_names, frames_names = map(list, zip(*testset.dirfilenames))
    with torch.inference_mode():
        for (feats, trgts), inds in zip(
            tqdm(testloader, desc="Predict"), testloader.batch_sampler
        ):
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.nn.functional.sigmoid(logits).cpu()
            trgts = trgts.int().cpu()
            prediction.extend(
                [
                    PredRow(
                        video_names[ind],
                        frames_names[ind],
                        *_trgts.tolist(),
                        *_probs.tolist(),
                    )
                    for ind, _trgts, _probs in zip(inds, trgts, probs)
                ]
            )
    # Save prediction
    pd.DataFrame(prediction).to_csv(predpath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[
            dataset_parser(),
            cvsmodel_parser(),
            compute_parser(),
        ]
    )
    parser.add_argument(
        "--seqlen", type=int, required=True, help="Length of frame sequence"
    )
    parser.add_argument(
        "--lblagg_method",
        choices=["union", "more_than_50%", "last"],
        default="union",
        help="Method to aggregate label of frame sequences",
    )
    parser.add_argument(
        "--featdir",
        type=str,
        required=True,
        help="Directory containing features of video-frame sequences",
    )
    parser.add_argument(
        "--stack_frame_feats",
        default=False,
        action="store_true",
        help="Whether to load frame features and stack them",
    )
    parser.add_argument(
        "--featsize", type=int, default=768, help="Size of feature array"
    )
    parser.add_argument(
        "--ckptpath",
        type=str,
        required=True,
        help="File path of model checkpoint",
    )
    parser.add_argument(
        "--predpath", type=str, required=True, help="File path of CVS prediction"
    )
    args = parser.parse_args()
    pathlib.Path(args.predpath).parent.mkdir(parents=True, exist_ok=True)
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
        cvs_score_items=_cvs_score_items(args.clstrgt),
        seqlen=args.seqlen,
        lblagg_method=args.lblagg_method,
        featdir=pathlib.Path(args.featdir),
        stack_frame_feats=args.stack_frame_feats,
        input_size=args.featsize,
        num_layers=args.temporal_num_layers,
        hidden_size=args.temporal_hidden_size,
        bias=not args.temporal_disable_bias,
        bidirectional=args.temporal_bidirectional,
        ckptpath=pathlib.Path(args.ckptpath),
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        device=args.device,
        predpath=pathlib.Path(args.predpath),
    )
