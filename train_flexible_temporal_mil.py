"""Train a temporal MIL model of CVS classification."""

import argparse
from collections.abc import Callable
from typing import Any, Literal
import pathlib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchmetrics as tm
import mlflow
from parser import (
    dataset_parser,
    learn_parser,
    compute_parser,
    train_parser,
)
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
from model import load_model_checkpoint
from temporal_mil import MIL_FACTORY
from callback import CheckpointCallback, MlflowLogger, BatchLossAveraging


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: tm.MetricCollection,
    log_metrics_cb: Callable[[dict[str, Any], int], None],
    logfreq: int = 50,
    losskey: str = "Loss/train",
) -> None:
    """Train the model for an epoch."""
    device = next(model.parameters()).device
    loss_accmlr = BatchLossAveraging()
    for step, batch in enumerate(tqdm(loader, leave=False, desc="Train"), start=1):
        feats, trgts = batch
        feats, trgts = feats.to(device), trgts.to(device)
        logits = model(feats)
        loss = criterion(logits, trgts)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_accmlr.update(loss.item(), weight=trgts.shape[0])
        metrics.update(
            torch.nn.functional.sigmoid(logits).detach().cpu(), trgts.cpu().int()
        )
        if step % logfreq == 0:
            log_metrics_cb({losskey: loss_accmlr.compute()}, step)
            loss_accmlr.reset()
            log_metrics_cb(
                {key: val.item() for key, val in metrics.compute().items()}, step
            )
            metrics.reset()


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    metrics: tm.MetricCollection,
    log_metrics_cb: Callable[[dict[str, Any], int], None],
    ckptcb: CheckpointCallback,
    ckptmetric: str,
    negate_ckptmetric: bool = False,
    losskey: str = "Loss/val",
) -> None:
    """Validate the model at the end of an epoch."""
    _metricnames = [losskey, *metrics.keys()]
    assert (
        ckptmetric in _metricnames
    ), f"{ckptmetric} is not accessable for monitoring; available options are {_metricnames}"
    device = next(model.parameters()).device
    loss_accmlr = BatchLossAveraging()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Validate"):
            feats, trgts = batch
            feats, trgts = feats.to(device), trgts.to(device)
            logits = model(feats)
            loss = criterion(logits, trgts)
            loss_accmlr.update(loss.item(), weight=trgts.shape[0])
            metrics.update(torch.nn.functional.sigmoid(logits).cpu(), trgts.cpu().int())
    _loss = loss_accmlr.compute()
    log_metrics_cb({losskey: _loss}, 0)
    loss_accmlr.reset()
    _metrics = {key: val.item() for key, val in metrics.compute().items()}
    log_metrics_cb(_metrics, 0)
    metrics.reset()
    ckptcb.update_save(
        _loss if ckptmetric == losskey else _metrics[ckptmetric],
        model,
        negate_metric=negate_ckptmetric,
    )


def main(
    datasetname: str,
    datadir: pathlib.Path,
    metadatadir: pathlib.Path | None,
    videometapath: pathlib.Path | None,
    no_icg: bool,
    cvs_score_items: CvsScoreItem,
    seqlen: int,
    lblagg_method: Literal["union", "more_than_50%"],
    featdir: pathlib.Path,
    stack_frame_feats: bool,
    modeltype: str,
    input_size: int,
    hidden_size: int,
    dropout: float,
    num_heads: int,
    ckptpath: pathlib.Path | None,
    learning_rate: float,
    weight_decay: float,
    use_warmup_cosine_scheduler: bool,
    num_warmup_epochs: int,
    cosine_decay_minlr: float,
    label_weights: list[float] | None,
    batchsize: int,
    num_workers: int,
    device: str,
    num_epochs: int,
    logfreq: int,
    mlflow_server_uri: str,
    mlflow_expname: str,
    mlflow_runname: str,
    mlflow_rundesc: str,
    ckptdir: pathlib.Path,
    selected_ckptname: str,
    save_last_ckpt: bool,
    last_ckptname: str,
) -> None:
    """Save model training logs and the model checkpoints."""
    # Instantiate data loaders
    if datasetname == "endoscapes":
        metadata = read_endoscapes(datadir)
        trainset, valset, _ = endoscapes_trainvaltest_flexible_sequence_datasets(
            metadata,
            datadir,
            image_size=224,
            seqlen=seqlen,
            cvs_score_items=cvs_score_items,
            label_method=lblagg_method,
        )
        trainloader = DataLoader(
            EndoscapesSequenceFeatureDataset(trainset, featdir, stack_frame_feats),
            batch_size=batchsize,
            num_workers=num_workers - 1,
            shuffle=True,
        )
        valloader = DataLoader(
            EndoscapesSequenceFeatureDataset(valset, featdir, stack_frame_feats),
            batch_size=batchsize,
            num_workers=num_workers - 1,
            shuffle=False,
        )
    else:
        assert (metadatadir is not None) and (videometapath is not None)
        metadata = read_ntucholecvs(metadatadir, videometapath)
        if no_icg:
            metadata = filter_no_icg(metadata)
        trainset, valset, _ = ntucholecvs_trainvaltest_flexible_sequence_datasets(
            metadata,
            datadir,
            image_size=224,
            seqlen=seqlen,
            cvs_score_items=cvs_score_items,
            label_method=lblagg_method,
        )
        trainloader = DataLoader(
            NtucholecvsSequenceFeatureDataset(trainset, featdir, stack_frame_feats),
            batch_size=batchsize,
            num_workers=num_workers - 1,
            shuffle=True,
        )
        valloader = DataLoader(
            NtucholecvsSequenceFeatureDataset(valset, featdir, stack_frame_feats),
            batch_size=batchsize,
            num_workers=num_workers - 1,
            shuffle=False,
        )
    # Instantiate model, optimizer, learning rate scheduler, and criterion
    model_kwargs = {
        "num_classes": len(cvs_score_items),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "dropout": dropout,
    }
    if modeltype in ["Self attention MIL"]:
        model_kwargs["num_heads"] = num_heads
    model = MIL_FACTORY[modeltype](**model_kwargs)
    model.train()
    if ckptpath is not None:
        load_model_checkpoint(model, ckptpath)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if use_warmup_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1 / num_warmup_epochs,
                    total_iters=num_warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs - num_warmup_epochs,
                    eta_min=cosine_decay_minlr,
                ),
            ],
            [num_warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    if label_weights is not None:
        assert len(label_weights) == len(cvs_score_items), (
            f"Classification target consists of {len(cvs_score_items)} items but "
            f"{len(label_weights)} label weights are given"
        )
        _label_weights = torch.tensor(label_weights)
    else:
        _label_weights = torch.ones(len(cvs_score_items))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_label_weights)
    criterion = criterion.to(device)
    # Instantiate callbacks and tracking metrics
    logger = MlflowLogger(
        server_uri=mlflow_server_uri,
        expname=mlflow_expname,
        runname=mlflow_runname,
        rundesc=mlflow_rundesc,
    )
    ckptcb = CheckpointCallback(ckptdir=ckptdir, ckptname=selected_ckptname)
    metric_kwargs: dict = {
        "task": "multilabel" if len(cvs_score_items) > 1 else "binary",
    }
    if len(cvs_score_items) > 1:
        metric_kwargs["num_labels"] = len(cvs_score_items)
        metric_kwargs["average"] = "macro"
    metrics = tm.MetricCollection(
        {
            "MeanAveragePrecision": tm.AveragePrecision(**metric_kwargs),
            "MeanRecall": tm.Recall(threshold=0.5, **metric_kwargs),
            "MeanSpecificity": tm.Specificity(threshold=0.5, **metric_kwargs),
            "MeanPrecision": tm.Precision(threshold=0.5, **metric_kwargs),
        }
    )
    trainmetrics = metrics.clone(postfix="/train")
    valmetrics = metrics.clone(postfix="/val")
    # Train the model
    with logger.start_run():
        mlflow.log_params(
            {
                "metadatadir": metadatadir,
                "videometapath": videometapath,
                "datadir": datadir,
                "no_icg": no_icg,
                "cvs_score_items": str(cvs_score_items),
                "featdir": featdir,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "ckptpath": ckptpath,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "use_warmup_cosine_scheduler": use_warmup_cosine_scheduler,
                "num_warmup_epochs": num_warmup_epochs,
                "cosine_decay_minlr": cosine_decay_minlr,
                "label_weights": label_weights,
                "batchsize": batchsize,
                "device": device,
                "num_epochs": num_epochs,
                "logfreq": logfreq,
                "ckptdir": ckptdir,
                "selected_ckptname": selected_ckptname,
                "save_last_ckpt": save_last_ckpt,
                "last_ckptname": last_ckptname,
            }
        )
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            train(
                model=model,
                loader=trainloader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=trainmetrics,
                log_metrics_cb=logger.log_metrics_callback(
                    start_step=epoch * len(trainloader)
                ),
                logfreq=logfreq,
            )
            validate(
                model=model,
                loader=valloader,
                criterion=criterion,
                metrics=valmetrics,
                log_metrics_cb=logger.log_metrics_callback(
                    start_step=(epoch + 1) * len(trainloader)
                ),
                ckptcb=ckptcb,
                ckptmetric="Loss/val",
                negate_ckptmetric=True,
            )
            mlflow.log_metric(
                "LearningRate",
                scheduler.get_last_lr()[-1],
                step=(epoch + 1) * len(trainloader),
            )
            scheduler.step()
        mlflow.log_params({"best_ckptpath": ckptcb.best_ckptpath})
        print(f"Checkpoint saved at {ckptcb.best_ckptpath}")
        if save_last_ckpt:
            _ckptcb = CheckpointCallback(ckptdir=ckptdir, ckptname=last_ckptname)
            _ckptcb.save(model)
            last_ckptpath = _ckptcb.best_ckptpath
            mlflow.log_params({"last_ckptpath": last_ckptpath})
            print(f"Checkpoint saved at {last_ckptpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[
            dataset_parser(),
            learn_parser(),
            compute_parser(),
            train_parser(),
        ]
    )
    parser.add_argument(
        "--clstrgt",
        choices=["C1", "C2", "C3"],
        nargs="+",
        default=["C1", "C2", "C3"],
        help="CVS score items to be the classification target",
    )
    parser.add_argument(
        "--seqlen", type=int, required=True, help="Length of frame sequence"
    )
    parser.add_argument(
        "--lblagg_method",
        choices=["union", "more_than_50%"],
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
        "--modeltype",
        choices=MIL_FACTORY.keys(),
        required=True,
        help="Type of MIL model",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Size of hidden state"
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of attention heads, if applicable",
    )
    parser.add_argument(
        "--ckptpath",
        type=str,
        default=None,
        help="File path of model checkpoint as the initial weights",
    )
    args = parser.parse_args()
    pathlib.Path(args.ckptdir).mkdir(parents=True, exist_ok=True)
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
        modeltype=args.modeltype,
        input_size=args.featsize,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_heads=args.num_heads,
        ckptpath=pathlib.Path(args.ckptpath) if args.ckptpath is not None else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_warmup_cosine_scheduler=args.use_warmup_cosine_scheduler,
        num_warmup_epochs=args.num_warmup_epochs,
        cosine_decay_minlr=args.cosine_decay_minlr,
        label_weights=args.label_weights,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        device=args.device,
        num_epochs=args.num_epochs,
        logfreq=args.logfreq,
        mlflow_server_uri=args.mlflow_server_uri,
        mlflow_expname=args.mlflow_expname,
        mlflow_runname=args.mlflow_runname,
        mlflow_rundesc=args.mlflow_rundesc,
        ckptdir=pathlib.Path(args.ckptdir),
        selected_ckptname=args.ckptname,
        save_last_ckpt=args.save_last_ckpt,
        last_ckptname=args.last_ckptname,
    )
