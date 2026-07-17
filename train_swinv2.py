"""Train SwinV2 for CVS classification."""

import argparse
from collections.abc import Callable
from typing import Any
import pathlib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchmetrics as tm
import mlflow
from peft.optimizers import create_loraplus_optimizer
from _analyze_data_distribution import read_ntucholecvs, filter_no_icg, read_endoscapes
from data import (
    trainvaltest_datasets as ntucholecvs_trainvaltest_datasets,
    CvsScoreItem,
    cvs_score_items as _cvs_score_items,
)
from endoscapes import trainvaltest_datasets as endoscapes_trainvaltest_datasets
from augmentation import AugmentFlag
from model import load_model_checkpoint
from swinv2 import Swinv2SpatialCvsModel, swinv2_layerwise_lr_decay_paramgroups
from lora import (
    LoraTarget,
    LoraModel,
    lora_model_factory,
    lora_targets as _lora_targets,
)
from callback import CheckpointCallback, MlflowLogger, BatchLossAveraging
from parser import (
    dataset_parser,
    cvsmodel_parser,
    learn_parser,
    lora_parser,
    compute_parser,
    train_parser,
)


def train(
    model: Swinv2SpatialCvsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: tm.MetricCollection,
    log_metrics_cb: Callable[[dict[str, Any], int], None],
    logfreq: int = 50,
    losskey: str = "Loss/train",
    num_accml_batches: int = 1,
) -> None:
    """Train the model for an epoch."""
    device = next(model.parameters()).device
    loss_accmlr = BatchLossAveraging()
    for step, batch in enumerate(tqdm(loader, leave=False, desc="Train"), start=1):
        inputs, trgts = batch
        inputs, trgts = inputs.to(device), trgts.to(device)
        logits = model(inputs)
        loss = criterion(logits, trgts)
        loss = loss / num_accml_batches  # scale loss for mean reduction
        loss.backward()
        if step % num_accml_batches == 0:
            optimizer.step()
            optimizer.zero_grad()
        loss_accmlr.update(loss.item() * num_accml_batches, weight=trgts.shape[0])
        metrics.update(
            torch.nn.functional.sigmoid(logits).detach().cpu(), trgts.cpu().int()
        )
        if step % num_accml_batches == 0 and (step // num_accml_batches) % logfreq == 0:
            log_metrics_cb({losskey: loss_accmlr.compute()}, step // num_accml_batches)
            loss_accmlr.reset()
            log_metrics_cb(
                {key: val.item() for key, val in metrics.compute().items()},
                step // num_accml_batches,
            )
            metrics.reset()
    # Handle remaining gradient
    if len(loader) % num_accml_batches != 0:
        optimizer.step()
        optimizer.zero_grad()


def validate(
    model: Swinv2SpatialCvsModel,
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
            inputs, trgts = batch
            inputs, trgts = inputs.to(device), trgts.to(device)
            logits = model(inputs)
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
    disable_minmaxscale: bool,
    cvs_score_items: CvsScoreItem,
    swincvs_ckptpath: pathlib.Path | None,
    swinv2_ckptpath: pathlib.Path | None,
    ckptpath: pathlib.Path | None,
    lora: bool,
    lora_targets: LoraTarget,
    lora_rank: int,
    lora_scale: int,
    lora_dropout: float,
    loraplus_ratio: float | None,
    learning_rate: float,
    weight_decay: float,
    layerwise_lr_decay: float | None,
    use_warmup_cosine_scheduler: bool,
    num_warmup_epochs: int,
    cosine_decay_minlr: float,
    configure_opt_schdlr_as_paper: bool,
    label_weights: list[float] | None,
    batchsize: int,
    num_workers: int,
    device: str,
    num_epochs: int,
    num_accml_batches: int,
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
    """Save model checkpoint and training log."""
    # Instantiate data loaders
    if datasetname == "endoscapes":
        metadata = read_endoscapes(datadir)
        trainset, valset, _ = endoscapes_trainvaltest_datasets(
            metadata,
            datadir,
            image_size=384,
            disable_minmaxscale=disable_minmaxscale,
            augment=AugmentFlag.ZOOM | AugmentFlag.COLOR,  # no color jittering
            cvs_score_items=cvs_score_items,
        )
    else:
        assert (metadatadir is not None) and (videometapath is not None)
        metadata = read_ntucholecvs(metadatadir, videometapath)
        if no_icg:
            metadata = filter_no_icg(metadata)
        trainset, valset, _ = ntucholecvs_trainvaltest_datasets(
            metadata,
            datadir,
            image_size=384,
            disable_minmaxscale=disable_minmaxscale,
            augment=AugmentFlag.ZOOM | AugmentFlag.COLOR,  # no color jittering
            cvs_score_items=cvs_score_items,
        )
    trainloader = DataLoader(
        trainset, batch_size=batchsize, num_workers=num_workers - 1, shuffle=True
    )
    valloader = DataLoader(
        valset, batch_size=batchsize, num_workers=num_workers - 1, shuffle=False
    )
    # Instantiate model
    if lora:
        lora_factory = lora_model_factory(
            targets=lora_targets, rank=lora_rank, scale=lora_scale, dropout=lora_dropout
        )
    else:
        lora_factory = None
    model = Swinv2SpatialCvsModel(
        num_classes=len(cvs_score_items),
        swincvs_ckptpath=swincvs_ckptpath,
        swinv2_ckptpath=swinv2_ckptpath,
        lora_factory=lora_factory,
    )
    if ckptpath is not None:
        load_model_checkpoint(model, ckptpath)
    model = model.to(device)
    model.train()
    # Instantiate criterion
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
    # Instantiate optimizer and learning rate scheduler
    if configure_opt_schdlr_as_paper:
        learning_rate = 5e-4 * batchsize / 256
        cosine_decay_minlr = 1e-6 * batchsize / 256
        num_warmup_epochs = 5
        weight_decay = 0.05
        layerwise_lr_decay = 0.7
        assert num_epochs >= num_warmup_epochs
        optimizer = torch.optim.AdamW(
            [
                *swinv2_layerwise_lr_decay_paramgroups(
                    (
                        model.encoder.model
                        if isinstance(model.encoder, LoraModel)
                        else model.encoder
                    ),
                    baselr=learning_rate,
                    decay=layerwise_lr_decay,
                ),
                {"params": model.head.parameters(), "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )
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
        if layerwise_lr_decay is not None:
            optimizer = torch.optim.AdamW(
                [
                    *swinv2_layerwise_lr_decay_paramgroups(
                        (
                            model.encoder.model
                            if isinstance(model.encoder, LoraModel)
                            else model.encoder
                        ),
                        baselr=learning_rate,
                        decay=layerwise_lr_decay,
                    ),
                    {"params": model.head.parameters(), "lr": learning_rate},
                ],
                weight_decay=weight_decay,
            )
        else:
            if loraplus_ratio is not None and isinstance(model, LoraModel):
                optimizer = create_loraplus_optimizer(
                    model,
                    optimizer_cls=torch.optim.AdamW,
                    lr=learning_rate,
                    loraplus_lr_ratio=loraplus_ratio,
                    loraplus_weight_decay=weight_decay,
                )
            else:
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
    # Instantiate callbacks
    logger = MlflowLogger(
        server_uri=mlflow_server_uri,
        expname=mlflow_expname,
        runname=mlflow_runname,
        rundesc=mlflow_rundesc,
    )
    ckptcb = CheckpointCallback(ckptdir=ckptdir, ckptname=selected_ckptname)
    # Instantiate tracking metrics
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
                "disable_minmaxscale": disable_minmaxscale,
                "cvs_score_items": str(cvs_score_items),
                "lora": lora,
                "lora_targets": str(lora_targets),
                "lora_rank": lora_rank,
                "lora_scale": lora_scale,
                "lora_dropout": lora_dropout,
                "loraplus_ratio": loraplus_ratio,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "layerwise_lr_decay": layerwise_lr_decay,
                "use_warmup_cosine_scheduler": use_warmup_cosine_scheduler,
                "num_warmup_epochs": num_warmup_epochs,
                "cosine_decay_minlr": cosine_decay_minlr,
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
                num_accml_batches=num_accml_batches,
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
            cvsmodel_parser(),
            learn_parser(),
            lora_parser(),
            compute_parser(),
            train_parser(),
        ]
    )
    parser.add_argument(
        "--swincvs_ckptpath",
        type=str,
        default=None,
        help="File path of SwinCVS checkpoint whose backbone to be the initial weights",
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
        help="File path of model checkpoint as the initial weights",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="Whether to use LoRA finetuning",
    )
    parser.add_argument(
        "--layerwise_lr_decay",
        type=float,
        default=None,
        help="Ratio of layerwise learning rate decay",
    )
    parser.add_argument(
        "--configure_opt_schdlr_as_paper",
        default=False,
        action="store_true",
        help=(
            "Whether to configure the optimizer and the LR scheduler as the paper "
            "suggested"
        ),
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
        disable_minmaxscale=args.disable_minmaxscale,
        cvs_score_items=_cvs_score_items(args.clstrgt),
        swincvs_ckptpath=(
            pathlib.Path(args.swincvs_ckptpath)
            if args.swincvs_ckptpath is not None
            else None
        ),
        swinv2_ckptpath=(
            pathlib.Path(args.swinv2_ckptpath)
            if args.swinv2_ckptpath is not None
            else None
        ),
        ckptpath=pathlib.Path(args.ckptpath) if args.ckptpath is not None else None,
        lora=args.lora,
        lora_targets=_lora_targets(args.lora_targets),
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=args.lora_dropout,
        loraplus_ratio=args.loraplus_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        layerwise_lr_decay=args.layerwise_lr_decay,
        use_warmup_cosine_scheduler=args.use_warmup_cosine_scheduler,
        num_warmup_epochs=args.num_warmup_epochs,
        cosine_decay_minlr=args.cosine_decay_minlr,
        configure_opt_schdlr_as_paper=args.configure_opt_schdlr_as_paper,
        label_weights=args.label_weights,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        device=args.device,
        num_epochs=args.num_epochs,
        num_accml_batches=args.num_accml_batches,
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
