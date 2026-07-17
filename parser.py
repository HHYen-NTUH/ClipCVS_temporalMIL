"""Utilities of command-line argument parsers."""

import argparse


def ntucholecvs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--metadatadir",
        type=str,
        required=True,
        help="Directory containing metadata of labelled video frames",
    )
    parser.add_argument(
        "--videometapath",
        type=str,
        required=True,
        help="File path of metadata of all video",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Directory containing all the video frames",
    )
    parser.add_argument(
        "--no_icg",
        default=False,
        action="store_true",
        help="Whether to exclude video with ICG",
    )
    return parser


def dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        choices=["ntucholecvs", "endoscapes"],
        default="ntucholecvs",
        help="Which dataset to use",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Directory containing all the video frames of the NTUcholeCVS dataset, or "
        "the directory of the Endoscapes dataset",
    )
    parser.add_argument(
        "--metadatadir",
        type=str,
        default=None,
        help="Directory containing metadata of labelled video frames of the "
        "NTUcholeCVS dataset",
    )
    parser.add_argument(
        "--videometapath",
        type=str,
        default=None,
        help="File path of metadata of all video of the NTUcholeCVS dataset",
    )
    parser.add_argument(
        "--no_icg",
        default=False,
        action="store_true",
        help="Whether to exclude video with ICG in the NTUcholeCVS dataset",
    )
    parser.add_argument(
        "--disable_minmaxscale",
        default=False,
        action="store_true",
        help="Whether to disable min-max scaling when loading data",
    )
    return parser


def cvsmodel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--spatial_model_ckptpath",
        type=str,
        default=None,
        help="File path of a spatial model checkpoint",
    )
    parser.add_argument(
        "--temporal_model_ckptpath",
        type=str,
        default=None,
        help="File path of a temporal model checkpoint",
    )
    parser.add_argument(
        "--clstrgt",
        choices=["C1", "C2", "C3"],
        nargs="+",
        default=["C1", "C2", "C3"],
        help="CVS score items to be the classification target",
    )
    parser.add_argument(
        "--temporal_num_layers",
        type=int,
        default=2,
        help="Number of layers in the temporal model",
    )
    parser.add_argument(
        "--temporal_hidden_size",
        type=int,
        default=256,
        help="Size of hidden state of the temporal model",
    )
    parser.add_argument(
        "--temporal_dropout",
        type=float,
        default=0,
        help="Dropout rate of the temporal model",
    )
    parser.add_argument(
        "--temporal_disable_bias",
        default=False,
        action="store_true",
        help="Whether to disable bias weights in the temporal model",
    )
    parser.add_argument(
        "--temporal_bidirectional",
        default=False,
        action="store_true",
        help="Whether to use the bidirectional version of the temporal model",
    )
    return parser


def lora_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lora_targets",
        choices=["qkv", "proj", "mlp", "cpb"],
        nargs="+",
        default=["qkv"],
        help="Targets modules of LoRA in the CTransPath model",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA")
    parser.add_argument(
        "--lora_scale",
        type=int,
        default=1,
        help="Scaling parameter of the LoRA layers' output",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability of LoRA layers",
    )
    parser.add_argument(
        "--lora_init",
        choices=["default", "gaussian", "pissa"],
        default="default",
        help="Way to initialize weights of LoRA layers",
    )
    parser.add_argument(
        "--loraplus_ratio",
        type=float,
        default=None,
        help=(
            "Learning rate ratio of the LoRA+ optimizer; if not specified, fallback "
            "to the default optimizer"
        ),
    )
    return parser


def learn_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate of optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay of optimizer"
    )
    parser.add_argument(
        "--label_weights",
        type=float,
        default=None,
        nargs="+",
        help="Label weights for class balance",
    )
    parser.add_argument(
        "--gradclip", type=float, default=None, help="L2 norm of gradient clipping"
    )
    parser.add_argument(
        "--use_warmup_cosine_scheduler",
        default=False,
        action="store_true",
        help="Whether to use warmup-and-cosine-decay learning rate scheduler",
    )
    parser.add_argument(
        "--num_warmup_epochs", type=int, default=40, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--cosine_decay_minlr",
        type=float,
        default=0,
        help="Minimum learning rate of cosine decay schedule",
    )
    return parser


def compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batchsize", type=int, default=4, help="Batch size of data processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for computation"
    )
    return parser


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_accml_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate the gradients",
    )
    parser.add_argument(
        "--logfreq",
        type=int,
        default=25,
        help="Logging interval among training batches",
    )
    parser.add_argument(
        "--mlflow_server_uri",
        type=str,
        required=True,
        help=(
            "URI of tracking server to store MLflow databases; if it is a local "
            "directory, it must have the 'file:' prefix"
        ),
    )
    parser.add_argument(
        "--mlflow_expname",
        type=str,
        required=True,
        help="Experiment name for MLflow logging",
    )
    parser.add_argument(
        "--mlflow_runname",
        type=str,
        required=True,
        help="Name of the experiemnt run for MLflow logging",
    )
    parser.add_argument(
        "--mlflow_rundesc",
        type=str,
        default=None,
        help="Description of the experiment run for MLflow logging",
    )
    parser.add_argument(
        "--ckptdir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--ckptname",
        type=str,
        required=True,
        help="File name of the selected model checkpoint",
    )
    parser.add_argument(
        "--ckptmetric",
        choices=["loss", "map", "last"],
        default="loss",
        help="By which validation metric to select the best checkpoint",
    )
    parser.add_argument(
        "--save_last_ckpt",
        default=False,
        action="store_true",
        help="Whether to save the last checkpoint as well",
    )
    parser.add_argument(
        "--last_ckptname",
        type=str,
        default="last_ckpt.pt",
        help="File name of the last model checkpoint",
    )
    parser.add_argument(
        "--loggradnorm",
        default=False,
        action="store_true",
        help="Whether to log gradient norm during training",
    )
    return parser


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
