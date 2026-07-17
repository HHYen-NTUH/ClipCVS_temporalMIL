"""Utilities of LoRA fine-tuning."""

from collections.abc import Callable
import enum
import itertools
import functools
import operator
from transformers import PreTrainedModel
from peft import LoraConfig, LoraModel


LORAFACTORY = Callable[[PreTrainedModel], LoraModel]


class LoraTarget(enum.Flag):
    """Target module of LoRA."""

    QKV = enum.auto()
    PROJ = enum.auto()
    MLP = enum.auto()


def lora_targets(names: list[str]) -> LoraTarget:
    return functools.reduce(operator.or_, [LoraTarget[name.upper()] for name in names])


LORA_TARGET_MODULES: dict[str, list[str]] = {
    "qkv": ["attn.qkv"],
    "proj": ["attn.proj"],
    "mlp": ["mlp.fc1", "mlp.fc2"],
}


def lora_target_modules(targets: LoraTarget) -> list[str]:
    return list(
        itertools.chain.from_iterable(
            LORA_TARGET_MODULES[trgt.name.lower()]
            for trgt in targets
            if trgt.name is not None
        )
    )


def lora_model_factory(
    targets: LoraTarget,
    rank: int,
    scale: int = 1,
    dropout: float = 0,
) -> LORAFACTORY:
    def factory(model: PreTrainedModel) -> LoraModel:
        cfg = LoraConfig(
            target_modules=lora_target_modules(targets),
            r=rank,
            lora_alpha=scale * rank,
            lora_dropout=dropout,
        )
        return LoraModel(model, cfg, adapter_name="default")

    return factory


def print_trainable_parameters(model: LoraModel) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        f"trainable parameters: {trainable:,d} || total parameters: {total:,d}"
        f" || trainable percentage: {(100 * trainable / total):.3f}%"
    )


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
