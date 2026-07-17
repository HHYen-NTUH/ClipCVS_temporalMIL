"""Utilities of SwinV2 model."""

from collections.abc import Callable
import re
import pathlib
import torch
from scripts.m_swinv2 import SwinTransformerV2
from model import load_model_checkpoint
from lora import LoraModel, print_trainable_parameters


class Swinv2SpatialCvsModel(torch.nn.Module):
    """Spatial module of a CVS model based on SwinV2 architecture."""

    def __init__(
        self,
        num_classes: int = 3,
        swincvs_ckptpath: pathlib.Path | None = None,
        swinv2_ckptpath: pathlib.Path | None = None,
        lora_factory: Callable[[SwinTransformerV2], LoraModel] | None = None,
    ) -> None:
        super().__init__()
        encoder = SwinTransformerV2(
            img_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=24,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[12, 12, 12, 6],
        )
        if swinv2_ckptpath is not None:
            encoder.head = torch.nn.Linear(1024, 3)
            load_model_checkpoint(encoder, swinv2_ckptpath)
        encoder.head = torch.nn.Identity()
        if swincvs_ckptpath is not None:
            load_swincvs_extractor_checkpoint(encoder, swincvs_ckptpath)
        self.encoder = encoder
        if lora_factory is not None:
            self.encoder = lora_factory(self.encoder)
            print_trainable_parameters(self.encoder)
        self.head = torch.nn.Linear(1024, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return extracted features."""
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        features = self.forward_features(inputs)
        logits = self.head(features)
        return logits


def load_swincvs_extractor_checkpoint(
    encoder: SwinTransformerV2, ckptpath: pathlib.Path
) -> None:
    """Load the extractor of a SwinCVS checkpoint into the encoder model."""
    _state_dict = torch.load(ckptpath, map_location="cpu")
    state_dict = {}
    for key, val in _state_dict.items():
        match = re.match(r"swinv2_model\.(.*)", key)
        if match is not None:
            state_dict[match.group(1)] = val
    encoder.load_state_dict(state_dict)


def swinv2_layerwise_lr_decay_paramgroups(
    model: SwinTransformerV2, baselr: float, decay: float
) -> list[dict]:
    num_layers = len(model.layers)
    paramgrps = [
        {
            "params": model.patch_embed.parameters(),
            "lr": baselr * decay ** (num_layers + 1),
        },
        *[
            {
                "params": model.layers[depth].parameters(),
                "lr": baselr * decay ** (num_layers - depth),
            }
            for depth in range(num_layers)
        ],
    ]
    return paramgrps


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    pass
