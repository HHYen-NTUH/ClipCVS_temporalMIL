"""Utilities of model builder."""

import pathlib
import torch


def load_model_checkpoint(model: torch.nn.Module, ckptpath: pathlib.Path) -> None:
    model.load_state_dict(torch.load(ckptpath, map_location="cpu"))


class LstmTemporalCvsModel(torch.nn.Module):
    """Temporal module of a CVS model based on LSTM."""

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bias=bias,
            bidirectional=bidirectional,
        )
        self.head = torch.nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_classes
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(features)
        logits = self.head(outputs[:, -1, :])  # logits of the last hidden state
        return logits


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
