"""Utilities of MIL heads."""

import torch


def _mlp_encoder(input_size: int, hidden_size: int, dropout: float) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_size, hidden_size),
    )


class MeanMilTemporalCvsModel(torch.nn.Module):
    """Temporal module of a CVS model based on mean MIL pooling."""

    def __init__(
        self, num_classes: int, input_size: int, hidden_size: int, dropout: float = 0
    ) -> None:
        super().__init__()
        self.encoder = _mlp_encoder(input_size, hidden_size, dropout)
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(feats)
        feats, _ = feats.max(dim=1)
        logits = self.head(feats)
        return logits


class MaxMilTemporalCvsModel(torch.nn.Module):
    """Temporal module of a CVS model based on max MIL pooling."""

    def __init__(
        self, num_classes: int, input_size: int, hidden_size: int, dropout: float = 0
    ) -> None:
        super().__init__()
        self.encoder = _mlp_encoder(input_size, hidden_size, dropout)
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(feats)
        feats = feats.mean(dim=1)
        logits = self.head(feats)
        return logits


class GatedAttnMilTemporalCvsModel(torch.nn.Module):
    """Temporal module of a CVS model based on gated attention MIL."""

    def __init__(
        self, num_classes: int, input_size: int, hidden_size: int, dropout: float = 0
    ) -> None:
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.attention_u = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Tanh(),
        )
        self.attention_v = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Sigmoid(),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_size // 2, 1)
        )
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.projection(feats)
        attns = torch.nn.functional.softmax(
            self.attention(self.attention_u(feats) * self.attention_v(feats)), dim=1
        )
        logits = self.head((attns * feats).sum(dim=1))
        return logits


class SelfAttnMilTemporalCvsModel(torch.nn.Module):
    """Temporal module of a CVS model based on self-attention MIL."""

    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.projection(feats)
        feats, _ = self.attention(feats, feats, feats)
        logits = self.head(feats.mean(dim=1))
        return logits


MIL_FACTORY = {
    "Mean MIL": MeanMilTemporalCvsModel,
    "Max MIL": MaxMilTemporalCvsModel,
    "Gated attention MIL": GatedAttnMilTemporalCvsModel,
    "Self attention MIL": SelfAttnMilTemporalCvsModel,
}


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
