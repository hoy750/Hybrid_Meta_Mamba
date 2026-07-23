from __future__ import annotations

from torch import nn

from models.deep.torch_base import TorchBinaryClassifier


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden
        layers.append(nn.Linear(current, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class MLPClassifier(TorchBinaryClassifier):
    def __init__(self, hidden_dims: list[int] | None = None, dropout: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout

    def _build_network(self, input_shape: tuple[int, ...]) -> nn.Module:
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim
        return MLPNet(input_dim=input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)
