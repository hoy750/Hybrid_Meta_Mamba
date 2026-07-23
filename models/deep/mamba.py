from __future__ import annotations

import torch
from torch import nn

from models.base import OptionalDependencyError
from models.deep.torch_base import (
    MetaProbabilitySequenceMixin,
    SharedStageSequenceMixin,
    StageSequenceMixin,
    TorchBinaryClassifier,
)


def mamba2_in_projection_dim(
    hidden_dim: int,
    state_dim: int,
    expand: int,
    head_dim: int,
    ngroups: int,
) -> int:
    d_inner = hidden_dim * expand
    if d_inner % head_dim != 0:
        raise ValueError(
            "Mamba2 requires hidden_dim * expand to be divisible by head_dim; "
            f"got hidden_dim={hidden_dim}, expand={expand}, head_dim={head_dim}."
        )
    nheads = d_inner // head_dim
    return 2 * d_inner + 2 * ngroups * state_dim + nheads


def validate_mamba2_cuda_alignment(
    hidden_dim: int,
    state_dim: int,
    expand: int,
    head_dim: int,
    ngroups: int,
) -> None:
    projection_dim = mamba2_in_projection_dim(hidden_dim, state_dim, expand, head_dim, ngroups)
    if projection_dim % 8 != 0:
        raise ValueError(
            "Invalid Mamba2 CUDA alignment: internal projection dimension must be a multiple of 8 "
            "for causal_conv1d channel-last fast path. "
            f"Current projection_dim={projection_dim} from hidden_dim={hidden_dim}, "
            f"state_dim={state_dim}, expand={expand}, head_dim={head_dim}, ngroups={ngroups}. "
            "Use a stride-safe combination such as expand=4 for this project."
        )


class MambaSequenceNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        conv_kernel: int,
        expand: int,
        head_dim: int,
        ngroups: int,
        chunk_size: int,
        dropout: float,
        require_mamba: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.using_mamba2 = False
        try:
            from mamba_ssm import Mamba2
        except Exception as exc:
            if require_mamba:
                raise OptionalDependencyError("Install mamba-ssm with a compatible torch/CUDA build.") from exc
            self.backbone = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            validate_mamba2_cuda_alignment(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                expand=expand,
                head_dim=head_dim,
                ngroups=ngroups,
            )
            print(f"backbone为mamba2")
            self.backbone = Mamba2(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=conv_kernel,
                expand=expand,
                headdim=head_dim,
                ngroups=ngroups,
                chunk_size=chunk_size,
            )
            self.using_mamba2 = True
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def _encode(self, x):
        z = self.input_proj(x)
        if self.using_mamba2:
            return self.backbone(z)
        output, _ = self.backbone(z)
        return output

    def forward(self, x):
        output = self._encode(x)
        return self.head(output[:, -1, :])


class MaskedMambaSequenceNet(MambaSequenceNet):
    def forward(self, x):
        visibility = x[:, :, -1] > 0.5
        lengths = visibility.sum(dim=1).clamp(min=1).long()
        output = self._encode(x)
        batch_index = torch.arange(output.shape[0], device=output.device)
        pooled = output[batch_index, lengths - 1, :]
        return self.head(pooled)


class BaseMambaClassifier(StageSequenceMixin, TorchBinaryClassifier):
    def __init__(
        self,
        hidden_dim: int = 256,
        state_dim: int = 256,
        conv_kernel: int = 4,
        expand: int = 4,
        head_dim: int = 128,
        ngroups: int = 1,
        chunk_size: int = 256,
        dropout: float = 0.3,
        require_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.head_dim = head_dim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.require_mamba = require_mamba

    def _build_network(self, input_shape: tuple[int, ...]) -> nn.Module:
        return MambaSequenceNet(
            input_dim=input_shape[-1],
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            conv_kernel=self.conv_kernel,
            expand=self.expand,
            head_dim=self.head_dim,
            ngroups=self.ngroups,
            chunk_size=self.chunk_size,
            dropout=self.dropout,
            require_mamba=self.require_mamba,
        )


class SharedStageMambaClassifier(SharedStageSequenceMixin, TorchBinaryClassifier):
    def __init__(
        self,
        hidden_dim: int = 256,
        state_dim: int = 256,
        conv_kernel: int = 4,
        expand: int = 4,
        head_dim: int = 128,
        ngroups: int = 1,
        chunk_size: int = 256,
        dropout: float = 0.3,
        require_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.head_dim = head_dim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.require_mamba = require_mamba
        # print(f"SharedStageMambaClassifier initialized with hidden_dim={hidden_dim}, state_dim={state_dim}, "
        #       f"conv_kernel={conv_kernel}, expand={expand}, head_dim={head_dim}, ngroups={ngroups}, "
        #       f"chunk_size={chunk_size}, dropout={dropout}, require_mamba={require_mamba}")
    def _build_network(self, input_shape: tuple[int, ...]) -> nn.Module:
        return MaskedMambaSequenceNet(
            input_dim=input_shape[-1],
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            conv_kernel=self.conv_kernel,
            expand=self.expand,
            head_dim=self.head_dim,
            ngroups=self.ngroups,
            chunk_size=self.chunk_size,
            dropout=self.dropout,
            require_mamba=self.require_mamba,
        )


class MetaMambaClassifier(MetaProbabilitySequenceMixin, BaseMambaClassifier):
    pass
