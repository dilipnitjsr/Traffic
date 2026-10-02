from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, hidden: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(num_nodes * num_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        z = x.reshape(batch, steps, nodes * features)
        output, _ = self.lstm(z)
        return self.head(output[:, -1])


class TCNForecaster(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, hidden: int = 64):
        super().__init__()
        channels = num_nodes * num_features
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        z = x.reshape(batch, steps, nodes * features).transpose(1, 2)
        z = self.network(z)
        # Convolution padding intentionally grows the sequence; last position
        # summarizes the causal receptive field for this compact baseline.
        return self.head(z[:, :, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_projection = nn.Linear(num_nodes * num_features, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        z = x.reshape(batch, steps, nodes * features)
        z = self.position(self.input_projection(z))
        z = self.encoder(z)
        return self.head(z[:, -1])


class STGNNForecaster(nn.Module):
    """Spatial graph convolution + temporal GRU baseline.

    At each timestamp, normalized adjacency mixes neighbouring junction
    representations. A GRU then models each junction's temporal evolution.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        adjacency: torch.Tensor,
        graph_hidden: int = 32,
        temporal_hidden: int = 64,
    ):
        super().__init__()
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency shape must be [num_nodes, num_nodes]")

        self.num_nodes = num_nodes
        self.register_buffer("adjacency", adjacency.float())
        self.graph_projection = nn.Linear(num_features, graph_hidden)
        self.temporal = nn.GRU(graph_hidden, temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        h = torch.relu(self.graph_projection(x))
        h = torch.einsum("nm,btmh->btnh", self.adjacency, h)

        batch, steps, nodes, hidden = h.shape
        node_sequences = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        encoded, _ = self.temporal(node_sequences)
        pred = self.head(encoded[:, -1]).reshape(batch, nodes)
        return pred


def make_torch_model(
    name: str,
    num_nodes: int,
    num_features: int,
    adjacency: torch.Tensor,
) -> nn.Module:
    key = name.lower()
    if key == "lstm":
        return LSTMForecaster(num_nodes, num_features)
    if key == "tcn":
        return TCNForecaster(num_nodes, num_features)
    if key == "transformer":
        return TransformerForecaster(num_nodes, num_features)
    if key == "stgnn":
        return STGNNForecaster(num_nodes, num_features, adjacency)
    raise ValueError(f"Unknown torch model: {name}")
