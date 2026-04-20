"""Bidirectional LSTM baseline for stroke classification."""

import torch
import torch.nn as nn


class StrokeLSTM(nn.Module):
    """Many-to-One Bidirectional LSTM for temporal pose classification.

    Input: (batch, T, input_dim) where input_dim = num_keypoints * 3
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_dim: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM output: (batch, T, hidden_dim * num_directions)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            forward_final = h_n[-2]  # last layer, forward
            backward_final = h_n[-1]  # last layer, backward
            hidden = torch.cat([forward_final, backward_final], dim=1)
        else:
            hidden = h_n[-1]

        return self.fc(hidden)


def build_lstm(config: dict) -> StrokeLSTM:
    """Build LSTM model from config."""
    lstm_cfg = config["model"]["lstm"]
    return StrokeLSTM(
        input_dim=lstm_cfg["input_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        num_layers=lstm_cfg["num_layers"],
        num_classes=config["classes"]["num_classes"],
        dropout=lstm_cfg["dropout"],
        bidirectional=lstm_cfg["bidirectional"],
        fc_dim=lstm_cfg["fc_dim"],
    )
