"""
Character-level deep learning models (Planejamento §6.2, PyTorch — D-001).

Each model embeds the integer-encoded URL, processes the character sequence, and
emits a **single logit** (use `BCEWithLogitsLoss`). Architecture dims come from
`src.config`. The embedding reserves index 0 for padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CNN_FILTERS, CNN_KERNEL, DL_DROPOUT, DL_EMBED_DIM, LSTM_HIDDEN

DEEP_MODEL_DISPLAY: dict[str, str] = {"cnn": "CNN", "lstm": "LSTM", "cnnlstm": "CNN-LSTM"}
DEEP_MODEL_NAMES: list[str] = list(DEEP_MODEL_DISPLAY)


class CharCNN(nn.Module):
    """Embedding -> Conv1d -> global max-pool -> Dense -> Dropout -> logit."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, DL_EMBED_DIM, padding_idx=0)
        self.conv = nn.Conv1d(DL_EMBED_DIM, CNN_FILTERS, CNN_KERNEL, padding=CNN_KERNEL // 2)
        self.fc1 = nn.Linear(CNN_FILTERS, 64)
        self.dropout = nn.Dropout(DL_DROPOUT)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)        # [B, embed, L]
        x = torch.relu(self.conv(x))             # [B, filters, L]
        x = torch.amax(x, dim=2)                 # global max-pool -> [B, filters]
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)            # [B] logits


class CharLSTM(nn.Module):
    """Embedding -> LSTM -> Dense -> Dropout -> logit."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, DL_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(DL_EMBED_DIM, LSTM_HIDDEN, batch_first=True)
        self.fc1 = nn.Linear(LSTM_HIDDEN, 64)
        self.dropout = nn.Dropout(DL_DROPOUT)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)                        # [B, L, embed]
        _, (h, _) = self.lstm(x)                 # h: [1, B, hidden]
        x = self.dropout(torch.relu(self.fc1(h[-1])))
        return self.fc2(x).squeeze(1)


class CharCNNLSTM(nn.Module):
    """Embedding -> Conv1d -> MaxPool -> LSTM -> Dropout -> logit (Alshingiti 2023)."""

    _LSTM_HIDDEN = 64

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, DL_EMBED_DIM, padding_idx=0)
        self.conv = nn.Conv1d(DL_EMBED_DIM, CNN_FILTERS, CNN_KERNEL, padding=CNN_KERNEL // 2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(CNN_FILTERS, self._LSTM_HIDDEN, batch_first=True)
        self.dropout = nn.Dropout(DL_DROPOUT)
        self.fc = nn.Linear(self._LSTM_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)        # [B, embed, L]
        x = self.pool(torch.relu(self.conv(x)))  # [B, filters, L/2]
        x = x.transpose(1, 2)                    # [B, L/2, filters]
        _, (h, _) = self.lstm(x)                 # h: [1, B, hidden]
        return self.fc(self.dropout(h[-1])).squeeze(1)


_FACTORY = {"cnn": CharCNN, "lstm": CharLSTM, "cnnlstm": CharCNNLSTM}


def get_deep_model(name: str, vocab_size: int) -> nn.Module:
    """Instantiate a DL model by key (``cnn`` | ``lstm`` | ``cnnlstm``)."""
    key = name.lower()
    if key not in _FACTORY:
        raise KeyError(f"Unknown DL model '{name}'. Known: {DEEP_MODEL_NAMES}")
    return _FACTORY[key](vocab_size)
