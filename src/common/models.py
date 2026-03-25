from __future__ import annotations

import torch
import torch.nn as nn


class CustomRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.input_linear(x_t) + self.hidden_linear(h_prev))


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h_prev], dim=-1)
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_hat = torch.tanh(self.candidate_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_t = f_t * c_prev + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class CustomRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cell_type: str = "rnn"):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        if cell_type == "rnn":
            self.cell = CustomRNNCell(input_size, hidden_size)
        elif cell_type == "lstm":
            self.cell = CustomLSTMCell(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            if self.cell_type == "rnn":
                h_t = self.cell(x_t, h_t)
            else:
                h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_t


class DecryptionModel(nn.Module):
    def __init__(self, input_vocab_size: int, output_vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float, cell_type: str):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.rnn = CustomRNNLayer(embedding_dim, hidden_size, cell_type=cell_type)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        out = self.dropout(out)
        return self.head(out)


class CustomBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fw = CustomRNNLayer(embedding_dim, hidden_size, cell_type="lstm")
        self.bw = CustomRNNLayer(embedding_dim, hidden_size, cell_type="lstm")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        fw_out, _ = self.fw(emb)
        bw_in = torch.flip(emb, dims=[1])
        bw_out, _ = self.bw(bw_in)
        bw_out = torch.flip(bw_out, dims=[1])
        out = torch.cat([fw_out, bw_out], dim=-1)
        out = self.dropout(out)
        return self.head(out)


class SimpleSSM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, state_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.a = nn.Parameter(torch.randn(state_size, state_size) * 0.02)
        self.b = nn.Parameter(torch.randn(embedding_dim, state_size) * 0.02)
        self.c = nn.Linear(state_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.state_size = state_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        batch_size, seq_len, _ = emb.shape
        state = torch.zeros(batch_size, self.state_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            state = torch.tanh(state @ self.a + emb[:, t, :] @ self.b)
            logits = self.c(self.dropout(state))
            outputs.append(logits.unsqueeze(1))
        return torch.cat(outputs, dim=1)
