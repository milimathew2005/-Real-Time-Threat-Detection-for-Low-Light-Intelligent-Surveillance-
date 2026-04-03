"""
model.py — CrimeLSTM Architecture
Real-Time Threat Detection for Low-Light Intelligent Surveillance
ARGUS Platform — Muthoot Institute of Technology and Science, Kochi

Architecture:
    Input (B, T, 2048)
      → LayerNorm
      → Bidirectional LSTM ×2
      → Temporal Attention
      → FC head (1024 → 256 → NUM_CLASSES)
      → Softmax probabilities
"""

import torch
import torch.nn as nn
from config import CNN_FEATURE_DIM, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT, FC_HIDDEN, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Soft attention over the time axis.
    Highlights which frames in the window contributed most to the prediction.
    Useful for explainability: high-attention frames are the "most suspicious" moments.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        scores  = self.attn(x).squeeze(-1)            # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# CrimeLSTM
# ─────────────────────────────────────────────────────────────────────────────
class CrimeLSTM(nn.Module):
    """
    Bidirectional LSTM with temporal attention for UCF-Crime classification.

    Input  : sequence of CNN features (B, T, CNN_FEATURE_DIM)
    Output : class logits (B, NUM_CLASSES),  attention weights (B, T)
    """

    def __init__(
        self,
        feature_dim: int = CNN_FEATURE_DIM,
        hidden_size: int = LSTM_HIDDEN,
        num_layers:  int = LSTM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout:     float = DROPOUT,
        fc_hidden:   int = FC_HIDDEN,
    ):
        super().__init__()
        lstm_out = hidden_size * 2   # bidirectional doubles the output dim

        self.norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attention  = TemporalAttention(lstm_out)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, feature_dim)  →  logits (B, C),  attn_weights (B, T)"""
        x = self.norm(x)
        lstm_out, _      = self.lstm(x)             # (B, T, hidden*2)
        context, weights = self.attention(lstm_out)
        return self.classifier(context), weights


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = None, **kwargs) -> CrimeLSTM:
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    return CrimeLSTM(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import SEQ_LEN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m      = build_model().to(device)
    total  = sum(p.numel() for p in m.parameters())
    print(f"Parameters  : {total:,}")
    dummy  = torch.randn(4, SEQ_LEN, CNN_FEATURE_DIM).to(device)
    logits, attn = m(dummy)
    print(f"Output shape: {logits.shape}   Attn: {attn.shape}")
