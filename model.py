import torch
import torch.nn as nn
from config import CNN_FEATURE_DIM, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT, FC_HIDDEN, NUM_CLASSES

class TemporalAttention(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1, bias=False))

    def forward(self, x: torch.Tensor):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return (context, weights)

class CrimeLSTM(nn.Module):

    def __init__(self, feature_dim: int=CNN_FEATURE_DIM, hidden_size: int=LSTM_HIDDEN, num_layers: int=LSTM_LAYERS, num_classes: int=NUM_CLASSES, dropout: float=DROPOUT, fc_hidden: int=FC_HIDDEN):
        super().__init__()
        lstm_out = hidden_size * 2
        self.norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(lstm_out)
        self.classifier = nn.Sequential(nn.Linear(lstm_out, fc_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fc_hidden, num_classes))

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return (self.classifier(context), weights)

def build_model(num_classes: int=None, **kwargs) -> CrimeLSTM:
    if num_classes is not None:
        kwargs['num_classes'] = num_classes
    return CrimeLSTM(**kwargs)
if __name__ == '__main__':
    from config import SEQ_LEN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = build_model().to(device)
    total = sum((p.numel() for p in m.parameters()))
    print(f'Parameters  : {total:,}')
    dummy = torch.randn(4, SEQ_LEN, CNN_FEATURE_DIM).to(device)
    logits, attn = m(dummy)
    print(f'Output shape: {logits.shape}   Attn: {attn.shape}')