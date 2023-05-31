import torch
from torch import nn

from onsets_and_frames.lstm import BiLSTM


class OnsetStack(nn.Module):
    def __init__(self, ConvStack, input_size, model_size, output_features, use_lstm=True, use_encoder=False) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.use_encoder = use_encoder
        self.conv = ConvStack(input_size, model_size)
        if self.use_lstm:
            self.rnn = BiLSTM(model_size, model_size // 2)

        fc_input = model_size
        if self.use_encoder:
            fc_input = model_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input, output_features),
            nn.Sigmoid(),
        )

    def forward(self, x, encoding=None):
        x = self.conv(x)
        if self.use_lstm:
            x = self.rnn(x)
        if self.use_encoder:
            encoding = encoding.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, encoding], dim=-1)
        x = self.fc(x)
        return x
