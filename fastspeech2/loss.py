import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, mel_target, energy, energy_target, duration, duration_target, pitch, pitch_target):
        mel_loss = self.mse_loss(mel, mel_target)

        p_loss = self.l1_loss(pitch, pitch_target.float())
        e_loss = self.l1_loss(energy, energy_target.float())
        d_loss = self.l1_loss(duration, duration_target.float())

        return mel_loss, p_loss, e_loss, d_loss
