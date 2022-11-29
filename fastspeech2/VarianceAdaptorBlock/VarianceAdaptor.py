import numpy as np
import torch
from torch import nn

from fastspeech2.VarianceAdaptorBlock.DurationPredictor import DurationPredictor
from torch.nn import functional as F

from fastspeech2.VarianceAdaptorBlock.utilities import create_alignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = DurationPredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = DurationPredictor()
        self.energy_predictor = DurationPredictor()
        n_bins = 256
        pitch_min, pitch_max = 0.00001, 759.0302321788763
        energy_min, energy_max = 0.01786651276051998, 314.9619140625
        self.pitch_ohe = nn.Embedding(n_bins, 256)
        self.energy_ohe = nn.Embedding(n_bins, 256)
        self.pitch_quant = nn.Parameter(torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False)
        self.energy_quant = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False)

    def pitch_vectorize(self, x, target, pitch_control):
        pitch_pred = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_ohe(torch.bucketize(target, self.pitch_quant))
        else:
            pitch_pred *= pitch_control
            embedding = self.pitch_ohe(torch.bucketize(pitch_pred, self.pitch_quant))
        return pitch_pred, embedding

    def energy_vectorize(self, x, target, energy_control):
        # print(target.shape, 'target_shape')
        energy_pred = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_ohe(torch.bucketize(target, self.energy_quant))
        else:
            energy_pred *= energy_control
            embedding = self.energy_ohe(torch.bucketize(energy_pred, self.energy_quant) )

        return energy_pred, embedding

    def forward(
            self,
            x,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            duration_target=None,
            p_c=1.0,
            e_c=1.0,
            d_c=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x)

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            dur_unlogged = duration_target
            # print(mel_len.shape, 'mel_len')
        else:
            dur_unlogged = torch.clamp((torch.round(torch.exp(log_duration_prediction) - 1) * d_c), min=0)
            # print(duration_rounded, 'ldp')
            x, mel_len = self.length_regulator(x, log_duration_prediction, max_len)
            # print(mel_len.shape, 'mel_len')


        # print(x.shape, 'after_dur_pred')

        # print(mel_mask.shape, 'mel_mask')

        pitch_p, pitch_emb = self.pitch_vectorize(x, pitch_target, p_c)
        # print(pitch_embedding.shape)
        x = x + pitch_emb

        # print(x.shape, pitch_embedding.shape, 'after_pitcj_embd')

        en_p, energy_emb = self.energy_vectorize(x, energy_target, p_c)
        # print(x.shape, energy_embedding.shape, 'after_energy_embd')
        x = x + energy_emb

        return (
            x,
            pitch_p,
            en_p,
            log_duration_prediction,
            dur_unlogged,
            mel_len,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration_predictor_output, mel_max_length=None):

        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, duration_predictor_output, mel_max_length=None):

        if mel_max_length is not None:
            output = self.LR(x, duration_predictor_output, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (
                    (duration_predictor_output + 0.5) * 1).int()
            output = self.LR(x, duration_predictor_output)

            length_mel = np.array(list())
            for mel in output:
                length_mel = np.append(length_mel, mel.size(0))
            mel_pos = list()
            max_mel_len = int(max(length_mel))
            for length_mel_row in length_mel:
                mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                                      (0, max_mel_len - int(length_mel_row)), 'constant'))
            mel_pos = torch.from_numpy(np.array(mel_pos)).to(device)
            return output, mel_pos
