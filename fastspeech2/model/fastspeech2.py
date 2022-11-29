import torch
from torch import nn

from fastspeech2.VarianceAdaptorBlock.VarianceAdaptor import VarianceAdaptor
from fastspeech2.model.EncoderDecoder import Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(256, 80)

    def forward(self, src_seq, src_len, max_src_len, mel_target=None, mel_len=None, max_mel_len=None, d_target=None,
                p_target=None, e_target=None, d_control=1.0, p_control=1.0, e_control=1.0):

        encoder_output, non_pad_mask = self.encoder(src_seq, src_len)

        if mel_target is not None:
            variance_adaptor_output, p_prediction, e_prediction, d_prediction, d_rounded, _ = self.variance_adaptor(
                encoder_output, max_mel_len, p_target, e_target, d_target, d_control, p_control, e_control)

            decoder_output = self.decoder(variance_adaptor_output, mel_len)


        else:
            variance_adaptor_output, p_prediction, e_prediction, d_prediction, d_rounded, mel_len = self.variance_adaptor(
                encoder_output, max_mel_len, p_target, e_target, d_target, d_control, p_control, e_control)
            decoder_output = self.decoder(variance_adaptor_output, mel_len)

        mel_output = self.mel_linear(decoder_output)

        return mel_output, d_prediction, p_prediction, e_prediction
