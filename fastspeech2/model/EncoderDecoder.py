import torch
from torch import nn

from TTS.fastspeech2.model.transformer import FFTBlock
from TTS.fastspeech2.text import symbols

from TTS.fastspeech2.configs import model_config


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(model_config.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        len_max_seq = 1000
        n_position = len_max_seq + 1
        n_layers = 4

        self.src_word_emb = nn.Embedding(
            len(symbols) + 1,
            256,
            padding_idx=0
        )

        self.position_enc = nn.Embedding(
            n_position,
            256,
            padding_idx=0
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            256,
            1024,
            2,
            256 // 2,
            256 // 2,
            [9, 1],
            dropout=0.5
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # print(src_seq.shape, src_pos.shape)

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward

        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        # print(enc_output.shape)

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self):

        super(Decoder, self).__init__()

        len_max_seq = 1000
        n_position = len_max_seq + 1
        n_layers = 4

        self.position_enc = nn.Embedding(
            n_position,
            256,
            padding_idx=0,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            256,
            1024,
            2,
            256 // 2,
            256 // 2,
            [9, 1],
            dropout=0.5
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
