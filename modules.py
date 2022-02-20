import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers import LinearNorm
from CoordConv import CoordConv2d
from utils import get_mask_from_lengths


class TextSideProsodyEncoder(nn.Module):
    '''
    embedded_text --- [N, seq_len, encoder_embedding_dim]
    mels --- [N, n_mels*r, Ty/r], r=1
    style_embed --- [N, seq_len, prosody_embedding_dim]
    alignments --- [N, seq_len, ref_len], Ty/r = ref_len
    '''
    def __init__(self, hparams):
        super(TextSideProsodyEncoder, self).__init__()
        self.prosody_embedding_dim = hparams.prosody_embedding_dim
        self.encoder = ReferenceEncoder(hparams)
        self.ref_attn = ScaledDotProductAttention(hparams)
        self.encoder_bottleneck = nn.Linear(hparams.ref_enc_gru_size, hparams.prosody_embedding_dim * 2)

    def forward(self, embedded_text, text_lengths, mels, mels_lengths):
        embedded_prosody, _ = self.encoder(mels)

        # Bottleneck
        embedded_prosody = self.encoder_bottleneck(embedded_prosody)

        # Obtain k and v from prosody embedding
        key, value = torch.split(embedded_prosody, self.prosody_embedding_dim, dim=-1) # [N, Ty, prosody_embedding_dim] * 2

        # Get attention mask
        text_mask = get_mask_from_lengths(text_lengths).float().unsqueeze(-1) # [B, seq_len, 1]
        mels_mask = get_mask_from_lengths(mels_lengths).float().unsqueeze(-1) # [B, req_len, 1]
        attn_mask = torch.bmm(text_mask, mels_mask.transpose(-2, -1)) # [N, seq_len, ref_len]

        # Attention
        style_embed, alignments = self.ref_attn(embedded_text, key, value, attn_mask)

        # Apply ReLU as the activation function to force the values of the prosody embedding to lie in [0, ∞].
        style_embed = F.relu(style_embed)

        return style_embed, alignments


class ScaledDotProductAttention(nn.Module): # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams.ref_attention_dropout)
        self.d_q = hparams.encoder_embedding_dim
        self.d_k = hparams.prosody_embedding_dim
        self.linears = nn.ModuleList([
            LinearNorm(in_dim, hparams.ref_attention_dim, bias=False, w_init_gain='tanh') \
                for in_dim in (self.d_q, self.d_k)
        ])
        self.score_mask_value = -1e9

    def forward(self, q, k, v, mask=None):
        q, k = [linear(vector) for linear, vector in zip(self.linears, (q, k))]

        alignment = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # [N, seq_len, ref_len]

        if mask is not None:
            alignment.data.masked_fill_(mask == 0, self.score_mask_value)

        attention_weights = self.dropout(F.softmax(alignment, dim=-1))
        attention_context = torch.bmm(attention_weights, v) # [N, seq_len, prosody_embedding_dim]

        return attention_context, attention_weights


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, n_mels*r, Ty/r], r=1
    memory --- [N, Ty, prosody_embedding_dim * 2]
    out --- [1, N, prosody_embedding_dim * 2]
    '''

    def __init__(self, hparams):
        super(ReferenceEncoder, self).__init__()
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(in_channels=filters[0],
                           out_channels=filters[0 + 1],
                           kernel_size=(3, 3),
                           stride=(1, 2),
                           padding=(1, 1), with_r=True)]
        convs2 = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(1, 2),
                           padding=(1, 1)) for i in range(1,K)]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.ref_enc_gru_size,
                          batch_first=True)
        self.n_mels = hparams.n_mel_channels

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty, 128*n_mels//2^K]

        memory, out = self.gru(out)  # memory --- [N, Ty, ref_enc_gru_size], out --- [1, N, ref_enc_gru_size]

        return memory, out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L