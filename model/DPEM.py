import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding

from mamba_ssm import Mamba


import torch.fft
from layers.Graph_Block import GraphBlock
from layers.Conv_Blocks import Inception_Block_V1
from layers.Cross_attention import CrossAttention
import torch.nn.functional as F


def FFT_for_Period(x, k=3):
    # [B, T, C]
    # x = x.permute(0, 2, 1)
    x = x.float()
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]




class Model(nn.Module):
   

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

 
        self.x_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.class_strategy = configs.class_strategy
        # Mamba Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

   
        self.gconv = GraphBlock(configs.enc_in, configs.d_model, 32, 32, 2, 0.1, 3, 32)  
     
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, 32, num_kernels=3),
            nn.GELU(),
            Inception_Block_V1(32, configs.d_model, num_kernels=3))
        self.cat_linear = nn.Linear(configs.d_model + configs.seq_len, configs.d_model, bias=True)
        #self.cat_linear1 = nn.Linear(configs.d_model , configs.d_model, bias=True)
        self.c_linear = nn.Linear(configs.d_model , configs.enc_in, bias=True)

        
        self.cross_attention1 = CrossAttention(d_model = configs.d_model, n_heads=8, d_k=None, d_v=None, attn_dropout=0,
                                               proj_dropout=0, res_attention=0)
        self.cross_attention2 = CrossAttention(d_model = configs.d_model, n_heads=8, d_k=None, d_v=None, attn_dropout=0,
                                               proj_dropout=0, res_attention=0)
        self.k = configs.top_k


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N_value = x_enc.shape 
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        enc_x = self.x_embedding(x_enc, None) 

        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
    

        B, T, N = enc_x.size()
        scale_list, scale_weight = FFT_for_Period(enc_x, self.k)
        # MSCM
        res_f_m = []
        for i in range(self.k):
            scale = scale_list[i]  # 
            # paddng
            if (T) % scale != 0:
                length = (((T) // scale) + 1) * scale
                padding = torch.zeros([enc_x.shape[0], (length - (T)), enc_x.shape[2]]).to(enc_x.device)
                out = torch.cat([enc_x, padding], dim=1)
            else:
                length = T
                out = enc_x
            out = out.reshape(B, length // scale, scale,
                              N).permute(0, 3, 1, 2).contiguous()
            out_m = self.conv(out) 
            out_m = out_m.permute(0, 2, 3, 1).reshape(B, -1, N)
            out_m = out_m[:, :T, :]
            out_m,a1 = self.encoder(out_m, attn_mask=None)
            res_f_m.append(out_m)
        res_f_m = torch.stack(res_f_m, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res_f_m = res_f_m * scale_weight
        out_c_m = torch.sum(res_f_m * scale_weight, -1)  + enc_x
   
        # AGCM
        out_g = self.gconv(enc_out) 
        out_g_m,a2 = self.encoder1(out_g, attn_mask=None) 
        out_g_m = out_g_m + enc_out 
        
        out_g_cross_out_c, c = self.cross_attention1(out_g_m, out_c_m, out_c_m, key_padding_mask=None, attn_mask=None)
        out_g_cross_out_c = out_g_cross_out_c + out_g_m
        
        out_c_cross_out_g, d = self.cross_attention2(out_c_m, out_g_m, out_g_m, key_padding_mask=None, attn_mask=None)
        out_c_cross_out_g = out_c_cross_out_g + out_c_m
        out_c_cross_out_g = self.c_linear(out_c_cross_out_g)
        out_c_cross_out_g = out_c_cross_out_g.permute(0, 2, 1)


        dec_out = torch.cat([out_g_cross_out_c,out_c_cross_out_g], -1)

        dec_out = self.cat_linear(dec_out)
        dec_out = self.projector(dec_out).permute(0, 2, 1)[:, :, :N_value] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
