import math

import numpy as np
import pywt
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm import Mamba


from pytorch_wavelets import DWTForward, DWT2D , DWTInverse




class SA_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SA_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention   ,attention



class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) ,y


class SpeMamba(nn.Module):
    def __init__(self, channels, token_num, group_num,use_residual):
        super(SpeMamba, self).__init__()

        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
        )


    def forward(self, x):

        x_pad = x
        x_pad = self.proj(x_pad)

        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape


        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)

        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C_pad)

        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        if self.use_residual:
            return x + x_recon

        else:
            return x_recon



n = 0

class WTCMC(nn.Module):
    def __init__(self,*,num_classes , inputsize, inputdim, k=30 ,dim = 64, num_tokens=81, depth = 2,heads = 10,dim_head = 10,mlp_dim = 512,dropout = 0.1,emb_dropout = 0.1,):
        super().__init__()

        self.dim = dim
        self.num_size=inputsize - 4
        self.inputsize = inputsize
        self.inputdim = inputdim
        self.k=k

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(5),
            nn.PReLU(num_parameters=1, init=0.01)
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=5*(inputdim-2), out_channels=dim, kernel_size=(3, 3)),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=1, init=0.01)
        )

        self.flatten = nn.Flatten(1, 2)

        self.spemamba = SpeMamba(
            channels=dim,
            token_num=4,
            group_num=4,
            use_residual=True,
        )


        self.dwt = DWT2D(J=1,wave="haar",mode="symmetric")

        self.conv_gaopin =  nn.Sequential(
            nn.Conv2d(in_channels=3 * dim, out_channels=dim, kernel_size=1),
            nn.PReLU(num_parameters=1, init=0.01)
        )

        self.conv_cH = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cH1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cH2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cV = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cV1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cV2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cD = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cD1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )
        self.conv_cD2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.PReLU(num_parameters=dim, init=0.01)
        )


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.num_size, stride=1)

        self.se = SE_Block(dim)

        self.sa_1 = SA_Block(dim)
        self.sa_2 = SA_Block(dim)
        self.sa_3 = SA_Block(dim)

        self.printed = False


    def forward(self, x,labels):

        if not hasattr(self, 'n'):
            self.n = 0

        m = self.n + 1
        if m == 1 and not self.printed:
            self.n = 3
            self.printed = True
            print("此模型为WTCMC")


        x_conv3d = self.conv3d(x)
        x_faltten = self.flatten(x_conv3d)
        x_conv2d = self.conv2d(x_faltten)


        x_duo=x_conv2d
        x_duo = F.interpolate(x_duo, scale_factor=2, mode='bilinear', align_corners=False)


        coeffs = self.dwt(x_duo)
        cA = coeffs[0]

        yh = coeffs[1]
        if len(yh) > 0:
            cH, cV, cD = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :], yh[0][:, :, 2, :, :]
        else:
            cH, cV, cD = torch.zeros_like(cA), torch.zeros_like(cA), torch.zeros_like(cA)
            print("小波变换错误")


        x_mamba_spe = cA
        x_mamba_spe = self.spemamba(x_mamba_spe)
        x_mamba_spa ,attention_mamba= self.se(x_mamba_spe)


        x_cnn2 = self.conv_cH(cH)
        x_cnn3 = self.conv_cH1(cH)
        x_cnn4 = self.conv_cH2(cH)
        x_cH = x_cnn2 + x_cnn3 + x_cnn4

        x_cnn7 = self.conv_cV(cV)
        x_cnn8 = self.conv_cV1(cV)
        x_cnn9 = self.conv_cV2(cV)
        x_cV = x_cnn7 + x_cnn8  + x_cnn9

        x_cnn12 = self.conv_cD(cD)
        x_cnn13 = self.conv_cD1(cD)
        x_cnn14 = self.conv_cD2(cD)
        x_cD = x_cnn12 + x_cnn13 + x_cnn14

        x_cH ,attention_4= self.sa_1(x_cH)
        x_cV ,attention_9= self.sa_2(x_cV)
        x_cD ,attention_14=self.sa_3(x_cD)

        x_gaopin = torch.cat([ x_cH, x_cV, x_cD], dim=1)
        x_gaopin = self.conv_gaopin(x_gaopin)
        x_gaopin = x_gaopin * attention_mamba


        x_wavelet = x_mamba_spa + x_gaopin

        x_reconstructed=self.avg_pool(x_wavelet)
        x_reconstructed = x_reconstructed.squeeze()
        x_class = self.mlp_head(x_reconstructed)


        return x_class


if __name__ == '__main__':
    inputs = torch.randn(64, 1, 130, 13, 13).cuda()
    labels = torch.randn(64).cuda()

    model = WTCMC(num_classes=16 , inputsize=13, inputdim=130).cuda()


    logits= model(inputs,labels)

    print(f"MassFormer模型输出形状: {logits.shape}")
