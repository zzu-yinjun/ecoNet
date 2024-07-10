import torch.nn as nn
import torch
import random
import torchvision.models as models_resnet
from timm.models.layers import trunc_normal_, DropPath
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from models.functions import *
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
 
)
from omegaconf import OmegaConf
from transformers import Wav2Vec2Model
import torch.nn.functional as F
from models.models_eco import UNetWrapper, EmbeddingAdapter
import os


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net



class UNet_aspp(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 32,
                 gpu_ids=[]):
        super(UNet_aspp, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.aspp1 = ASPP(32,32)
        self.aspp2 = ASPP(64,64)
        self.aspp3 = ASPP(128,128)
        self.aspp4 = ASPP(256,256)
        # self.aspp1 = ASPP(64,64)
        # self.aspp2 = ASPP(128,128)
        # self.aspp3 = ASPP(256,256)
        # self.aspp4 = ASPP(512,512)

        # self.aspp4 = ASPP()

     
    def forward(self, x):
        # x = torch.transpose(x, 0, 1)
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x1 = self.in_conv(x)
  
        x1 = self.aspp1(x1)
        x2 = self.down1(x1)
   
        x2 = self.aspp2(x2)
        x3 = self.down2(x2)
    
        x3 = self.aspp3(x3)
        x4 = self.down3(x3)

        x4 = self.aspp4(x4)
        # x5 = self.down4(x4)
        print(x4.shape)
       

        return x4



class Upsample(nn.Module):  # this
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,  # this conv let the size unchanged
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest"
        )  # double the size
        if self.with_conv:
            x = self.conv(x)
        return x

class DownsampleFPN(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels,  # halves the size
                out_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(
                x, pad, mode="constant", value=0
            )  # 此動作相當於在每個圖片的channel的右邊下面pad 0
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class FPN(
    nn.Module
):  # 此處預設每次的resolutions都是上一次的一半 第一次的resolution是原圖的 1/4
    def __init__(self, config):
        super().__init__()
        # [64, 128, 256, 512]
        resolutions = config["model"]["FPN_conv_res"].copy()

        # resolutions = [64, 128, 256, 512]
        self.resolutions = resolutions.copy()

        # self.target_channel = int(resolutions[0] / 2)
        self.target_channel = config["model"]["FPN_target_C"]  # 256

        resolutions.insert(0, 2)
        # self.resolutions = resolutions # which is list
        self.ConvList = nn.ModuleList()
        self.tuneChannels = nn.ModuleList()
        self.Upsampple = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.1)
        # self.Lrelu = nonlinearity
        self.bn0 = nn.BatchNorm2d(resolutions[0])
        for idx in range(len(resolutions) - 1):
            self.ConvList.append(
                DownsampleFPN(resolutions[idx], resolutions[idx + 1], True)
            )

            self.tuneChannels.append(
                torch.nn.Conv2d(
                    resolutions[idx + 1],
                    self.target_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if idx != len(resolutions) - 2:
                self.Upsampple.append(Upsample(self.target_channel, True))

        self.convOut = torch.nn.Conv2d(
            self.target_channel, self.target_channel, kernel_size=1, stride=1, padding=0
        )
        self.norm_seq = nn.ModuleList()
        for idx in range((len(self.resolutions) - 2) * 2 + 2):

            self.norm_seq.append(nn.BatchNorm2d(self.target_channel))

    def forward(self, x):
        h = self.bn0(x)
        FPN_list = []

        for idx in range(len(self.resolutions)):

            if idx == 0:
                h = self.pool(self.Lrelu(self.ConvList[idx](h)))
            else:
                h = self.Lrelu(self.ConvList[idx](temp))

            temp = h
            h = self.Lrelu(self.tuneChannels[idx](h))
            FPN_list.append(h)
        count = 0
        for idx in reversed(range(len(self.resolutions))):
            if idx == 0:
                hold = self.norm_seq[count](hold)
                count += 1
                hold = self.convOut(hold + self.norm_seq[count](FPN_list[idx]))
                break
            if idx == len(self.resolutions) - 1:
                hold = self.Upsampple[idx - 1](FPN_list[idx])
            else:
                hold = self.norm_seq[count](hold)
                count += 1

                hold = hold + self.norm_seq[count](FPN_list[idx])
                count = count + 1
                hold = self.Upsampple[idx - 1](hold)

        return hold

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class EcoDepthEncoder(nn.Module):
    def __init__(
        self,
        out_dim=1024,
        ldm_prior=[32,64,256],
        sd_path=None,
        emb_dim=768,
        dataset="nyu",
        args=None,
    ):
        super().__init__()
 
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

        self.cide_module = CIDE(emb_dim)

        self.config = OmegaConf.load("/home/yinjun/project/test_ecoNet/conf/model/v1-inference.yaml")

       

        # 配置实例化模型
        # sd_model = LatentDiffusion(**self.config.get("params", dict()))
        sd_model = instantiate_from_config(self.config.model)

        # self.unet = UNetWrapper(sd_model.model, use_attn=False)
        self.unet = UNetWrapper( sd_model.model, use_attn=False)
        file_path = "./conf/config_diffusion.yml"

        config_fpn = load_config(file_path)
        self.fpn = FPN(config_fpn)    
        self.unet_aspp=UNet_aspp()
    
        del self.unet.unet.diffusion_model.out



    # 初始化神经网络模型的权重
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x = self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, audio_spec,audio_wave):
        # 在使用 torch.no_grad() 时防止对 VQ 编码器的梯度计算，因为它被冻结了

        latents=self.fpn(audio_spec)
        # latents=self.unet_aspp(audio_spec)

        # 生成一个表示条件场景嵌入的张量
        conditioning_scene_embedding = self.cide_module(audio_wave)  # 由vit得到
     

        # 这行代码创建了一个大小为x.shape[0]的张量t，该张量的值全部为1，数据类型为long，并且将其放置在与输入张量x相同的设备上。
        t = torch.ones((audio_spec.shape[0],), device=audio_spec.device).long()

        # latents是潜在空间表示 conditioning_scene_embedding是由vit得到的
        outs = self.unet(latents, t, c_crossattn=[conditioning_scene_embedding])
      
        # 创建了一个名为feats的列表，它包含了outs列表中的前三个元素。对于feats中的第三个元素，
        # 还使用了torch.cat函数将outs中的第三个元素和经过缩放的outs中的第四个元素进行合并
        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]
       
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)

        return self.out_layer(x)


class CIDE(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
 

        # 参数不需要梯度计算
        # self.resnet=models_resnet.resnet50(pretrained=False)
        # self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1000)
        # # 将输入维度从1000映射到400，再映射到args.no_of_classe
        # self.fc = nn.Sequential(
        #     nn.Linear(1000, 400), nn.GELU(), nn.Linear(400, 100)
        # )
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.freeze_feature_extractor()  # 冻结特征提取器
        self.conv = nn.Conv1d(2, 1, kernel_size=1)
        self.fc = nn.Sequential(nn.Linear(768, 400), nn.GELU(), nn.Linear(400, 100))
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        # 初始化了一个可学习的参数作为嵌入矩阵embeddings，表示用于条件图像生成的类嵌入
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
        # 创建了一个名为embedding_adapter的EmbeddingAdapter实例，其参数为emb_dim，可能用于连接类嵌入和条件场景嵌
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        # 创建了一个名为embedding_adapter的EmbeddingAdapter实例，其参数为emb_dim，可能用于连接类嵌入和条件场景嵌
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)

    # 输入张量转换为一个方形的张量，并在需要时进行填充


    def forward(self, x):

  
        # 止梯度通过ViT流动，因为它被保持冻结状态。通过ViT模型，将处理后的输入图像转换为logits
        # vit_logits=self.resnet(x)
        # # 使用线性层(fc)和softmax激活函数(m)将logits转换为类别概率。接着，将类别概率与类别嵌入矩阵(embeddings)相乘，得到类别嵌入表征。
        x = self.conv(x)  # 输出 shape: [64, 1, 2646]
        x = x.squeeze(1)  # 移除通道维度，shape: [64, 2646]
        
       
         # 确保输入长度足够
        mask_length = 10  # Wav2Vec2的默认mask_length
        min_input_length = self.wav2vec.config.inputs_to_logits_ratio * mask_length * 2
        if x.shape[1] < min_input_length:
            pad_length = min_input_length - x.shape[1]
            x = F.pad(x, (0, pad_length))

        # 使用Wav2Vec2模型
        with torch.no_grad():
            wav2vec_output = self.wav2vec(x).last_hidden_state  # shape: [64, 512, 19]

        # 使用平均池化来减少时间维度
        wav2vec_output = wav2vec_output.mean(dim=1)  # shape: [64, 768]
        class_probs = self.fc( wav2vec_output)
        class_probs = self.m(class_probs)

        # 计算类别概率（class_probs）与类别嵌入矩阵（embeddings）的矩阵乘法，得到类别嵌入表征（class_embeddings
        class_embeddings = class_probs @ self.embeddings

        conditioning_scene_embedding = self.embedding_adapter(
            class_embeddings, self.gamma
        )

        return conditioning_scene_embedding


# 论文模型
class EcoDepth(nn.Module):
    def __init__(self):
        super().__init__()
        # self.max_depth = 30


      
        # 嵌入向量的维度
        embed_dim = 192
        # 输入通道的数量
        channels_in = embed_dim * 8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset="nyu")
        self.decoder = Decoder(channels_in, channels_out)
       

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
        )

     

    def forward(self, audio_spec,audio_wave):

        conv_feats = self.encoder(audio_spec,audio_wave)
        out = self.decoder([conv_feats])
        out_depth = self.last_layer_depth(out)

        # out_depth = torch.sigmoid(out_depth) * self.max_depth

        return out_depth


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = 3
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            3,
            [32,32,32],
            [2,2,2],
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        conv_layers.append(build_norm_layer(dict(type="BN"), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, conv_feats):
        # 3次连续反卷积
        out = self.deconv_layers(conv_feats[0])
        # 卷积+bn+relu
        out = self.conv_layers(out)

        # 两次上采样
        out = self.up(out)
        # out = self.up(out)

        return out

    # 通过循环迭代创建了多个反卷积层 默认是三层
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type="deconv"),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    # 根据给定的反卷积核大小，返回相应的填充和输出填充配置，用于配置反卷积层的参数。
    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

        return deconv_kernel, padding, output_padding



if __name__ == "__main__":

    device = torch.device("cpu")


 
    inputs_rgb = torch.randn((4, 2, 128, 128)).to(device)
  
    model = EcoDepth()

    pred= model(inputs_rgb)
    
