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
# from Zoe.zoedepth.models.builder import build_model
# from Zoe.zoedepth.utils.config import get_config
import torch
from models.unrt_aspp import*

from diffusers import AutoencoderKL

# from Zoe.zoedepth.utils.misc import pil_to_batched_tensor

# class ModifiedZoeDepth(nn.Module):
#     def __init__(self, original_model, latent_dim):
#         super(ModifiedZoeDepth, self).__init__()
#         self.backbone = original_model
        
#         # 添加额外的层来将深度特征转换为指导向量
#         self.guidance_layers = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(128, latent_dim)
#         )

#     def forward(self, x):
#         # 获取 ZoeDepth 的中间特征，而不是最终的深度图
#         features = self.backbone.core.core_out(self.backbone.core.get_encoder_features(x))
        
#         # 将特征转换为指导向量
#         guidance_vector = self.guidance_layers(features)
#         return guidance_vector

# class TeacherWrapper:
#     def __init__(self, model_type="zoedepth_nk", latent_dim=256):
  

#         model_zoe_n = torch.hub.load("./Zoe", "ZoeD_N", source="local", pretrained=True)

        
#         self.model = ModifiedZoeDepth(model_zoe_n, latent_dim)
#         self.model.eval()

#     @torch.no_grad()
#     def process(self, rgb_image):
#         # 直接获取指导向量
#         guidance_vector = self.model(rgb_image)
#         return guidance_vector





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

        self.unet_aspp=UNet()

        # self.encoder_vq = sd_model.first_stage_model

        # self.teacher_wrapper = TeacherWrapper(latent_dim=256) 
        # self.vae = AutoencoderKL.from_pretrained("/home/yinjun/project/test_ecoNet/sd-vae-ft-mse",local_files_only=True)
        # self.pre_vae = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
        # for param in self.vae.parameters():
        #   param.requires_grad = False
    
        # del sd_model.cond_stage_model
        # del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        # for param in self.encoder_vq.parameters():
        #     param.requires_grad = False
      


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
 
    
       
  
        # latents_spec=self.fpn(audio_spec)
        # latents=self.unet_aspp(audio_spec)
      
        # guidance_vector = self.teacher_wrapper.process(rgb)
        # print("gui",guidance_vector.shape)
        # audio_spec=self.pre_vae(audio_spec)
        # latents = self.vae.encode(audio_spec).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor
        latents=self.unet_aspp(audio_spec)
        
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
        self.max_depth = 14.104


      
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

        out_depth = torch.sigmoid(out_depth) * self.max_depth

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



# if __name__ == "__main__":

#     device = torch.device("cpu")


 
#     inputs_rgb = torch.randn((4, 2, 128, 128)).to(device)
  
#     model = EcoDepth()

#     pred= model(inputs_rgb)
    
