import sys
import os

from torchvision import models

# 方法1：添加项目根目录到Python路径
# 获取当前文件（resnet.py）的绝对路径
current_file = os.path.abspath(__file__)
# 获取network目录
network_dir = os.path.dirname(current_file)
# 获取项目根目录（network的上级目录）
project_root = os.path.dirname(network_dir)
# 添加到Python路径
sys.path.insert(0, project_root)

# 现在导入utils
try:
    import utils

    print("成功导入utils模块！")
except ImportError as e:
    print(f"导入utils失败: {e}")
    print(f"Python路径: {sys.path}")
    print(f"项目根目录: {project_root}")
    # 检查utils.py是否存在
    utils_path = os.path.join(project_root, "utils.py")
    print(f"utils.py路径: {utils_path}")
    print(f"utils.py是否存在: {os.path.exists(utils_path)}")

# from utils import *
# import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def linear(indim, outdim):
    return nn.Linear(indim, outdim)

import sys
import os

from torchvision import models

# 方法1：添加项目根目录到Python路径
# 获取当前文件（resnet.py）的绝对路径
current_file = os.path.abspath(__file__)
# 获取network目录
network_dir = os.path.dirname(current_file)
# 获取项目根目录（network的上级目录）
project_root = os.path.dirname(network_dir)
# 添加到Python路径
sys.path.insert(0, project_root)

# 现在导入utils
try:
    import utils

    print("成功导入utils模块！")
except ImportError as e:
    print(f"导入utils失败: {e}")
    print(f"Python路径: {sys.path}")
    print(f"项目根目录: {project_root}")
    # 检查utils.py是否存在
    utils_path = os.path.join(project_root, "utils.py")
    print(f"utils.py路径: {utils_path}")
    print(f"utils.py是否存在: {os.path.exists(utils_path)}")

# from utils import *
# import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def linear(indim, outdim):
    return nn.Linear(indim, outdim)


# class WaveletDenoise(nn.Module):
#     def __init__(self, wavelet='db1', level=1):
#         super().__init__()
#         self.wavelet = wavelet
#         self.level = level
#         self.threshold = nn.Parameter(torch.tensor(0.1))  # 可学习阈值

#     def forward(self, x):
#         # x: (B, C, H, W)  对每个通道独立进行DWT
#         coeffs = pywt.wavedec2(x.detach().cpu().numpy(), self.wavelet, level=self.level)
#         # 简化：仅演示逻辑，实际需在GPU上实现可微DWT（可使用pytorch_wavelets库）
#         # 对高频系数应用软阈值
#         coeffs_th = [coeffs[0]]  # 保留低频
#         for detail in coeffs[1:]:
#             coeffs_th.append(tuple(pywt.threshold(c, self.threshold.item(), mode='soft') for c in detail))
#         x_denoised = pywt.waverec2(coeffs_th, self.wavelet)
#         return torch.tensor(x_denoised, device=x.device)


# class LearnableDenoise(nn.Module):
#     def __init__(self, in_channels, kernel_size=3, sigma=1.7):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
#                                padding=kernel_size//2, groups=in_channels, bias=False)
#         # 初始化高斯核
#         with torch.no_grad():
#             kernel = self._gaussian_kernel(kernel_size, sigma)
#             # 复制到每个输入通道
#             self.conv.weight.data = kernel.repeat(in_channels, 1, 1, 1)
#         self.conv.weight.requires_grad = True  # 允许微调

#     def _gaussian_kernel(self, size, sigma):
#         coords = torch.arange(size) - size//2
#         g = torch.exp(-(coords**2) / (2*sigma**2))
#         kernel = g[:, None] * g[None, :]  # 外积
#         kernel = kernel / kernel.sum()
#         return kernel.view(1, 1, size, size)





#     def forward(self, x):
#         return self.conv(x)



class ResNet18(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, rotations, use_prior,dropout_rate=0.3):
        super(ResNet18, self).__init__()

        self.use_prior = use_prior
        self.rotations = rotations

        # ✅ 预训练 backbone
        resnet = models.resnet18(pretrained=True)

        # model = torchvision.models.resnet18()
        # weights = torch.load('resnet18_224.pth')
        # model.load_state_dict(weights)

        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = 512
        self.dropout = nn.Dropout(dropout_rate)




        self.linear = linear(self.feature_dim, num_classes)
        if rotations:
            self.linear_rot = linear(self.feature_dim, 4)

        if use_prior:
            self.prior = Prior(self.feature_dim)

    def forward(self, x, index_mixup=None, lam=-1, return_map=False):

        # ✅ 只在输入做 mixup（推荐）
        if lam != -1:
            x = lam * x + (1 - lam) * x[index_mixup]

        # ✅ backbone
        out = self.backbone(x)  # [B,512,1,1]
        

        features = out.view(out.size(0), -1)

        # ✅ Prior
        if self.use_prior:
            features, context_prior_map = self.prior(features)
        else:
            context_prior_map = None

        # ✅ 分类
        out = self.linear(features)

        if return_map:
            if self.rotations:
                out_rot = self.linear_rot(features)
                return (out, out_rot, context_prior_map), features
            return (out, context_prior_map), features
        else:
            if self.rotations:
                out_rot = self.linear_rot(features)
                return (out, out_rot), features
            return out, features


# class ResNet18(nn.Module):
#     def __init__(self, feature_maps, input_shape, num_classes, rotations, use_prior, dropout_rate=0.3, use_denoise=True,sim_type='l2'):
#         super(ResNet18, self).__init__()

#         self.use_prior = use_prior
#         self.rotations = rotations
#         self.use_denoise = use_denoise

#         # backbone: 去掉最后的 avgpool 和 fc，保留空间维度用于降噪
#         resnet = models.resnet18(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉 avgpool 和 fc

#         # 获取输出通道数
#         with torch.no_grad():
#             dummy = torch.zeros(1, *input_shape)
#             out_channels = self.backbone(dummy).shape[1]

#         # 降噪模块
#         if use_denoise:
#             self.denoise = LearnableDenoise(out_channels, kernel_size=3, sigma=1.5)
#             print("✅ 降噪模块已创建")
#         else:
#             self.denoise = nn.Identity()
#             print("❌ 降噪模块未启用")

#         # 全局池化（降噪之后使用）
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.feature_dim = out_channels
#         self.dropout = nn.Dropout(dropout_rate)

#         self.linear = linear(self.feature_dim, num_classes)
#         if rotations:
#             self.linear_rot = linear(self.feature_dim, 4)

#         if use_prior:
#             # self.prior = Prior(self.feature_dim)
#             self.prior = Prior(self.feature_dim, sim_type=sim_type)

#     def forward(self, x, index_mixup=None, lam=-1, return_map=False):
#         if lam != -1:
#             x = lam * x + (1 - lam) * x[index_mixup]

#         # 特征提取
#         out = self.backbone(x)  # [B, 512, H, W]
        
#         # 降噪（特征提取之后，关系建模之前）
#         out = self.denoise(out)
        
#         # 全局池化
#         out = self.global_pool(out)
#         features = out.view(out.size(0), -1)

#         # Prior
#         if self.use_prior:
#             features, context_prior_map = self.prior(features)
#         else:
#             context_prior_map = None

#         out_class = self.linear(features)

#         if return_map:
#             if self.rotations:
#                 out_rot = self.linear_rot(features)
#                 return (out_class, out_rot, context_prior_map), features
#             else:
#                 # 关键修改：返回3个值，out_rot 为 None
#                 return (out_class, None, context_prior_map), features
#         else:
#             if self.rotations:
#                 out_rot = self.linear_rot(features)
#                 return (out_class, out_rot), features
#             return out_class, features


def ResNet18Prior(feature_maps=64, input_shape=(3, 84, 84), num_classes=64, rotations=True, use_prior=True):
    return ResNet18(
        feature_maps=feature_maps,
        input_shape=input_shape,
        num_classes=num_classes,
        rotations=rotations,
        use_prior=use_prior,
        # use_denoise=use_denoise
    )


# def ResNet18Prior(feature_maps=64, input_shape=(3, 84, 84), num_classes=64, rotations=True, use_prior=True,use_denoise=True):
#     """工厂函数，保持与ResNet12类似的接口"""
#     return ResNet18(
#         feature_maps=feature_maps,
#         input_shape=input_shape,
#         num_classes=num_classes,
#         rotations=rotations,
#         use_prior=use_prior,
#         use_denoise=use_denoise
#     )



class Prior(nn.Module):
    def __init__(self, in_planes):
        super(Prior, self).__init__()
        self.reduce_dim = in_planes

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layernorm1 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)
        self.layernorm2 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)

        self.fc_q = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.fc_k = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.fc_v = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.softmax = nn.Softmax(dim=1)

        # 融合层
        self.fusion = nn.Linear(self.reduce_dim * 2, self.reduce_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        Q, K, V = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        context_prior_map = torch.sigmoid(torch.matmul(Q, K.transpose(0, 1)) / self.reduce_dim ** 0.5)
        intra_context = torch.matmul(self.softmax(context_prior_map), V)

        # 先concat再线性变换回原维度
        concat_features = torch.cat([self.layernorm1(intra_context), self.layernorm2(x)], dim=1)
        total_context = self.fusion(concat_features)

        return total_context, context_prior_map



# class Prior(nn.Module):
#     def __init__(self, in_planes, sim_type='l2'):  # 添加 sim_type 参数，默认使用 l2
#         super(Prior, self).__init__()
#         self.reduce_dim = in_planes
#         self.sim_type = sim_type  # 'dot'(原始) 或 'l2'

#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.layernorm1 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)
#         self.layernorm2 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)

#         self.fc_q = nn.Linear(self.reduce_dim, self.reduce_dim)
#         self.fc_k = nn.Linear(self.reduce_dim, self.reduce_dim)
#         self.fc_v = nn.Linear(self.reduce_dim, self.reduce_dim)
#         self.softmax = nn.Softmax(dim=1)

#         # ===== 新增：可学习温度参数（用于 L2 距离）=====
#         self.temperature = nn.Parameter(torch.tensor(1.0))

#         # 融合层
#         self.fusion = nn.Linear(self.reduce_dim * 2, self.reduce_dim)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         Q, K, V = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        
#         # ===== 根据 sim_type 选择相似度计算方式 =====
#         if self.sim_type == 'dot':
#             # 原始点积相似度
#             context_prior_map = torch.sigmoid(torch.matmul(Q, K.transpose(0, 1)) / self.reduce_dim ** 0.5)
        
#         elif self.sim_type == 'l2':
#             # 负L2距离 + 可学习温度
#             dist2 = torch.cdist(Q, K, p=2) ** 2  # 计算成对欧氏距离平方
#             context_prior_map = torch.sigmoid(-dist2 / (self.temperature.abs() + 1e-8))
        
#         else:
#             # 默认使用原始方式
#             context_prior_map = torch.sigmoid(torch.matmul(Q, K.transpose(0, 1)) / self.reduce_dim ** 0.5)
        
#         intra_context = torch.matmul(self.softmax(context_prior_map), V)

#         # 先concat再线性变换回原维度
#         concat_features = torch.cat([self.layernorm1(intra_context), self.layernorm2(x)], dim=1)
#         total_context = self.fusion(concat_features)

#         return total_context, context_prior_map
   