# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import sys
import os
# 将项目根目录加入Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
# from data.dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
from abc import abstractmethod

# 图像预处理封装
# datamgr.py 修改部分

class TransformLoader:
    def __init__(self, image_size, repeat_num=1):
        # 修改：BUSI超声图像的归一化参数（建议重新计算）
        self.normalize_param = dict(
            mean=[0.485, 0.456, 0.406],  # ImageNet标准值，可以先使用
            std=[0.229, 0.224, 0.225]  # 后续可以根据数据集重新计算
        )
        self.repeat_num = repeat_num
        self.image_size = image_size

        # 根据图像尺寸调整resize_size
        if image_size == 84:
            self.resize_size = 92
        elif image_size == 224:
            self.resize_size = 256
        elif image_size == 128:  # 如果使用128x128
            self.resize_size = 146

    

    # 组合变换：动态生成对应的图像预处理
# 组合变换：动态生成对应的图像预处理
    def get_composed_transform(self, aug=False):
        # 根据repeat_num样本重复次数和aug（是否增强）
        if self.repeat_num == 1 or aug:
            if aug:
                transform = transforms.Compose([
                    # 随机裁剪+缩放：从原图随机区域裁剪，再缩放到image_size
                    transforms.RandomResizedCrop(self.image_size),
                    # 随机水平翻转
                    transforms.RandomHorizontalFlip(),
                    # 颜色抖动
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    # 方案1：固定高斯滤波（对所有图像去噪）
                    # transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 0.5)),
                    
                    # ===== 或者方案2：概率性高斯滤波（推荐，保留部分原始信息）=====
                    # transforms.RandomApply([
                    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))
                    # ], p=0.5),  # 50%概率应用高斯滤波
                    


                    # 格式转换：PIL图像(HWC, 0-255) → PyTorch张量(CHW, 0-1)
                    transforms.ToTensor(),
                    # 按预定义均值/标准差归一化
                    transforms.Normalize(**self.normalize_param)
                ])
            else:
                transform = transforms.Compose([
                    # 固定缩放到resize_size
                    transforms.Resize(self.resize_size),
                    # 中心裁剪到目标尺寸
                    transforms.CenterCrop(self.image_size),
                    # 格式转换
                    transforms.ToTensor(),
                    # 归一化
                    transforms.Normalize(**self.normalize_param)
                ])
        else:
            transform = transforms.Compose([
                # 限定缩放范围的随机裁剪
                transforms.RandomResizedCrop(self.image_size, scale=(0.14, 1)),
                # transforms.RandomResizedCrop(self.image_size, scale=(0.48, 1)),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])

        return transform

   



# 定义抽象数据管理器类，统一数据加载接口
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

# 普通批次数据加载器
# 继承DataManager，实现 “普通批次采样”的加载逻辑
class SimpleDataManager(DataManager):
    # 图像数据根路径、目标图像尺寸、批次大小、是否读取 JSON 格式的标注文件
    def __init__(self, data_path, image_size, batch_size, json_read=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.data_path = r'D:\PyCharmprogects\data\BUSI'
        # 初始化图像预处理加载器
        self.trans_loader = TransformLoader(image_size)
        self.json_read = json_read

    # 根据aug获取预处理流水线；根据json_read选择数据集类
    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        # 获取预处理流水线
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            # JSON标注数据集
            dataset = SimpleDataset_JSON(self.data_path, data_file, transform)
        else:
            # 普通标注数据集
            dataset = SimpleDataset(self.data_path, data_file, transform)
            # 导入平台检测模块
        import platform
        # 获取操作系统类型
        sys = platform.system()
        if sys == "Windows":
            # Windows下关闭多线程
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        else:
            # Linux/macOS下启用6个工作线程
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True)

        # dataset = torch.utils.data.distributed.DistributedSampler(dataset)
        # 创建数据加载器
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class SetDataManager(DataManager):
    def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode, json_read=False, repeat_num=1):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        # 每个episode的类别数
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.data_path = data_path
        self.json_read = json_read
        self.repeat_num = repeat_num
        self.trans_loader = TransformLoader(image_size, repeat_num=repeat_num)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            dataset = SetDataset_JSON(self.data_path, data_file, self.batch_size, transform, repeat_num=self.repeat_num)
        else:
            dataset = SetDataset(self.data_path, data_file, self.batch_size, transform, repeat_num=self.repeat_num)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)

        import platform

        sys = platform.system()
        if sys == "Windows":
            data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)
        else:
            data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=True)

        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

# class SetDataManager(DataManager):
#     def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode,
#                  json_read=False, repeat_num=1, dataset_type='default'):  # 添加dataset_type参数
#         super(SetDataManager, self).__init__()
#         self.image_size = image_size
#         self.n_way = n_way
#         self.batch_size = n_support + n_query
#         self.n_episode = n_episode
#         self.data_path = data_path
#         self.json_read = json_read
#         self.repeat_num = repeat_num
#         self.dataset_type = dataset_type  # 保存数据集类型

#         # 根据数据集类型选择不同的TransformLoader
#         if dataset_type == 'busi':
#             self.trans_loader = BUSITransformLoader(image_size, repeat_num=repeat_num)
#         else:
#             self.trans_loader = TransformLoader(image_size, repeat_num=repeat_num)

#     def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
#         transform = self.trans_loader.get_composed_transform(aug)
#         # ... 其余代码保持不变 ...
