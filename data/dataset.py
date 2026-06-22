
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

# 定义恒等函数
identity = lambda x: x

# 简单数据集的加载
# dataset.py

# dataset.py 修改部分

class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []

        data_dir_list = data_file_list.replace(" ", "").split(',')

        for data_file in data_dir_list:
            img_dir = os.path.join(data_path, data_file)

            for class_name in os.listdir(img_dir):
                class_path = os.path.join(img_dir, class_name)

                if os.path.isdir(class_path):
                    # 修改：将文件夹名映射为数字标签
                    if class_name.lower() == 'benign':
                        cls = 0
                    elif class_name.lower() == 'malignant':
                        cls = 1
                    elif class_name.lower() == 'normal':
                        cls = 2
                    else:
                        continue  # 跳过其他文件夹

                    for fname in os.listdir(class_path):
                        # 修改：支持png格式
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            full_path = os.path.join(class_path, fname)
                            data.append(full_path)
                            label.append(cls)

        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform


class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform, repeat_num=1):
        label = []
        data = []

        data_dir_list = data_file_list.replace(" ", "").split(',')

        for data_file in data_dir_list:
            img_dir = os.path.join(data_path, data_file)

            for class_name in os.listdir(img_dir):
                class_path = os.path.join(img_dir, class_name)

                if os.path.isdir(class_path):
                    if class_name.lower() == 'benign':
                        cls = 0
                    elif class_name.lower() == 'malignant':
                        cls = 1
                    elif class_name.lower() == 'normal':
                        cls = 2
                    else:
                        continue

                    for fname in os.listdir(class_path):
                        if fname.lower().endswith('.png'):
                            full_path = os.path.join(class_path, fname)
                            data.append(full_path)
                            label.append(cls)

        self.data = data
        self.label = label
        self.transform = transform
        self.cl_list = np.unique(self.label).tolist()

        # 按类别分组
        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        # 打印每个类别的样本数，用于调试
        for cl in self.cl_list:
            print(f"Class {cl} has {len(self.sub_meta[cl])} samples")

        # 为每个类别创建独立的DataLoader
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True  # 添加这行，丢弃最后一个不完整的batch
        )

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform, repeat_num=repeat_num)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, repeat_num=1):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.repeat_num = repeat_num

    def __getitem__(self, i):
        # 确保索引在范围内
        i = i % len(self.sub_meta)  # 添加这行，循环使用数据
        image_path = self.sub_meta[i]
        img = Image.open(image_path).convert('RGB')

        img_replicas = [img] * self.repeat_num
        transformed_imgs = []
        for img_replica in img_replicas:
            if self.transform:
                transformed_img = self.transform(img_replica)
            else:
                transformed_img = img_replica
            transformed_imgs.append(transformed_img)

        img = torch.cat(transformed_imgs, dim=0)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class SimpleDataset_JSON:
    def __init__(self, data_path, data_file, transform, target_transform=identity):
        # JSON文件完整路径
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            # 加载JSON标注（格式：{"image_names": [...], "image_labels": [...]}）
            self.meta = json.load(f)
            # 图像预处理
        self.transform = transform
        # 标签变换
        self.target_transform = target_transform

    def __getitem__(self, i):
        # 从JSON取第i张图片路径
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # 从JSON取标签
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        # JSON中图片列表的长度
        return len(self.meta['image_names'])


class SetDataset_JSON:
    def __init__(self, data_path, data_file, batch_size, transform, repeat_num=1):
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            # 加载JSON标注
            self.meta = json.load(f)
        # 提取所有类别
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            # 按类别分组（从JSON的image_names和image_labels取数据）
            self.sub_meta[cl] = []
        # 按类别分组：从JSON的image_names和image_labels取数据
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        # 为每个类别创建DataLoader
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset_JSON(self.sub_meta[cl], cl, transform=transform, repeat_num=repeat_num)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset_JSON:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, repeat_num=1):
        # 单个类别的图片路径
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

        self.repeat_num = repeat_num

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))

        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img_replicas = [img] * self.repeat_num
        transformed_imgs = []
        for img_replica in img_replicas:
            if self.transform:
                transformed_img = self.transform(img_replica)
            else:
                transformed_img = img_replica
            transformed_imgs.append(transformed_img)
        img = torch.cat(transformed_imgs, dim=0)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

# 批次采样器：从类别集合中采样N个类别（N-way）
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        # 总类别数
        self.n_classes = n_classes
        # 每次采样的类别数（N-way，如5-way）
        self.n_way = n_way
        # 训练轮次
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # 随机打乱所有类别，取前n_way个（每次采样不同的N个类别）
            yield torch.randperm(self.n_classes)[:self.n_way]
