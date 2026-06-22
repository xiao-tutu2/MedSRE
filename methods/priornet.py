import torch
import torch.nn as nn
from certifi.__main__ import args
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from prior_template import MetaTemplate

# 元学习模型
class PriorNet(MetaTemplate):
    # 初始化：params：模型超参数配置、model_func：特征提取器构建函数、n_way：小样本分类的 “类别数”、n_support：每个类别的支持集样本数、repeat_num：特征重复计算的次数
    def __init__(self, params, model_func, n_way, n_support, repeat_num=1):
        super(PriorNet, self).__init__(params, model_func, n_way, n_support)
        # 分类损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 特征重复计算的次数
        self.repeat_num = repeat_num
        # 分类损失的权重系数
        self.ls = params.ls
        # 亲和度损失的权重系数
        self.lu = params.lu

    def set_forward(self, x, is_feature=False, return_map=True):
        # 多尺度特征融合
        if x.shape[2] == 3:
            # 单片段直接提取特征
            z_support, z_query, context_prior_map = self.parse_feature(x, is_feature)
        else:
            # 多片段：循环处理每个3通道片段
            num_segments = int(x.shape[2] / 3)
            for repeat in range(num_segments):
                if repeat == 0:
                    z_support, z_query, context_prior_map = self.parse_feature(
                        x[:, :, repeat * 3:repeat * 3 + 3, :, :], is_feature
                    )
                else:
                    z_support_temp, z_query_temp, context_prior_map_temp = self.parse_feature(
                        x[:, :, repeat * 3:repeat * 3 + 3, :, :], is_feature
                    )
                    z_support += z_support_temp
                    z_query += z_query_temp
                    if context_prior_map_temp is not None:
                        context_prior_map += context_prior_map_temp

            z_support /= num_segments
            z_query /= num_segments
            if context_prior_map is not None:
                context_prior_map /= num_segments

        # 计算原型向量
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        # 重塑查询集特征
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # 计算分类分数
        scores = self.euclidean_dist(z_query, z_proto)






        if return_map:
            return scores, context_prior_map
        else:
            return scores

# 前向传播

    # 前向传播并计算损失
    # priornet.py 修改 set_forward_loss 方法

    def set_forward_loss(self, x):
        scores, context_prior_map = self.set_forward(x, is_feature=False)

        # 构建真实标签（三分类）
        y_query = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_query)
        ).long()
        y_query = y_query.to(scores.device)
        y_label = np.repeat(range(self.n_way), self.n_query)

        # 计算准确率
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), (
            self.ls * self.loss_fn(scores, y_query),
            self.lu * self.affinity_loss(context_prior_map)
        ), scores
    # 欧式距离计算作为分类得分
    def euclidean_dist(self, x, y):
        # x: N x D  查询集特征：N个样本，D维特征
        # y: M x D  原型向量：M个类别，D维特征
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score

    # 亲和度损失计算
    # def affinity_loss(self, context_prior_map):
    #     # 构建理想亲和矩阵
    #     one_hot_labels = F.one_hot(torch.from_numpy(np.repeat(range(self.n_way), self.n_support + self.n_query)),
    #                                self.n_way)
    #     ideal_affinity_matrix = torch.matmul(one_hot_labels,
    #                                          one_hot_labels.transpose(0, 1)).cuda().float()
    #     # 计算二分类交叉熵损失
    #     BCE_LOSS = nn.BCELoss()
    #     bce = BCE_LOSS(context_prior_map, ideal_affinity_matrix)
    
    #     return bce

    # EG亲和损失：让特征空间的相似度矩阵尽量接近标签空间的相似度矩阵
    # def affinity_loss(self, features):

    #     """
    #     Compute affinity loss for the features
    #     """
    #     # 获取设备 - 使用 features 的设备
    #     device = features.device

    #     # 创建 one-hot 标签
    #     # 假设每个类有 (n_support + n_query) 个样本
    #     # 1.构建标签相似度矩阵
    #     labels = torch.arange(self.n_way, device=device)
    #     labels = labels.repeat_interleave(self.n_support + self.n_query)
    #     # 创建 one-hot 编码
    #     one_hot_labels = F.one_hot(labels.long(), num_classes=self.n_way).float()
    #     label_similarity = torch.mm(one_hot_labels, one_hot_labels.t())

    #     # 2.计算特征相似度矩阵
    #     features_norm = F.normalize(features, p=2, dim=1)
    #     similarity_matrix = torch.mm(features_norm, features_norm.t())
    #     # 计算标签相似度矩阵


    #     # 3.计算亲和损失
    #     loss = F.mse_loss(similarity_matrix, label_similarity)

    #     return loss

    # def affinity_loss(self, context_prior_map):
    #     """
    #     二阶监督 + 目标归一化（训练更稳定）
    #     """
    #     device = context_prior_map.device
        
    #     # 构建标签
    #     labels = torch.arange(self.n_way, device=device)
    #     labels = labels.repeat_interleave(self.n_support + self.n_query)
    #     one_hot_labels = F.one_hot(labels.long(), num_classes=self.n_way).float()
    #     Y = torch.mm(one_hot_labels, one_hot_labels.t())
        
    #     # 二阶目标 G = Y @ Y.T
    #     G = torch.mm(Y, Y.t())
        
    #     # 归一化目标到 [0, 1] 范围（让训练更稳定）
    #     max_val = self.n_support + self.n_query  # 每类样本数
    #     G_normalized = G / max_val  # 现在同类样本之间 = 1，异类 = 0
        
    #     # 预测的二阶矩阵
    #     A_norm = F.normalize(context_prior_map, p=2, dim=1)
    #     S = torch.mm(A_norm, A_norm.t())
        
    #     # MSE损失
    #     loss = F.mse_loss(S, G_normalized)
        
    #     return loss

    def affinity_loss(self, A):
        device = A.device

        # ===== label affinity =====
        labels = torch.arange(self.n_way, device=device)
        labels = labels.repeat_interleave(self.n_support + self.n_query)
        one_hot = F.one_hot(labels, num_classes=self.n_way).float()
        Y = one_hot @ one_hot.t()

        # ===== 1. 原始结构约束（你的核心）=====
        A_norm = F.normalize(A, dim=1)
        S = A_norm @ A_norm.t()
        loss_structure = F.mse_loss(S, Y)

        # ===== 2. 防塌缩约束（关键！）=====
        # 防止 A 所有行一样
        diversity = torch.var(A, dim=1).mean()

        loss = loss_structure - 0.1 * diversity

        return loss


                    
                            