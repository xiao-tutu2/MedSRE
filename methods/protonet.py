import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate

# 定义ProtoNet类
class ProtoNet(MetaTemplate):
    # 初始化方法：params：模型超参数、model_func：特征提取器的构建函数、n_way：样本类别数、n_support：每个类别支持集的样本数
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        # 交叉熵损失
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    # 特征前向传播
    def feature_forward(self, x):
        # 进行平均池化和张量展平
        out = self.avgpool(x).view(x.size(0),-1)
        # 返回一维特征向量
        return out

    # 前向传播
    # is_feature = False：原始图像；否则是提取的特征
    def set_forward(self, x, is_feature=False):
        # 将输入x划分为支持集特征和查询集特征
        z_support, z_query = self.parse_feature(x, is_feature)
        # 计算每个类的原型
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        # 确保查询集特征连续
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # 计算查询集特征与每个类原型的欧式距离
        scores = self.euclidean_dist(z_query, z_proto)
        return scores

    # 前向传播 + 损失计算
    def set_forward_loss(self, x):
        scores, context_prior_map = self.set_forward(x)
        # 生成查询集的真实标签
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        # 将标签张量封装为Variable
        y_query = y_query.to(self.feature.parameters().__next__().device)
        # y_query = Variable(y_query.cuda())
        y_query = y_query.to(scores.device)
        # 生成标签用于后续准确率计算
        y_label = np.repeat(range(self.n_way), self.n_query)
        # 查询集相似度分数
        scores = self.set_forward(x)
        # 取相似度分数的Top-1预测结果
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        # topk_ind = topk_labels.numpy()
        topk_ind = topk_labels.cpu().numpy()
        # 计算Top1正确的样本数：取所有查询样本的Top1预测标签，与真实标签y_label比较，求和得到正确数
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        # 返回:Top1 正确数、查询集总样本数、交叉熵损失、与标签y_query计算损失、相似度分数
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores
    # 欧氏距离计算
    def euclidean_dist(self, x, y):
        # x: N x D查询特征
        # y: M x D原型特征
        # 查询样本数
        n = x.size(0)
        # 类别数
        m = y.size(0)
        # 特征维度
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        # 计算负欧式距离平方和作为相似度分数
        score = -torch.pow(x - y, 2).sum(2)
        return score
