import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from tqdm import tqdm


class MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()

        self.n_way = n_way
        # 每类支持集样本数
        self.n_support = n_support
        # 每类查询集样本数
        self.n_query = params.n_query  # (change depends on input)
        self.feature = model_func()
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.params = params

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def feature_forward(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            # 如果x已是提取好的特征，直接赋值
            z_all = x
        else:
            # 重塑输入形状
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            x = self.feature.forward(x)
            # 提取特征
            z_all = self.feature_forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        # 拆分支持集特征
        z_support = z_all[:, :self.n_support]
        # 拆分查询集特征
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    # 准确率计算
    def correct(self, x):
        # 前向传播得到查询集的类别得分
        scores = self.set_forward(x)
        # 生成查询集的真实标签
        y_query = np.repeat(range(self.n_way), self.n_query)

        # 取每个查询样本得分最高的1个类别
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        # 统计正确预测数：预测标签 == 真实标签的数量
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        # 返回正确数、总样本数
        return float(top1_correct), len(y_query)

# 训练循环方法
    def train_loop(self, epoch, train_loader, optimizer):
        # print_freq = 200
        # 累计损失
        avg_loss = 0
        # 保存每个batch的准确率
        acc_all = []
        # 训练迭代次数
        iter_num = len(train_loader)
        if epoch == 0 or epoch == 51:
            # 遍历训练数据加载器
            for i, (x, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                # 更新查询集数量：输入x的第二维是support+query总数，减去n_support得到n_query
                self.n_query = x.size(1) - self.n_support
                # 若允许动态改类别数
                if self.change_way:
                    self.n_way = x.size(0)
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播计算损失和准确率
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                # 记录当前batch的准确率
                acc_all.append(correct_this / count_this * 100)
                # 反向传播计算梯度
                loss.backward()
                # 优化器更新模型参数
                optimizer.step()
                # 累计损失
                avg_loss = avg_loss + loss.item()
        else:
            for i, (x, _) in enumerate(train_loader, 0):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                optimizer.zero_grad()
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss + loss.item()
            # if i % print_freq == 0:
            #     print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
            #                                                             avg_loss / float(i + 1)))
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                avg_loss / float(i + 1)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        # 返回epoch平均损失、平均准确率
        return avg_loss / iter_num, acc_mean

    # 测试循环方法
    def test_loop(self, test_loader, record=None):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        # 禁用梯度计算
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                # 更新n_query和n_way
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                    # 前向计算损失和准确率
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        # 测试准确率均值
        acc_mean = np.mean(acc_all)
        # 准确率标准差
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean
