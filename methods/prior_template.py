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
    # 基础架构接口
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        # 小样本任务的“类别数”
        self.n_way = n_way
        # 每个类别的支持集样本数
        self.n_support = n_support
        # 每个类别的查询集样本数
        self.n_query = params.n_query  # (change depends on input)
        # 特征提取网络
        self.feature = model_func()
        # 是否允许训练/测试时类别数不同
        self.change_way = change_way  # some methods allow different_way classification during training and test
        # 全局参数配置
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

    # 前向传播（特征提取）
    def forward(self, x):
        out = self.feature.forward(x)
        return out
    # 特征分离（分割支持集 / 查询集）




    def parse_feature(self, x, is_feature):
        # 将数据移到GPU，并封装为Variable
        x = Variable(x.cuda())
        # x = Variable(x)
        # 如果输入已经是特征
        if is_feature:
            z_all = x
        # 如果输入是原始数据，先提取特征
        else:
            # 重塑数据形状
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            # 调用特征提取网络
            (out, out_rot, context_prior_map), z_all = self.feature.forward(x, return_map=True)

            # x = self.feature.forward(x)
            # z_all, context_prior_map = self.feature_forward(x)
            # 重塑特征形状

            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        # 分割支持集特征：取每个类的前n_support个样本
        z_support = z_all[:, :self.n_support]
        # 分割查询集特征：取每个类的后n_query个样本
        z_query = z_all[:, self.n_support:]

        return z_support, z_query, context_prior_map


# 计算分类准确率
    def correct(self, x):
        # 步骤1：获取模型预测。调用子类实现的前向传播，得到查询集分类得分
        scores = self.set_forward(x)
        # 步骤2：生成查询集的真实标签
        y_query = np.repeat(range(self.n_way), self.n_query)
        # 步骤3：获取预测标签。取分类得分的top1（最高得分），返回得分和对应的标签
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        # 转到CPU并转为numpy数组
        topk_ind = topk_labels.cpu().numpy()
        # topk_ind = topk_labels.numpy()
        # 步骤4：统计top1正确的数量：预测标签 == 真实标签
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        # 返回正确数、总样本数w
        return float(top1_correct), len(y_query)

# 训练循环
    def train_loop(self, epoch, train_loader, optimizer):
        # print_freq = 200
        # 欧氏距离损失均值
        avg_euclidean_dist_loss = 0
        # 亲和度距离损失均值
        avg_affinity_dist_loss = 0
        # 总损失值
        avg_loss = 0
        # 存储每个批次的准确率
        acc_all = []
        # 总批次数量
        iter_num = len(train_loader)
        # 特殊处理：第0/51个epoch显示进度条（tqdm），其他epoch不显示
        if epoch == 0 or epoch == 51:
            # ①：数据准备
            for i, (x, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                # 更新查询集样本数：总样本数 - 支持集样本数
                self.n_query = x.size(1) - self.n_support
                # 如果允许动态调整类别数，从输入数据中获取当前批次的类别数
                if self.change_way:
                    self.n_way = x.size(0)
                    # 清空梯度
                optimizer.zero_grad()
                # ②：前向传播与损失计算。调用子类实现的set_forward_loss，返回正确数、总样本数、损失、（未使用的返回值）
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                # 存储当前批次准确率
                acc_all.append(correct_this / count_this * 100)
                # 解析损失：loss是元组，[0]是欧式损失，[1]是亲和度损失（为0则仅用欧式损失）
                # ③：损失处理
                if loss[1] == 0:
                    euclidean_dist_loss = loss[0]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()

                    loss = euclidean_dist_loss
                else:
                    euclidean_dist_loss = loss[0]
                    affinity_dist_loss = loss[1]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    avg_affinity_dist_loss = avg_affinity_dist_loss + affinity_dist_loss.item()
                    # 总损失 = 两种损失之和
                    loss = euclidean_dist_loss + affinity_dist_loss
                # ③：反向传播与参数更新
                # 反向传播计算梯度
                loss.backward()
                # 更新参数
                optimizer.step()
                # 累加总损失
                avg_loss = avg_loss + loss.item()

        # 非0/51 epoch，逻辑同上，但不显示进度条
        else:
            for i, (x, _) in enumerate(train_loader, 0):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                optimizer.zero_grad()
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                if loss[1] == 0:
                    euclidean_dist_loss = loss[0]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    loss = euclidean_dist_loss
                else:
                    euclidean_dist_loss = loss[0]
                    affinity_dist_loss = loss[1]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    avg_affinity_dist_loss = avg_affinity_dist_loss + affinity_dist_loss.item()
                    loss = euclidean_dist_loss + affinity_dist_loss
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss + loss.item()
            # if i % print_freq == 0:
            #     print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
            #                                                             avg_loss / float(i + 1)))
        # print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
        #                                                         avg_loss / float(i + 1)))
        # 打印当前epoch的损失统计
        print('Epoch {:d} | Batch {:d}/{:d} | Euclidean Loss {:f} | Affinity_loss {:f}'.format(epoch, i,
                                                                                               len(train_loader),
                                                                                               avg_euclidean_dist_loss / float(
                                                                                                   i + 1),
                                                                                               avg_affinity_dist_loss / float(
                                                                                                  i + 1)))
        # ④统计指标
        # 计算平均准确率
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        # 返回平均损失、平均准确率
        return avg_loss / iter_num, acc_mean

# 测试循环
    def test_loop(self, test_loader, record=None, tqdm_bar=False):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            if tqdm_bar:
                for i, (x, _) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                    # for i, (x, _) in enumerate(test_loader, 0):
                    self.n_query = x.size(1) - self.n_support
                    if self.change_way:
                        self.n_way = x.size(0)
                    correct_this, count_this, loss, _ = self.set_forward_loss(x)
                    if loss[1] == 0:
                        euclidean_dist_loss = loss[0]
                        loss = euclidean_dist_loss
                    else:
                        euclidean_dist_loss = loss[0]
                        affinity_dist_loss = loss[1]
                        loss = euclidean_dist_loss + affinity_dist_loss

                    acc_all.append(correct_this / count_this * 100)
                    avg_loss = avg_loss + loss.item()
            else:
                for i, (x, _) in enumerate(test_loader, 0):
                    self.n_query = x.size(1) - self.n_support
                    if self.change_way:
                        self.n_way = x.size(0)
                    correct_this, count_this, loss, _ = self.set_forward_loss(x)
                    if loss[1] == 0:
                        euclidean_dist_loss = loss[0]
                        loss = euclidean_dist_loss
                    else:
                        euclidean_dist_loss = loss[0]
                        affinity_dist_loss = loss[1]
                        loss = euclidean_dist_loss + affinity_dist_loss
                    acc_all.append(correct_this / count_this * 100)
                    avg_loss = avg_loss + loss.item()
        # 统计准确率均值和标准差
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        # 打印测试结果：平均准确率 + 95%置信区间（1.96*std/sqrt(样本数)）
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean
