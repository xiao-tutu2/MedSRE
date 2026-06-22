import numpy as np
import torch.optim
import time
import os
from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
import importlib
import sys

from utils import *
import argparse
import tqdm

# 参数解析
# 创建参数解析器对象
parser = argparse.ArgumentParser()
# 图像尺寸：默认84，可选84/224
parser.add_argument('--image_size', default=224, type=int, choices=[84, 224])
# 数据集：默认mini_imagenet
parser.add_argument('--dataset', default='mini_imagenet',
                    choices=['mini_imagenet', 'tiered_imagenet', 'cub', 'cifarfs'])
# 数据集根路径
parser.add_argument('--data_path',type=str)
# 骨干网络
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18', 'ResNet12Prior'])
# 少样本方法
parser.add_argument('--method', default='priornet',
                    choices=['protonet', 'priornet'])
# 元训练/验证/测试的类别数
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')

# 支持集样本数
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
# 查询集样本数
parser.add_argument('--n_query', default=16, type=int,
                    help='number of unlabeled data in each class during meta validation')
# 测试的episode数
parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
# 预训练模型路径
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
# 测试重复次数
parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')
# 逻辑回归惩罚系数
parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
# 降维层输出维度
parser.add_argument('--reduce_dim', default=640, type=int,
                    help='the output dimensions of BDC dimensionality reduction layer')
# Dropout率
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
# 重复次数
parser.add_argument('--repeat_num', type=int, default=1,
                    help='')
# RML损失的权重系数
parser.add_argument("--ls", type=float, default=1.0,
                    help="RML Lamadas")
parser.add_argument("--lu", type=float, default=1.0,
                    help="RML Lamadau")
# 解析命令行参数，存入params对象
params = parser.parse_args()

# 总：环境与参数校准
import platform
system = platform.system()  #系统
#gpu
params.gpu = str(get_least_used_gpu_memory())
torch.cuda.set_device(get_least_used_gpu_memory())
# 根据骨干网络调整降维维度
if params.model == 'ResNet12' or 'ResNet12Prior':
    params.reduce_dim = 640
else:
    params.reduce_dim = 512
# 数据集适配：CIFAR-FS的图像尺寸强制为32
if params.dataset == 'cifarfs':
    params.image_size = 32

# 数据集加载器构建
# 标记是否需要读取JSON标签
json_file_read = False
# 根据数据集类型选择测试集文件
if params.dataset == 'cub':
    # CUB的测试集标签是JSON文件
    novel_file = 'novel.json'
    # 开启JSON读取模式
    json_file_read = True
elif params.dataset == 'cifarfs':
    # CIFAR-FS的测试集目录名
    novel_file = 'meta-test'
else:
    # mini/tiered_imagenet的测试集目录名
    novel_file = 'test'
    # 少样本测试的核心参数
novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, repeat_num=params.repeat_num)
# 初始化数据管理器：数据集路径、图像尺寸、每类查询集样本数、测试episode数、是否读JSON标签、少样本核心参数
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                               n_episode=params.test_n_episode, json_read=json_file_read, **novel_few_shot_params)
# 获取测试集数据加载器
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)


# 总：模型初始化（加载核心网络）
# 分支1：Protonet
if params.method == 'protonet':
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
# 分支2：Priornet
elif params.method == 'priornet':
    # 获取模型路径的上级目录
    experiment_dir = os.path.dirname(params.model_path)
    # 将目录加入系统路径
    sys.path.append(experiment_dir)
    # 动态导入priornet模块
    model = importlib.import_module(params.method)
    # 初始化Priornet
    model = model.PriorNet(params, model_dict[params.model], **novel_few_shot_params)

# model save path
model = model.cuda()
# 测试阶段：设置模型为评估模式
model.eval()

print(params.model_path)

# 拼接模型文件的完整路径
model_file = os.path.join(params.model_path)
# 加载预训练权重
model = load_model(model, model_file, is_train=False)
print(params)

# 测试循环
# 每个测试任务的episode数
iter_num = params.test_n_episode
# 存储多次测试任务的精度结果
acc_all_task = []
# 外层循环：重复test_task_nums次测试
for _ in range(params.test_task_nums):
    # 存储当前测试任务的所有episode精度
    acc_all = []
    # 记录测试开始时间
    test_start_time = time.time()
    # 包装数据加载器，显示进度条
    tqdm_gen = tqdm.tqdm(novel_loader)
    # 内层循环：遍历每个episode
    for _, (x, _) in enumerate(tqdm_gen):
        # 禁用梯度计算
        with torch.no_grad():
            # 设置模型的查询集样本数
            model.n_query = params.n_query
            # 分支1：Priornet前向传播
            if params.method in ['priornet']:
                scores = model.set_forward(x, False, False)
            # 分支2：Protonet前向传播
            else:
                scores = model.set_forward(x, False)
            # 预测：取score最大值的索引
            pred = scores.data.cpu().numpy().argmax(axis=1)

        # 构建真实标签：每类重复n_query次
        y = np.repeat(range(params.test_n_way), params.n_query)
        # 计算当前episode的精度
        acc = np.mean(pred == y) * 100
        # 保存当前episode精度
        acc_all.append(acc)
        # print(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')
        # 更新进度条描述：显示当前平均精度+当前episode精度
        tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')
    # 统计当前测试任务的精度
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
          % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
    # 保存当前测试任务的所有episode精度
    acc_all_task.append(acc_all)
# 统计所有测试任务的平均精度
acc_all_task_mean = np.mean(acc_all_task)
print('%d test mean acc = %4.2f%%' % (params.test_task_nums, acc_all_task_mean))
