import torch.optim
import time
from data.datamgr import SetDataManager
import sys
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('./methods')
from methods.protonet import ProtoNet
from methods.priornet import PriorNet
import shutil
from utils import *
# from network import utils
# 初始化:参数解析 + 环境配置
def train(params, base_loader, val_loader, model, stop_epoch):
    # 初始化训练日志字典，用于记录训练过程指标
    trlog = {}
    # 保存训练参数
    trlog['args'] = vars(params)
    # 记录每个epoch的训练损失
    trlog['train_loss'] = []
    # 记录每个epoch的验证损失
    trlog['val_loss'] = []
    # 记录每个epoch的训练准确率
    trlog['train_acc'] = []
    # 记录每个epoch的验证准确率
    trlog['val_acc'] = []
    # 初始化最优验证准确率
    trlog['max_acc'] = 0.0
    # 初始化最优准确率对应的epoch
    trlog['max_acc_epoch'] = 0

    # 冻结 / 解冻骨干网络
    Freezed = True
    for param in model.feature.parameters():
        param.requires_grad = False  # 冻结骨干网络，这部分网络有与训练权重
    if hasattr(model.feature, 'prior'):
        for param in model.feature.prior.parameters():
            # 仅解冻prior分支参数
            param.requires_grad = True
    # 初始化Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=params.lr,
    #     weight_decay=1e-4  # 添加L2正则化
    # )
    # 学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    

    def get_optimizer(self):
        return torch.optim.Adam(self.param_groups)

    if not os.path.isdir(params.checkpoint_dir):
        # 创建模型保存目录，避免路径不存在报错
        os.makedirs(params.checkpoint_dir)

    # 主训练循环
    # epoch循环与骨干网络解冻
    for epoch in range(0, stop_epoch):
        print('-------------------------------------------')
        print('Start Epoch:', epoch)
        if epoch > 5 and Freezed:
            print('Unfreezed Backbone....')
            Freezed = False
            for param in model.feature.parameters():
                param.requires_grad = True  # 冻结骨干网络，这部分网络有与训练权重
        # 单epoch训练
        # 记录epoch开始时间
        start = time.time()
        # 切换模型为训练模式
        model.train()
        # 执行模型训练循环，返回训练损失和准确率
        trainObj, top1 = model.train_loop(epoch, base_loader, optimizer)
        # trainObj, top1 = 0,0

        # 单epoch验证
        # 切换模型为验证模式
        model.eval()
        print('Start Testing Epoch:', epoch)
        # 执行验证循环，返回验证损失和准确率
        valObj, acc = model.test_loop(val_loader, tqdm_bar=epoch == 0)
        # 最优模型保存
        if acc > trlog['max_acc']:
            print("best model! save...")
            # 更新最优准确率
            trlog['max_acc'] = acc
            # 更新最优epoch
            trlog['max_acc_epoch'] = epoch
            # 保存最优模型参数
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            # if acc > 84:
            #     os.system(
            #         'python test.py --data_path %s --method %s --model_path %s  --test_task_nums 1  --test_n_episode 1000' % (
            #             params.data_path, params.method, outfile))
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            # 按频率保存
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar'.format(epoch))
            # 保存最后一个epoch的模型
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        # 日志更新与保存
        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)
        trlog['val_loss'].append(valObj)
        trlog['val_acc'].append(acc)
        # 保存日志字典

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))
        # 学习率更新与epoch信息打印
        # 更新学习率
        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))
        print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
        print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))

    return model
# 参数解析器初始化
if __name__ == '__main__':
    # 创建命令行参数解析器
    # 基础参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=224, type=int, choices=[84, 224],
                        help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate of the backbone')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=100, type=int, help='Stopping epoch')
    # parser.add_argument('--gpu', default='3', help='gpu id')
    parser.add_argument('--gpu', default='', help='gpu device ids for CUDA')


    # 数据集参数
    parser.add_argument('--dataset', default='busi',
                        choices=['mini_imagenet', 'tiered_imagenet', 'cub', 'cifarfs','busi'])
    parser.add_argument('--data_path', type=str, default=r'/home/BUSI',help='dataset path')
    # 模型与方法参数
    parser.add_argument('--model', default='ResNet18Prior', choices=['ResNet12', 'ResNet50Prior','ResNet18', 'ResNet18Prior','ResNet12Prior'])
    parser.add_argument('--method', default='priornet',
                        choices=[ 'protonet', 'priornet'])
    # 少样本训练参数
    parser.add_argument('--train_n_episode', default=300, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=3, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=3, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=8, type=int, help='number of unlabeled data in each class')
    # 辅助参数
    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
    # parser.add_argument('--pretrain_path',
    #                     help='pre-trained model .tar file path')
    # parser.add_argument('--pretrain_path', type=str, default='/home/SRE-ProtoNet-BUSI/model.pt_ResNet50_5shot.pt',
    parser.add_argument('--pretrain_path', type=str, default='/home/SRE-ProtoNet-BUSI/model.pt_ResNet18_5shot.pt',
    # parser.add_argument('--pretrain_path', type=str, default='/home/SRE-ProtoNet-BUSI/model.pt_epoch80.pt',
                        help='Path to pretrain weights (no need if training from scratch)')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--reduce_dim', type=int,
                        help='the output dimension of ProtoNet dimensionality reduction layer')

    parser.add_argument('--repeat_num', type=int, default=1,
                        help='')
    parser.add_argument("--ls", type=float, default=1,
                        help="RML Lamadas")
    parser.add_argument("--lu", type=float, default=1,
                        help="RML Lamadau")
    # 解析命令行参数，存入params对象
    params = parser.parse_args()

    # 总：设备与随机种子设置
    import platform
    system = platform.system()
    # params.gpu = str(get_least_used_gpu_memory())
    # print('use gpu:', params.gpu)

    device = torch.device('cuda')  # 直接设置为CPU
    print('使用设备: gpu')

    torch.cuda.set_device(get_least_used_gpu_memory())

    if params.model == 'ResNet12' or 'ResNet12Prior':
        # ResNet12系列的降维层输出维度
        params.reduce_dim = 640
    else:
        # ResNet18系列的降维层输出维度
        params.reduce_dim = 512
    if params.n_shot == 1:
        # 1-shot时增加训练任务数，保证数据量
        params.train_n_episode = 1000
    else:
        params.train_n_episode = 300
    if params.dataset == 'cifarfs':
        # CIFAR-FS数据集图像尺寸为32x32
        params.image_size = 32
    # 随机生成种子
    params.seed = random.randint(1, 999)
    set_seed(params.seed)
    print('seed:', params.seed)


    

    # 总：数据集路径与参数配置
    json_file_read = False
    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        val_file = 'val.json'
        # CUB数据集需读取JSON文件划分训练/验证集
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 351
    elif params.dataset == "cifarfs":
        base_file = 'meta-train'
        val_file = 'meta-val'
        params.num_classes = 643
    elif params.dataset == 'busi':  # 新增
        base_file = 'train'
        val_file = 'val'
        json_file_read = False
        params.num_classes = 3
    else:
        ValueError('dataset error')

    #总： 数据加载器创建
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, repeat_num=1)
    base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                  n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)

    # if params.dataset == 'busi':
    #     base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
    #                                   n_episode=params.train_n_episode, json_read=json_file_read,
    #                                   dataset_type='busi', **train_few_shot_params)
    # else:
    #     base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
    #                                   n_episode=params.train_n_episode, json_read=json_file_read,
    #                                   **train_few_shot_params)
    # 训练集开启数据增强
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot, repeat_num=params.repeat_num)
    val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                 n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
    # 验证集关闭数据增强
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    #总： 检查点目录拼接
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_metatrain'
    params.checkpoint_dir += params.extra_dir
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    print(params.checkpoint_dir)
    print(params.pretrain_path)

    # 初始化原型网络
    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot_params)
    elif params.method == 'priornet':
        # 初始化先验网络
        model = PriorNet(params, model_dict[params.model], **train_few_shot_params)     
        print("步骤1: 初始化 PriorNet 模型...")

# 按需启用插件（默认全部关闭）
    

        if hasattr(model.feature, 'use_denoise'):
            print(f"  ✓ 自适应降噪 (Denoise): {'启用' if model.feature.use_denoise else '关闭'}")
        if hasattr(model.feature, 'use_robust_sim'):
            print(f"  ✓ 鲁棒相似度 (Robust Similarity): {'启用' if model.feature.use_robust_sim else '关闭'}")
        if hasattr(model.feature, 'use_gated_fusion'):
            print(f"  ✓ 门控融合 (Gated Fusion): {'启用' if model.feature.use_gated_fusion else '关闭'}")
        print("="*50 + "\n")



        # 复制关键代码到检查点目录
        # shutil.copy('./methods/%s.py' % params.method, str(params.checkpoint_dir))
        # shutil.copy('./methods/prior_template.py', str(params.checkpoint_dir))
        # shutil.copy('./meta_train.py', str(params.checkpoint_dir))

        shutil.copy(os.path.join(BASE_DIR, 'methods', '%s.py' % params.method), str(params.checkpoint_dir))
        shutil.copy(os.path.join(BASE_DIR, 'methods', 'prior_template.py'), str(params.checkpoint_dir))
        shutil.copy(os.path.join(BASE_DIR, 'meta_train.py'), str(params.checkpoint_dir))

    model = model.cuda()
    # model = model.to('cpu')

    # 加载预训练模型
    modelfile = os.path.join(params.pretrain_path)
    model = load_model(model, modelfile, is_train=True)
    # modelfile = None  # 初始化预训练文件路径为None
    # # 只有当 pretrain_path 不为空时，才处理路径
    # if params.pretrain_path and params.pretrain_path.strip() != '':
    #     modelfile = os.path.join(params.pretrain_path)
    #     # 如果有预训练文件，才加载）
    #     if os.path.exists(modelfile):
    #         # checkpoint = torch.load(modelfile, map_location=f'cuda:{params.gpu}')
    #         checkpoint = torch.load(modelfile, map_location=device)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print(f"Loaded pretrained weights from {modelfile}")
    #     else:
    #         print(f"Warning: Pretrain path {modelfile} not found, training from scratch")
    # else:
    #     # 无预训练路径，直接提示「从头训练」
    #     print("No pretrain path provided, training from scratch")
    # 启动训练与测试
    # 启动核心训练函数
    model = train(params, base_loader, val_loader, model, params.epoch)
    # 训练完成后，自动执行测试脚本，评估最优模型
    os.system(
        'python test.py --data_path %s --method %s --model_path %s --n_shot %d  --dataset %s --model %s --repeat_num  %d' % (
            params.data_path, params.method,
            os.path.join(
                str(params.checkpoint_dir),
                'best_model.tar'),
            params.n_shot, params.dataset, params.model, params.repeat_num))



# ============= 画图部分 =============
import matplotlib.pyplot as plt
import torch
import os

# 加载训练日志
trlog_path = os.path.join(params.checkpoint_dir, 'trlog')
if os.path.exists(trlog_path):
    trlog = torch.load(trlog_path)

    # 创建画布
    plt.figure(figsize=(12, 5))

    # 左图：损失函数
    plt.subplot(1, 2, 1)
    epochs = range(1, len(trlog['train_loss']) + 1)
    plt.plot(epochs, trlog['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, trlog['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trlog['train_acc'], 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, trlog['val_acc'], 'r-', label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记最佳准确率
    best_acc = trlog['max_acc']
    best_epoch = trlog['max_acc_epoch'] + 1
    plt.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5,
                label=f'Best: {best_acc:.2f}%')
    plt.legend()

    plt.suptitle(f'{params.dataset} - {params.method} - {params.n_shot}shot',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(params.checkpoint_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存到: {save_path}")

    # 显示图片（如果在有图形界面的环境）
    plt.show()

    # 打印简单总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"最终准确率: {trlog['val_acc'][-1]:.2f}%")
    print("=" * 50)
else:
    print("未找到训练日志文件")