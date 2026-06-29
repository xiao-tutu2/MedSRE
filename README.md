# README.md
## 项目说明

本项目用于 BUSI 乳腺超声图像的小样本分类实验，代码主要包含训练入口、测试入口、方法模块和网络结构模块。

项目入口文件：
meta_train.py   # 训练入口
test.py         # 测试入口

项目结构：
<img width="4235" height="3352" alt="image" src="https://github.com/user-attachments/assets/28b08d81-7138-4690-ae7a-afe6d9c95c1e" />



## 目录结构

├── data/

│   └── BUSI/

│

├── methods/

│   ├── __init__.py

│   ├── prior_template.py

│   ├── priornet.py

│   ├── protonet.py

│   └── template.py

│

├── network/

│   ├── __init__.py

│   ├── resnet.py

│   └── resnet18.py

│

├── meta_train.py

├── test.py

└── README.md

## 环境配置
建议使用 Python 3.8 及以上版本。
安装常用依赖：

pip install torch torchvision

pip install numpy scipy scikit-learn

pip install pillow opencv-python

pip install tqdm matplotlib

也可以直接运行：pip install -r requirements.txt

## 数据集准备

将 BUSI 数据集放入 `data/` 目录下，推荐目录结构如下：

data/

└── BUSI/

    ├── benign/
    
    ├── malignant/
    
    └── normal/

三类图像分别对应：

benign      # 良性

malignant   # 恶性

normal      # 正常

运行前请确认数据路径与代码中的数据读取路径一致。

## 训练模型
在项目根目录下运行：

python meta_train.py

如果代码支持命令行参数，可以通过以下命令查看参数：

python meta_train.py --help

常见训练参数示例：

python meta_train.py --shot 5

如果需要训练不同 shot 设置，可以分别运行：

python meta_train.py --shot 1

python meta_train.py --shot 2

python meta_train.py --shot 3

python meta_train.py --shot 4

python meta_train.py --shot 

python meta_train.py --shot 10

注意：如果 `meta_train.py` 中没有设置命令行参数，请直接在代码文件中修改对应参数后运行。

## 测试模型

在项目根目录下运行：

python test.py

如果代码支持命令行参数，可以通过以下命令查看参数：

python test.py --help

常见测试命令示例：

python test.py --shot 5

如果需要测试不同 shot 设置，可以分别运行：

python test.py --shot 1

python test.py --shot 2

python test.py --shot 3

python test.py --shot 4

python test.py --shot 5

python test.py --shot 10

注意：如果 `test.py` 中没有设置命令行参数，请直接在代码文件中修改对应参数后运行。

## 主要文件说明

| 文件                          | 作用            |
| --------------------------- | ------------- |
| `meta_train.py`             | 模型训练入口        |
| `test.py`                   | 模型测试入口        |
| `methods/protonet.py`       | ProtoNet 方法实现 |
| `methods/priornet.py`       | PriorNet 方法实现 |
| `methods/template.py`       | 方法基础模板        |
| `methods/prior_template.py` | PriorNet 相关模板 |
| `network/resnet.py`         | ResNet 网络结构   |
| `network/resnet18.py`       | ResNet18 网络结构 |

## 运行流程
完整运行流程如下：

# 1. 安装依赖
# 安装 PyTorch 及 torchvision

pip install torch torchvision

# 安装科学计算库

pip install numpy

pip install scipy

pip install scikit-learn

# 安装图像处理库

pip install pillow

pip install opencv-python

# 安装工具库

pip install tqdm

pip install matplotlib

# 2. 准备数据集
# 将 BUSI 数据集放到 data/BUSI/ 目录下

# 3. 训练模型
python meta_train.py

# 4. 测试模型
python test.py
 注意事项
1. 运行代码前，请确认数据集路径正确。
2. 如果代码中使用 GPU，请确认 CUDA 和 PyTorch 版本匹配。
3. 如果运行时报路径错误，请检查 `data/BUSI/` 是否存在。
4. 如果运行时报参数错误，请先执行：

python meta_train.py --help
python test.py --help

5. 如果脚本不支持命令行参数，需要在 `meta_train.py` 或 `test.py` 内部直接修改参数。
