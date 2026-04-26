# hw05
极简CNN手写数字识别与LeNet-5 实现

## 目录结构

```
hw05/
├── data/             # MNIST数据集存储目录
├── simpleCNN.py      # 极简CNN模型实现
├── trainlenet.py     # LeNet-5模型实现
├── requirements.txt  # 依赖库清单
├── report.md         # 实验报告
├── debug_notes.md    # 调试记录
├── mnist_samples.png # MNIST样本图像
├── predictions.png   # 极简CNN预测结果
└── lenet_predictions.png # LeNet-5预测结果
```

## 任务说明

### 任务一：极简CNN实现
- 文件：`simpleCNN.py`
- 实现了一个简单的卷积神经网络，包含一个卷积层、一个池化层和一个全连接层
- 用于MNIST手写数字分类任务

### 任务二：LeNet-5实现
- 文件：`trainlenet.py`
- 实现了经典的LeNet-5网络结构，包含两个卷积层、两个池化层和三个全连接层
- 用于MNIST手写数字分类任务

## 环境配置

1. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```

2. Python版本要求：Python 3.7或更高版本

## 运行说明

### 训练和评估极简CNN模型
```bash
python simpleCNN.py
```
- 自动下载MNIST数据集（如果不存在）
- 训练模型并在测试集上评估
- 生成训练损失曲线和预测结果图像

### 训练和评估LeNet-5模型
```bash
python trainlenet.py
```
- 自动下载MNIST数据集（如果不存在）
- 训练模型并在测试集上评估
- 生成训练损失曲线和预测结果图像

## 调试记录

调试过程中遇到的问题及解决方案请参考`debug_notes.md`文件。

## 实验报告

详细的实验报告请参考`report.md`文件，包含LeNet-5结构说明、实验超参数与结果、模型对比分析等内容。

