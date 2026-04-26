import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用支持中文的字体

class LeNet5(nn.Module):
    """LeNet-5卷积神经网络模型，用于MNIST手写数字分类
    
    结构：卷积层1 -> ReLU激活 -> 池化层1 -> 卷积层2 -> ReLU激活 -> 池化层2 -> 全连接层1 -> ReLU激活 -> 全连接层2
    """
    def __init__(self):
        """初始化网络结构，定义各层组件"""
        super(LeNet5, self).__init__()
        # 卷积层1：输入1通道(灰度图)，输出6通道，5x5卷积核，步长1，填充0
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # ReLU激活函数：增加非线性
        self.relu = nn.ReLU()
        # 池化层1：2x2窗口，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2：输入6通道，输出16通道，5x5卷积核，步长1，填充0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 全连接层1：将特征映射到120个神经元
        # 输入维度计算：16(通道数) * 4(池化后高度) * 4(池化后宽度) = 256
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        # 全连接层2：将120个神经元映射到84个神经元
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 全连接层3：将84个神经元映射到10个类别(数字0-9)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """定义前向传播过程
        
        参数:
            x: 输入张量，形状为[batch_size, 1, 28, 28]
            
        返回:
            输出张量，形状为[batch_size, 10]
        """
        # 卷积层1 -> ReLU激活 -> 池化层1
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积层2 -> ReLU激活 -> 池化层2
        x = self.pool(self.relu(self.conv2(x)))
        # 将特征图展平
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层1 -> ReLU激活
        x = self.relu(self.fc1(x))
        # 全连接层2 -> ReLU激活
        x = self.relu(self.fc2(x))
        # 全连接层3
        x = self.fc3(x)
        return x

def load_data(batch_size=64):
    """加载MNIST数据集并进行预处理
    
    参数:
        batch_size: 批量大小，默认为64
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据变换：转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量，并将像素值归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST数据集的均值和标准差进行归一化
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',  # 数据存储路径
        train=True,  # 指定为训练集
        download=True,  # 如果数据不存在，则下载
        transform=transform  # 应用上面定义的变换
    )

    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True# 打乱数据，增加随机性
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False# 测试集不需要打乱
    )

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device, epochs=5):
    """训练模型函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备(CPU或GPU)
        epochs: 训练轮数，默认为5
        
    返回:
        train_losses: 每个epoch的训练损失列表
        training_time: 训练总时间
    """
    model.train()  # 设置为训练模式
    train_losses = []  # 存储每个epoch的平均损失
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累加损失
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    training_time = time.time() - start_time
    return train_losses, training_time


def test(model, test_loader, criterion, device):
    """测试模型函数
    
    参数:
        model: 要测试的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备(CPU或GPU)
        
    返回:
        test_loss: 测试损失
        accuracy: 测试准确率
    """
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度，节省内存
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return test_loss, accuracy

def visualize_predictions(model, test_loader, device, num_images=5):
    """可视化模型的预测结果
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备(CPU或GPU)
        num_images: 要显示的图像数量，默认为5
    """
    model.eval()  # 设置为评估模式
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 将数据移动到指定设备
    images = images.to(device)
    labels = labels.to(device)
    
    # 前向传播获取预测结果
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        # 转换图像格式：[C,H,W] -> [H,W]，并反归一化
        img = images[i][0].cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'真实标签: {labels[i].item()}\n预测标签: {predicted[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lenet_predictions.png')
    plt.show()

def main():
    """主函数，协调整个流程：加载数据、创建模型、训练、测试和可视化
    """
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)  # 42是一个常用的随机种子值

    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')  # 打印使用的计算设备

    # 加载训练和测试数据，批量大小为64
    train_loader, test_loader = load_data(batch_size=64)

    # 创建模型实例并移动到指定设备（CPU或GPU）
    model = LeNet5().to(device)
    print(model)  # 打印模型结构

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率为0.001

    # 训练模型，运行5个训练周期（epoch）
    train_losses, training_time = train(model, train_loader, criterion, optimizer, device, epochs=5)

    # 在测试集上评估模型性能
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    # 可视化一些预测结果，直观地展示模型效果
    visualize_predictions(model, test_loader, device)  # 默认显示5个样本

    # 保存训练好的模型参数，便于未来使用
    torch.save(model.state_dict(), 'lenet5_mnist.pth')  # 只保存模型参数，而非整个模型
    print('模型已保存为 lenet5_mnist.pth')

    # 绘制训练损失曲线，可视化训练过程
    plt.figure(figsize=(10, 5))  # 创建图形，大小为10x5英寸
    plt.plot(train_losses, label='Training Loss')  # 绘制损失曲线
    plt.title('Training Loss Over Epochs (LeNet-5)')  # 添加标题
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.legend()  # 显示图例
    plt.savefig('lenet_training_loss.png')  # 保存图像到文件
    plt.show()  # 显示图像

    # 打印训练时间
    print(f'训练时间: {training_time:.2f}秒')

# 程序入口点
# 作为主程序运行
if __name__ == '__main__':
    # 这确保代码只在直接运行该文件时执行，而不是在被导入为模块时执行
    main()
