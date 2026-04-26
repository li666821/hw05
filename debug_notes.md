# 极简CNN代码调试记录

## 问题1：中文路径导致的编码问题

### 现象
运行代码时出现文件路径相关的错误，特别是在保存图像文件时。

### 原因分析
matplotlib在处理中文路径时可能会出现编码问题，导致无法正确保存图像。

### 修改点
- 确保所有文件路径使用英文，避免中文路径
- 代码中已设置`matplotlib.rcParams['font.family'] = 'SimHei'`来支持中文显示，确保字体正确加载

## 问题2：CUDA不可用导致的设备选择问题

### 现象
代码运行时显示"使用设备: CPU"，即使有GPU可用。

### 原因分析
- CUDA驱动未安装或版本不兼容
- PyTorch未安装GPU版本

### 修改点
- 安装对应版本的CUDA驱动
- 安装GPU版本的PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 问题3：依赖库版本问题

### 现象
运行时出现库函数调用错误，如`torch.utils.data.DataLoader`参数错误。

### 原因分析
PyTorch或torchvision版本过低，导致API不兼容。

### 修改点
- 更新依赖库：`pip install --upgrade torch torchvision matplotlib numpy`
- 确保版本兼容性：PyTorch >= 1.10.0

## 问题4：数据下载路径问题

### 现象
数据下载失败，或路径不存在导致的错误。

### 原因分析
- 网络连接问题导致MNIST数据下载失败
- 数据存储路径权限不足

### 修改点
- 确保网络连接正常
- 检查`data`目录是否存在且有写入权限
- 如下载失败，可手动下载MNIST数据集并解压到`data/MNIST/raw/`目录

## 问题5：内存不足问题

### 现象
运行时出现`CUDA out of memory`错误。

### 原因分析
- 批量大小(batch_size)设置过大，超出GPU内存

### 修改点
- 减小批量大小：将`batch_size`从64调整为32或16
- 使用CPU进行训练（速度会变慢）

## 问题6：matplotlib显示问题

### 现象
图像显示失败，或保存图像时出错。

### 原因分析
- matplotlib后端配置问题
- 缺少必要的字体文件

### 修改点
- 确保matplotlib正确安装
- 在代码中添加`plt.switch_backend('Agg')`以支持无显示环境下的图像保存
