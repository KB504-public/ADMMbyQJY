# ADMM 重建算法

### 简介：

迭代求解的 ADMM 算法，来源于 [LenslessLearning](https://github.com/Waller-Lab/LenslessLearning)，但是不包含可学习参数部分。

### 快速开始：

1. 克隆仓库
2. 按照标题三的步骤配置环境
3. 运行 ADMM.ipynb 文件

### 环境配置：

```shell
# 创建一个名为 lensless_env 的新 Conda 环境（此时还没有安装任何包）
conda create -n lensless_env -y

# 激活刚刚创建的环境，使后续安装和运行都在这个环境中进行
conda activate lensless_env

# 安装 Python、NumPy、Matplotlib、scikit-image 和 Pillow（图像处理库）到当前环境
conda install python numpy matplotlib scikit-image pillow -y

# 从 PyTorch 官方通道安装最新稳定版的 PyTorch、TorchVision 和 TorchAudio
# （-c pytorch 指定从 pytorch 官方频道获取软件包）
conda install pytorch torchvision torchaudio -c pytorch -y

# 查看当前环境中 Python 的版本，确认 Python 已安装成功
python -V

# 测试核心科学计算与图像处理库是否安装成功
# 如果没有错误并打印出 "All packages imported successfully!"，说明一切正常
python -c "import numpy, matplotlib, skimage, PIL; print('All packages imported successfully!')"
```

**注意事项：** Jupyter 组件未在环境中安装，运行时由 VSCode 自动管理。

