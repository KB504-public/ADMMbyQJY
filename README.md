# ADMM 重建算法

### 简介：

迭代求解的 ADMM 算法，来源于 [LenslessLearning](https://github.com/Waller-Lab/LenslessLearning)，但是不包含可学习参数部分。

### 快速开始：

1. 克隆仓库
2. 按照标题三的步骤配置环境
3. 运行 ADMM.ipynb 文件

### 环境配置：

项目配置的核心点在于要使用 PyTorch 的 1.7 版本，因为要使用老版本的 fft 和 ifft 函数。

```
# 1️⃣ 创建名为 lensless_env 的 Conda 虚拟环境，Python 版本为 3.7
conda create -n lensless_env python=3.7 -y

# 2️⃣ 激活该环境
conda activate lensless_env

# 3️⃣ 安装指定版本的 NumPy（PyTorch 1.7.1 兼容的版本）
conda install numpy=1.21.5 -y

# 4️⃣ 安装 PyTorch 依赖 typing-extensions（部分旧版 PyTorch 需要）
conda install typing-extensions -y

# 5️⃣ 从本地文件安装 PyTorch CPU 版本（无需联网）
# 下载地址为：https://download.pytorch.org/whl/torch_stable.html
pip install ./torch-1.7.1+cpu-cp37-cp37m-win_amd64.whl

# 6️⃣ 安装常用的图像处理和可视化库（从 conda-forge 获取最新兼容版本）
conda install matplotlib scikit-image pillow -c conda-forge -y
```

**注意事项：**Jupyter 组件未在环境中安装，运行时由 VSCode 自动管理。
