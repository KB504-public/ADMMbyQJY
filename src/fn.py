import numpy as np
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt

def load_and_downsample_normal2_image(
    path,
    downsample=16,
    mode="gray",        # "gray" 或 "rgb"
    remove_bg=False,    # 是否去背景（仅灰度图有意义）
    normalize=True,     # 是否进行 L2 归一化
    visualize=False     # 是否显示前后效果
):
    """
    通用图像读取 + 均值降采样函数。
    
    参数:
        path: 图像路径
        downsample: 降采样倍数（例如 16）
        mode: 'gray' 或 'rgb'
        remove_bg: 是否去除左上角背景均值 (仅灰度模式)
        normalize: 是否进行 L2 归一化
        visualize: 是否显示原图与降采样图
    返回:
        处理后的图像 (float64 numpy 数组)
    """
    # === 读取图像 ===
    if mode == "gray":
        img = io.imread(path, as_gray=True).astype(np.float64)
        if remove_bg:
            bg = np.mean(img[:15, :15])
            img -= bg
    elif mode == "rgb":
        img = np.array(Image.open(path).convert('RGB'), dtype=np.float64)
    else:
        raise ValueError("mode 必须是 'gray' 或 'rgb'")
    
    # === 计算降采样比例 ===
    if mode == "gray":
        down_shape = (img.shape[0] // downsample, img.shape[1] // downsample)
        img_down = transform.downscale_local_mean(img, (downsample, downsample))
    else:
        # RGB 三通道情况：对每个通道分别降采样
        img_down = np.stack([
            transform.downscale_local_mean(img[..., c], (downsample, downsample))
            for c in range(3)
        ], axis=-1)
    
    # === L2 归一化 ===
    if normalize:
        img_down = img_down / (np.linalg.norm(img_down) + 1e-12)
    
    # === 可视化 ===
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if mode == "gray":
            axes[0].imshow(img, cmap='gray')
            axes[1].imshow(img_down, cmap='gray')
        else:
            axes[0].imshow(img / 255.0)
            axes[1].imshow(img_down / np.max(img_down))
        axes[0].set_title("Original Image")
        axes[1].set_title("Downsampled Image")
        plt.tight_layout()
        plt.show()
    
    return img_down


import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstruction(
    psf_resized,
    measurement_resized,
    reconstruction,
    ground_truth_file=None,
    iterations=300
):
    """
    可视化结果，包括：
    1. Ground truth（可选）
    2. PSF
    3. Measurement
    4. Reconstructed image
    """
    stages = []
    
    # 1️⃣ Ground truth
    if ground_truth_file is not None and os.path.exists(ground_truth_file):
        import imageio
        gt_img = imageio.imread(ground_truth_file)
        if gt_img.ndim == 2:  # 灰度转RGB
            gt_img = np.stack([gt_img]*3, axis=-1)
        stages.append(("Ground Truth", gt_img))
    else:
        stages.append(("Ground Truth", None))
    
    # 2️⃣ PSF
    stages.append(("PSF", psf_resized))
    
    # 3️⃣ Measurement
    stages.append(("Measurement", measurement_resized))
    
    # 4️⃣ Reconstruction
    stages.append((f"Reconstructed (iter={iterations})", reconstruction))
    
    fig, axes = plt.subplots(1, len(stages), figsize=(5*len(stages), 5))
    if len(stages) == 1:
        axes = [axes]
    
    for ax, (title, img) in zip(axes, stages):
        if img is None:
            ax.set_title(title + " (None)")
            continue
        
        if img.ndim == 2:  # 灰度
            imshow_obj = ax.imshow(img, cmap='gray')
        else:  # RGB
            img_show = np.clip(img / np.max(np.abs(img)), 0, 1)
            imshow_obj = ax.imshow(img_show)
        
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
