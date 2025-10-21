import numpy as np
import os
from skimage import io, transform
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
        img = io.imread(path).astype(np.float64)
        if img.ndim == 2:  # 灰度转 RGB
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:  # RGBA → RGB
            img = img[..., :3]
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

def visualize_reconstruction(
    psf_resized,
    measurement_resized,
    reconstruction,
    ground_truth_file=None,
    iterations=300
):
    """
    可视化结果：
    1. Ground truth（可选）
    2. PSF
    3. Measurement
    4. Reconstruction
    """
    stages = []

    # === 1️⃣ Ground truth (可选) ===
    if ground_truth_file and os.path.exists(ground_truth_file):
        gt_img = io.imread(ground_truth_file)
        if gt_img.ndim == 2:  # 灰度 → RGB
            gt_img = np.stack([gt_img]*3, axis=-1)
        elif gt_img.shape[2] == 4:  # RGBA → RGB
            gt_img = gt_img[..., :3]
        stages.append(("Ground Truth", gt_img))

    # === 2️⃣ PSF ===
    stages.append(("PSF", psf_resized))

    # === 3️⃣ Measurement ===
    stages.append(("Measurement", measurement_resized))

    # === 4️⃣ Reconstruction ===
    stages.append((f"Reconstructed (iter={iterations})", reconstruction))

    # === 仅保留有效的图像 ===
    valid_stages = [(title, img) for title, img in stages if img is not None]

    # === 创建对应数量的子图 ===
    fig, axes = plt.subplots(1, len(valid_stages), figsize=(5 * len(valid_stages), 5))
    if len(valid_stages) == 1:
        axes = [axes]

    # === 绘制每个图 ===
    for ax, (title, img) in zip(axes, valid_stages):
        if img.ndim == 2:  # 灰度
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img_show = np.clip(img / np.max(np.abs(img)), 0, 1)
            ax.imshow(img_show)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def save_reconstruction_image(reconstruction, save_path=None):
    """
    保存重建图像（Reconstruction）。

    Args:
        reconstruction (np.ndarray): 重建结果图像，形状可以是 (H, W) 或 (H, W, 3)
        save_path (str or None): 保存路径（如 "output/recon.png"）。
                                 若为 None，则不保存，只打印提示。

    Returns:
        None
    """
    # --- 检查输入 ---
    if reconstruction is None:
        print("[Warning] Reconstruction is None, nothing to save.")
        return

    # --- 归一化到 [0, 1] ---
    recon_norm = np.clip(reconstruction / np.max(np.abs(reconstruction)), 0, 1)

    # --- 如果没有指定保存路径 ---
    if save_path is None:
        print("[Info] No save path provided — image will not be saved.")
        return

    # --- 确保目录存在 ---
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- 保存 ---
    if recon_norm.ndim == 2:  # 灰度图
        io.imsave(save_path, (recon_norm * 255).astype(np.uint8))
    elif recon_norm.ndim == 3 and recon_norm.shape[2] == 3:  # RGB
        io.imsave(save_path, (recon_norm * 255).astype(np.uint8))
    else:
        raise ValueError(f"Unsupported reconstruction shape: {reconstruction.shape}")

    print(f"[Saved] Reconstruction image saved to: {save_path}")

def save_all_images(
    psf,
    measurement,
    reconstruction,
    ground_truth=None,
    save_dir=None,
    filenames=None
):
    """
    保存 PSF、Measurement、Reconstruction（以及可选 Ground Truth）。

    Args:
        psf (np.ndarray): PSF 图像
        measurement (np.ndarray): Measurement 图像
        reconstruction (np.ndarray): Reconstruction 图像
        ground_truth (np.ndarray or None): 真值图像（可选）
        save_dir (str or None): 保存目录路径，例如 "results/"
                                若为 None，则不保存，只打印提示
        filenames (dict or None): 自定义文件名，如：
                                  {"gt": "gt.png", "psf": "psf.png", "meas": "m.png", "recon": "r.png"}

    Returns:
        None
    """

    # === 情况1：没有给出保存目录 ===
    if save_dir is None:
        print("[Info] No save directory provided — images will not be saved.")
        return

    # === 情况2：确保目录存在 ===
    os.makedirs(save_dir, exist_ok=True)

    # === 文件名设置 ===
    default_filenames = {
        "gt": "gt.png",
        "psf": "psf.png",
        "meas": "m.png",
        "recon": "r.png"
    }
    if filenames is not None:
        default_filenames.update(filenames)

    # === 定义辅助函数（单张图保存） ===
    def _save_img(img, path, label):
        if img is None:
            print(f"[Skip] {label} is None, not saving.")
            return
        img_norm = np.clip(img / np.max(np.abs(img)), 0, 1)
        io.imsave(path, (img_norm * 255).astype(np.uint8))
        print(f"[Saved] {label} saved to: {path}")

    # === Ground Truth (可选) ===
    if ground_truth is not None:
        save_path = os.path.join(save_dir, default_filenames["gt"])
        _save_img(ground_truth, save_path, "Ground Truth")

    # === PSF ===
    save_path = os.path.join(save_dir, default_filenames["psf"])
    _save_img(psf, save_path, "PSF")

    # === Measurement ===
    save_path = os.path.join(save_dir, default_filenames["meas"])
    _save_img(measurement, save_path, "Measurement")

    # === Reconstruction ===
    save_path = os.path.join(save_dir, default_filenames["recon"])
    _save_img(reconstruction, save_path, "Reconstruction")