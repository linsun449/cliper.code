import cv2
import numpy as np


def save_heatmap(gray_image, save_path=None, base=2.5):
    # 将灰度值缩放到0-1之间
    gray_values_normalized = gray_image.astype(np.float32) / 255.0

    gray_values_normalized = np.clip(base ** gray_values_normalized - 1, 0, np.inf) / (base ** 1 - 1)
    # 定义颜色映射
    color_at_min = np.array([95, 127, 159]) / 255.0  # BDA5E7
    color_at_max = np.array([214, 240, 64]) / 255.0  # FF009C

    # 根据灰度值线性插值得到RGB值
    rgb_values = gray_values_normalized[:, :, np.newaxis] * (color_at_max - color_at_min) + color_at_min
    rgb_values = np.clip(rgb_values * 255, 0, 255).astype(np.uint8)

    # 如果指定了保存路径，则保存图像
    if save_path is not None:
        cv2.imwrite(save_path, rgb_values[..., [2, 1, 0]])

    return rgb_values