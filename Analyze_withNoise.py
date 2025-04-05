import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- 配置 Matplotlib 支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 10})
# ---

# --- 参数 ---
TARGET_VIDEO = 'video_stay_dark.mp4' # 选择用于分析的视频
FRAME_INDEX_TO_ANALYZE = 50       # 选择要分析的帧索引
OUTPUT_DIR_NOISE = 'noise_analysis_output'

# 噪声参数
GAUSSIAN_MEAN = 0
GAUSSIAN_VAR = 0.01 # 方差，值越大噪声越强
SALT_VS_PEPPER_RATIO = 0.5
SALT_PEPPER_AMOUNT = 0.04 # 噪声比例，值越大噪声点越多

if not os.path.exists(OUTPUT_DIR_NOISE):
    os.makedirs(OUTPUT_DIR_NOISE)

# --- 加噪声函数 ---

def add_gaussian_noise(image, mean=0, var=0.01):
    """添加高斯噪声"""
    image = np.array(image/255, dtype=float) # 归一化到 [0, 1]
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1) # 截断到 [0, 1]
    noisy_image = np.uint8(noisy_image*255) # 转换回 uint8
    return noisy_image

def add_salt_pepper_noise(image, amount=0.04, salt_vs_pepper=0.5):
    """添加椒盐噪声"""
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

    # 添加盐噪声 (白色)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    # Ensure coords are valid indices
    coords_valid = tuple(c[c < s] for c, s in zip(coords, noisy_image.shape))
    if all(len(c) > 0 for c in coords_valid):
         noisy_image[coords_valid] = 255

    # 添加胡椒噪声 (黑色)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    # Ensure coords are valid indices
    coords_valid = tuple(c[c < s] for c, s in zip(coords, noisy_image.shape))
    if all(len(c) > 0 for c in coords_valid):
        noisy_image[coords_valid] = 0
    return noisy_image

# --- 统计量计算与绘图 ---

def calculate_stats(image):
    """计算图像的基本统计量"""
    mean = np.mean(image)
    variance = np.var(image)
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    return {'mean': mean, 'variance': variance, 'histogram': hist}

def plot_image_and_stats(original_img, noisy_img, noise_type, filename_suffix):
    """绘制原始图像、加噪图像及统计量对比图"""
    stats_orig = calculate_stats(original_img)
    stats_noisy = calculate_stats(noisy_img)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'原始图像 vs {noise_type}噪声', fontsize=16)

    # 绘制图像
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title('原始图像')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(noisy_img, cmap='gray')
    ax2.set_title(f'{noise_type}噪声图像')
    ax2.axis('off')

    # 绘制直方图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(stats_orig['histogram'], color='b', label=f'原始 (均值:{stats_orig["mean"]:.2f}, 方差:{stats_orig["variance"]:.2f})')
    ax3.plot(stats_noisy['histogram'], color='r', label=f'{noise_type} (均值:{stats_noisy["mean"]:.2f}, 方差:{stats_noisy["variance"]:.2f})', alpha=0.7)
    ax3.set_title('灰度直方图对比')
    ax3.set_xlabel('灰度值')
    ax3.set_ylabel('频率')
    ax3.legend()
    ax3.grid(True)

    # 文本显示统计量对比
    ax4 = fig.add_subplot(2, 2, 4)
    text_content = f"--- 统计量对比 ({noise_type}) ---\n\n"
    text_content += f"原始均值: {stats_orig['mean']:.3f}\n"
    text_content += f"{noise_type}均值: {stats_noisy['mean']:.3f}\n"
    text_content += f"均值变化: {stats_noisy['mean'] - stats_orig['mean']:.3f}\n\n"
    text_content += f"原始方差: {stats_orig['variance']:.3f}\n"
    text_content += f"{noise_type}方差: {stats_noisy['variance']:.3f}\n"
    text_content += f"方差变化: {stats_noisy['variance'] - stats_orig['variance']:.3f}"
    ax4.text(0.1, 0.5, text_content, fontsize=10, verticalalignment='center')
    ax4.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # 保存图像
    save_path = os.path.join(OUTPUT_DIR_NOISE, f'noise_comparison_{filename_suffix}.png')
    plt.savefig(save_path, dpi=150)
    print(f"噪声对比图已保存到: {save_path}")
    plt.close(fig)

# --- 主程序 --- 
if __name__ == "__main__":
    # 1. 读取指定视频的特定帧
    if not os.path.exists(TARGET_VIDEO):
        print(f"错误: 目标视频文件 {TARGET_VIDEO} 不存在！")
        exit()

    cap = cv2.VideoCapture(TARGET_VIDEO)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {TARGET_VIDEO}")
        exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if FRAME_INDEX_TO_ANALYZE >= frame_count:
        print(f"错误: 帧索引 {FRAME_INDEX_TO_ANALYZE} 超出视频总帧数 {frame_count}！")
        cap.release()
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX_TO_ANALYZE) # 定位到指定帧
    ret, original_frame_color = cap.read()
    cap.release()

    if not ret:
        print(f"错误: 无法读取视频 {TARGET_VIDEO} 的第 {FRAME_INDEX_TO_ANALYZE} 帧！")
        exit()

    # 转换为灰度图进行分析
    original_frame_gray = cv2.cvtColor(original_frame_color, cv2.COLOR_BGR2GRAY)
    print(f"已成功读取视频 '{TARGET_VIDEO}' 的第 {FRAME_INDEX_TO_ANALYZE} 帧 (灰度图)。")

    # 2. 添加不同类型的噪声
    print("正在添加高斯噪声...")
    noisy_gaussian = add_gaussian_noise(original_frame_gray, GAUSSIAN_MEAN, GAUSSIAN_VAR)

    print("正在添加椒盐噪声...")
    noisy_salt_pepper = add_salt_pepper_noise(original_frame_gray, SALT_PEPPER_AMOUNT, SALT_VS_PEPPER_RATIO)

    # 3. 分析并可视化结果
    print("正在生成高斯噪声对比图...")
    plot_image_and_stats(original_frame_gray, noisy_gaussian, "高斯", "gaussian")

    print("正在生成椒盐噪声对比图...")
    plot_image_and_stats(original_frame_gray, noisy_salt_pepper, "椒盐", "salt_pepper")

    print("--- 噪声分析完成 --- ")
