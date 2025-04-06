import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import os
import time

# --- 配置 Matplotlib 支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams.update({'font.size': 12}) # 设置合适的字体大小
# ---

# --- 全局参数 ---
VIDEO_FILES = ['video_move_light.mp4', 'video_stay_dark.mp4']
OUTPUT_DIR = 'video_analysis_output'
PCA_N_COMPONENTS = 5 # PCA 保留的主成分数量 (修改：必须 <= 用于 PCA 的帧数)
MOTION_VAR_THRESHOLD = 500 # 运动检测时，时间方差的阈值 (需要根据实际情况调整)
ACF_PIXEL_COORDS = [(50, 50), (100, 150)] # 计算自相关函数的像素坐标 (示例)
# ---

# --- 创建输出目录 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ---

def save_plot(filename):
    """保存当前 matplotlib 图像到输出目录"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150) # 调整 DPI 以控制图像大小和清晰度
    print(f"图表已保存到: {filepath}")
    plt.close() # 关闭当前图像以释放内存

def analyze_video(video_path):
    """对单个视频文件进行统计分析"""
    print(f"\n--- 开始分析视频: {video_path} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}, {frame_count} 帧, {fps:.2f} FPS")

    # --- 初始化用于存储统计量的列表/数组 ---
    spatial_means = []
    spatial_vars = []
    frames_gray_list = [] # 存储灰度帧用于后续计算
    frames_rgb_list = [] # 存储彩色帧 (少量用于示例)
    pixel_time_series = {coord: [] for coord in ACF_PIXEL_COORDS} # 存储特定像素的时间序列

    # --- 逐帧读取和计算 ---
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray_list.append(gray_frame)
        if frame_num < 5: # 仅存储少量彩色帧示例
            frames_rgb_list.append(frame)

        # 1. 均值: 空间平均 (亮度)
        spatial_mean = np.mean(gray_frame)
        spatial_means.append(spatial_mean)

        # 2. 方差: 空间方差 (复杂度)
        spatial_var = np.var(gray_frame)
        spatial_vars.append(spatial_var)

        # 提取特定像素值用于 ACF
        for coord in ACF_PIXEL_COORDS:
            if 0 <= coord[0] < height and 0 <= coord[1] < width:
                 pixel_time_series[coord].append(gray_frame[coord[0], coord[1]])

        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  已处理 {frame_num}/{frame_count} 帧...")

    cap.release()
    print(f"视频读取完毕，总共 {frame_num} 帧。")

    if frame_num == 0:
        print("错误：未能读取任何帧。")
        return

    frames_gray_np = np.array(frames_gray_list) # (T, H, W)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- 可视化与进一步分析 ---

    # 1a. 绘制空间均值 (亮度) 随时间变化
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(frame_num) / fps, spatial_means)
    plt.title(f'{video_name} - 帧平均亮度随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('平均像素值')
    plt.grid(True)
    save_plot(f'{video_name}_spatial_mean_vs_time.png')

    # 1b. 计算并显示时间均值 (平均帧)
    temporal_mean_frame = np.mean(frames_gray_np, axis=0).astype(np.uint8)
    plt.figure(figsize=(8, 6))
    plt.imshow(temporal_mean_frame, cmap='gray')
    plt.title(f'{video_name} - 时间平均帧')
    plt.axis('off')
    save_plot(f'{video_name}_temporal_mean_frame.png')

    # 2a. 绘制空间方差 (复杂度) 随时间变化 (判断镜头移动)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(frame_num) / fps, spatial_vars)
    plt.title(f'{video_name} - 帧内方差 (复杂度/镜头移动指示) 随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('像素值方差')
    plt.grid(True)
    save_plot(f'{video_name}_spatial_variance_vs_time.png')

    # 2b. 计算时间方差 (运动区域检测)
    print("计算时间方差 (可能需要一些时间)...")
    start_time = time.time()
    temporal_variance_frame = np.var(frames_gray_np, axis=0)
    print(f"  时间方差计算耗时: {time.time() - start_time:.2f} 秒")

    # 对时间方差进行阈值处理，得到运动区域掩码
    motion_mask = (temporal_variance_frame > MOTION_VAR_THRESHOLD).astype(np.uint8) * 255

    # 寻找轮廓并绘制边界框
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_motion_boxes = cv2.cvtColor(temporal_mean_frame, cv2.COLOR_GRAY2BGR) # 在平均帧上绘制
    for contour in contours:
        if cv2.contourArea(contour) > 100: # 过滤掉太小的区域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_motion_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2) # 红色框

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(temporal_variance_frame, cmap='hot') # 用热力图显示方差大小
    plt.title(f'{video_name} - 像素时间方差')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(frame_with_motion_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f'{video_name} - 检测到的运动区域 (红框)')
    plt.axis('off')
    save_plot(f'{video_name}_motion_detection.png')

    # 3. 计算并绘制自相关函数 (ACF)
    plt.figure(figsize=(10, 5))
    for coord, series in pixel_time_series.items():
        if len(series) > 1:
            series_norm = series - np.mean(series) # 中心化
            # 计算 ACF，限制 lag 数量
            max_lag = min(len(series_norm) - 1, 100) # 最多计算 100 帧的延迟
            acf = np.correlate(series_norm, series_norm, mode='full')
            acf = acf[len(series_norm)-1 : len(series_norm)-1 + max_lag + 1] # 取单边
            acf /= acf[0] # 归一化
            plt.plot(np.arange(max_lag + 1) / fps, acf, label=f'像素 {coord}')

    plt.title(f'{video_name} - 像素点灰度值的时间自相关函数 (ACF)')
    plt.xlabel('时间延迟 (秒)')
    plt.ylabel('归一化自相关系数')
    plt.legend()
    plt.grid(True)
    save_plot(f'{video_name}_pixel_acf.png')

    # 4. 协方差 / PCA / 结构相似性
    if len(frames_rgb_list) > 0:
        sample_frame_rgb = frames_rgb_list[0] # 使用第一帧作为样本
        # 4a. RGB 通道协方差矩阵 (单帧示例)
        pixels_rgb = sample_frame_rgb.reshape(-1, 3) # (H*W, 3)
        cov_matrix_rgb = np.cov(pixels_rgb, rowvar=False)
        print(f"\n{video_name} - 第一帧 RGB 通道协方差矩阵:")
        print(cov_matrix_rgb)

        # 4b. PCA (基于少量帧的像素数据)
        print("\n执行 PCA (使用前几帧)...")
        start_time = time.time()
        # 将少量 RGB 帧数据扁平化用于 PCA
        # 注意：对整个视频进行 PCA 可能非常耗内存和时间
        pca_data_frames = [f.flatten() for f in frames_rgb_list]
        pca_data = np.array(pca_data_frames) # (num_sampled_frames, H*W*3)

        pca = PCA(n_components=PCA_N_COMPONENTS)
        pca.fit(pca_data)

        # 显示解释方差比例
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, PCA_N_COMPONENTS + 1), np.cumsum(pca.explained_variance_ratio_))
        plt.title(f'{video_name} - PCA 累积解释方差比例 (前 {len(frames_rgb_list)} 帧)')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比例')
        plt.xticks(np.arange(1, PCA_N_COMPONENTS + 1))
        plt.grid(True)
        save_plot(f'{video_name}_pca_explained_variance.png')
        print(f"  PCA 计算耗时: {time.time() - start_time:.2f} 秒")

        # 重构第一帧并比较
        transformed_frame = pca.transform(pca_data[0:1])
        reconstructed_frame_flat = pca.inverse_transform(transformed_frame)
        reconstructed_frame = reconstructed_frame_flat.reshape(sample_frame_rgb.shape).astype(np.uint8)

        # 保存原始帧和 PCA 重构帧
        pca_original_path = os.path.join(OUTPUT_DIR, f'{video_name}_pca_original_frame.png')
        pca_reconstructed_path = os.path.join(OUTPUT_DIR, f'{video_name}_pca_reconstructed_{PCA_N_COMPONENTS}comp_frame.png')
        cv2.imwrite(pca_original_path, sample_frame_rgb)
        cv2.imwrite(pca_reconstructed_path, reconstructed_frame)
        print(f"  PCA 原始帧已保存到: {pca_original_path}")
        print(f"  PCA 重构帧已保存到: {pca_reconstructed_path}")

        # 比较文件大小 (仅为示意，真实压缩需要保存为视频格式)
        original_size_kb = os.path.getsize(pca_original_path) / 1024
        reconstructed_size_kb = os.path.getsize(pca_reconstructed_path) / 1024
        print(f"  示例 PNG 文件大小: 原始 {original_size_kb:.2f} KB, 重构 {reconstructed_size_kb:.2f} KB")
        print(f"  (注意: 这只是单帧 PNG 大小对比，不完全代表视频压缩效果)")


        # 4c. 结构相似性 (SSIM) 示例 - 比较连续两帧
        if len(frames_gray_list) > 1:
             ssim_val, ssim_diff = ssim(frames_gray_list[0], frames_gray_list[1], full=True)
             print(f"\n{video_name} - 前两帧的结构相似性 (SSIM): {ssim_val:.4f}")
             plt.figure(figsize=(8, 6))
             plt.imshow(ssim_diff, cmap='gray')
             plt.title(f'{video_name} - 前两帧的 SSIM 差异图')
             plt.axis('off')
             save_plot(f'{video_name}_ssim_diff_frame0_1.png')


    # 5. 一维概率密度 (PDF) / 累积分布 (CDF) / 直方图均衡化
    sample_gray_frame = frames_gray_list[frame_num // 2] # 取中间一帧作为样本
    hist, bins = np.histogram(sample_gray_frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max() # 归一化以便于绘制

    # 绘制 PDF (直方图) 和 CDF
    plt.figure(figsize=(10, 5))
    plt.plot(cdf_normalized, color='b', label='CDF')
    plt.hist(sample_gray_frame.flatten(), 256, [0, 256], color='r', alpha=0.7, label='PDF (直方图)')
    plt.title(f'{video_name} - 中间帧灰度直方图 (PDF) 与累积分布函数 (CDF)')
    plt.xlabel('像素灰度值')
    plt.ylabel('频率 / 累积频率')
    plt.legend()
    plt.grid(True)
    save_plot(f'{video_name}_histogram_cdf.png')

    # 直方图均衡化
    equ_frame = cv2.equalizeHist(sample_gray_frame)

    # 绘制均衡化前后的图像及直方图
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(sample_gray_frame, cmap='gray')
    plt.title(f'{video_name} - 原始中间帧')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(equ_frame, cmap='gray')
    plt.title(f'{video_name} - 直方图均衡化后')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.hist(sample_gray_frame.flatten(), 256, [0, 256], color='r')
    plt.title('原始直方图')
    plt.xlabel('灰度值')
    plt.ylabel('频率')
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.hist(equ_frame.flatten(), 256, [0, 256], color='r')
    plt.title('均衡化后直方图')
    plt.xlabel('灰度值')
    plt.ylabel('频率')
    plt.grid(True)
    plt.tight_layout()
    save_plot(f'{video_name}_histogram_equalization.png')

    # 6. 二维概率密度 (联合 PDF)
    # 6a. 相邻像素 (水平方向)
    pixels_current = sample_gray_frame[:, :-1].flatten()
    pixels_right = sample_gray_frame[:, 1:].flatten()
    hist2d_adj, xedges_adj, yedges_adj = np.histogram2d(pixels_current, pixels_right, bins=64, range=[[0, 255], [0, 255]]) # 降低 bin 数量以便可视化

    plt.figure(figsize=(8, 7))
    plt.imshow(np.log1p(hist2d_adj), cmap='hot', origin='lower', extent=[xedges_adj[0], xedges_adj[-1], yedges_adj[0], yedges_adj[-1]])
    plt.colorbar(label='log(1 + 频率)')
    plt.xlabel('当前像素灰度值')
    plt.ylabel('右侧相邻像素灰度值')
    plt.title(f'{video_name} - 水平相邻像素联合灰度分布 (Log Scale)')
    save_plot(f'{video_name}_joint_pdf_adjacent.png')

    # 6b. RGB 不同通道之间 (使用第一帧彩色样本)
    if len(frames_rgb_list) > 0:
        sample_frame_rgb = frames_rgb_list[0]
        r_channel = sample_frame_rgb[:, :, 2].flatten() # BGR -> R is index 2
        g_channel = sample_frame_rgb[:, :, 1].flatten() # BGR -> G is index 1
        b_channel = sample_frame_rgb[:, :, 0].flatten() # BGR -> B is index 0

        plt.figure(figsize=(18, 5))
        # R vs G
        plt.subplot(1, 3, 1)
        hist2d_rg, xedges_rg, yedges_rg = np.histogram2d(r_channel, g_channel, bins=64, range=[[0, 255], [0, 255]])
        plt.imshow(np.log1p(hist2d_rg), cmap='hot', origin='lower', extent=[xedges_rg[0], xedges_rg[-1], yedges_rg[0], yedges_rg[-1]])
        plt.colorbar(label='log(1 + 频率)')
        plt.xlabel('R 通道值')
        plt.ylabel('G 通道值')
        plt.title('R-G 联合分布')
        # G vs B
        plt.subplot(1, 3, 2)
        hist2d_gb, xedges_gb, yedges_gb = np.histogram2d(g_channel, b_channel, bins=64, range=[[0, 255], [0, 255]])
        plt.imshow(np.log1p(hist2d_gb), cmap='hot', origin='lower', extent=[xedges_gb[0], xedges_gb[-1], yedges_gb[0], yedges_gb[-1]])
        plt.colorbar(label='log(1 + 频率)')
        plt.xlabel('G 通道值')
        plt.ylabel('B 通道值')
        plt.title('G-B 联合分布')
        # R vs B
        plt.subplot(1, 3, 3)
        hist2d_rb, xedges_rb, yedges_rb = np.histogram2d(r_channel, b_channel, bins=64, range=[[0, 255], [0, 255]])
        plt.imshow(np.log1p(hist2d_rb), cmap='hot', origin='lower', extent=[xedges_rb[0], xedges_rb[-1], yedges_rb[0], yedges_rb[-1]])
        plt.colorbar(label='log(1 + 频率)')
        plt.xlabel('R 通道值')
        plt.ylabel('B 通道值')
        plt.title('R-B 联合分布')

        plt.suptitle(f'{video_name} - RGB 通道间联合灰度分布 (Log Scale, 第一帧)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠
        save_plot(f'{video_name}_joint_pdf_rgb_channels.png')


    print(f"--- 完成分析视频: {video_path} ---")
    print(f"--- 输出结果保存在目录: {OUTPUT_DIR} ---")


# --- 主程序 ---
if __name__ == "__main__":
    for video_file in VIDEO_FILES:
        if os.path.exists(video_file):
            analyze_video(video_file)
        else:
            print(f"警告: 找不到视频文件 {video_file}，跳过分析。")

    print("\n--- 所有分析完成 ---")
