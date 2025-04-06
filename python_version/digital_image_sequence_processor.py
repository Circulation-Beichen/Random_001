import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.font_manager as fm
from functools import partial

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class ImageSequenceProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("数字图像序列处理实验程序")
        self.root.geometry("1200x800")
        
        self.video_paths = ["video_stay_dark.mp4", "video_move_light.mp4"]
        self.videos = []
        self.video_frames = []
        
        # 创建输出目录
        os.makedirs("processed_video", exist_ok=True)
        
        # 创建界面
        self.create_widgets()
        
        # 加载视频
        self.load_videos()
        
        # 在加载后立即进行所有处理并保存结果
        self.process_all_data()
    
    def create_widgets(self):
        # 创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # 创建各个标签页
        self.tab_videos = ttk.Frame(self.notebook)
        self.tab_mean = ttk.Frame(self.notebook)
        self.tab_variance = ttk.Frame(self.notebook)
        self.tab_correlation = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_videos, text="视频展示")
        self.notebook.add(self.tab_mean, text="均值分析")
        self.notebook.add(self.tab_variance, text="方差分析")
        self.notebook.add(self.tab_correlation, text="自相关分析")
        
        # 视频展示页面
        self.frame_video1 = ttk.LabelFrame(self.tab_videos, text="静态暗视频")
        self.frame_video1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.frame_video2 = ttk.LabelFrame(self.tab_videos, text="动态亮视频")
        self.frame_video2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_video1 = ttk.Label(self.frame_video1)
        self.lbl_video1.pack(padx=5, pady=5)
        
        self.lbl_video2 = ttk.Label(self.frame_video2)
        self.lbl_video2.pack(padx=5, pady=5)
        
        # 均值分析页面
        self.tab_mean_frame = ttk.Frame(self.tab_mean)
        self.tab_mean_frame.pack(expand=True, fill=tk.BOTH)
        
        # 方差分析页面
        self.tab_variance_frame = ttk.Frame(self.tab_variance)
        self.tab_variance_frame.pack(expand=True, fill=tk.BOTH)
        
        # 自相关分析页面
        self.tab_correlation_frame = ttk.Frame(self.tab_correlation)
        self.tab_correlation_frame.pack(expand=True, fill=tk.BOTH)
        
        # 添加显示结果按钮，而不是处理按钮
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_mean_time = ttk.Button(self.frame_buttons, text="显示时间域均值结果", command=self.display_mean_time)
        self.btn_mean_time.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_mean_space = ttk.Button(self.frame_buttons, text="显示空间域均值结果", command=self.display_mean_space)
        self.btn_mean_space.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_var_time = ttk.Button(self.frame_buttons, text="显示时间域方差结果", command=self.display_variance_time)
        self.btn_var_time.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_var_space = ttk.Button(self.frame_buttons, text="显示空间域方差结果", command=self.display_variance_space)
        self.btn_var_space.grid(row=0, column=3, padx=5, pady=5)
        
        self.btn_corr_time = ttk.Button(self.frame_buttons, text="显示时间域自相关结果", command=self.display_correlation_time)
        self.btn_corr_time.grid(row=0, column=4, padx=5, pady=5)
        
        self.btn_corr_space = ttk.Button(self.frame_buttons, text="显示空间域自相关结果", command=self.display_correlation_space)
        self.btn_corr_space.grid(row=0, column=5, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_videos(self):
        self.status_var.set("加载视频中...")
        self.root.update()
        
        for i, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.status_var.set(f"无法打开视频: {video_path}")
                continue
            
            self.videos.append(cap)
            
            # 提取帧
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            self.video_frames.append(frames)
            
            # 显示第一帧
            if i == 0:
                img = Image.fromarray(frames[0])
                img = img.resize((400, 300), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.lbl_video1.configure(image=img_tk)
                self.lbl_video1.image = img_tk
            else:
                img = Image.fromarray(frames[0])
                img = img.resize((400, 300), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.lbl_video2.configure(image=img_tk)
                self.lbl_video2.image = img_tk
        
        self.status_var.set(f"视频加载完成。第一个视频: {len(self.video_frames[0])}帧，第二个视频: {len(self.video_frames[1])}帧")
    
    def process_all_data(self):
        """在程序启动时处理所有数据并保存结果"""
        self.status_var.set("正在进行所有数据处理...")
        self.root.update()
        
        # 调用所有数据处理函数，但只保存结果，不显示在界面上
        self.process_mean_time()
        self.process_mean_space()
        self.process_variance_time()
        self.process_variance_space()
        self.process_correlation_time()
        self.process_correlation_space()
        
        self.status_var.set("所有数据处理完成，结果已保存在 processed_video 文件夹")
    
    def create_figure(self, parent, rows=1, cols=1):
        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        return fig, canvas
    
    def process_mean_time(self):
        """在时间域上统计均值，得到平均帧图像（只保存，不显示）"""
        # 计算两个视频的平均帧
        mean_frames = []
        for i, frames in enumerate(self.video_frames):
            mean_frame = np.mean(frames, axis=0).astype(np.uint8)
            mean_frames.append(mean_frame)
            
            # 保存结果
            output_path = f"processed_video/mean_time_video{i+1}.png"
            plt.imsave(output_path, mean_frame)
    
    def display_mean_time(self):
        """显示时间域均值分析结果"""
        # 清除旧图
        for widget in self.tab_mean_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_mean_frame, 1, 2)
        
        # 加载并显示保存的结果
        for i in range(2):
            img_path = f"processed_video/mean_time_video{i+1}.png"
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax = fig.add_subplot(1, 2, i+1)
                ax.imshow(img)
                ax.set_title(f"视频{i+1}的时间域均值帧")
                ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_mean)
    
    def process_mean_space(self):
        """在空间域上统计均值，得到时间均值曲线，闪烁检测（只保存，不显示）"""
        # 计算每帧的空间均值
        spatial_means = []
        for frames in self.video_frames:
            means = [np.mean(frame) for frame in frames]
            spatial_means.append(means)
        
        # 保存各视频均值曲线结果
        plt.figure(figsize=(10, 6))
        for i, means in enumerate(spatial_means):
            plt.plot(means, label=f"视频{i+1}")
        plt.title("各视频的空间域均值变化")
        plt.xlabel("帧")
        plt.ylabel("均值")
        plt.legend()
        plt.savefig("processed_video/mean_space_separate.png")
        plt.close()
        
        # 拼接两个视频进行闪烁检测
        combined_means = spatial_means[0] + spatial_means[1]
        
        # 检测闪烁（简单阈值法）
        mean_val = np.mean(combined_means)
        std_val = np.std(combined_means)
        threshold = std_val * 2
        flicker_points = []
        
        for i, mean in enumerate(combined_means):
            if abs(mean - mean_val) > threshold:
                flicker_points.append(i)
        
        # 保存拼接结果
        plt.figure(figsize=(10, 6))
        plt.plot(combined_means)
        for point in flicker_points:
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
        plt.title("拼接视频的空间域均值变化（闪烁检测）")
        plt.xlabel("帧")
        plt.ylabel("均值")
        plt.savefig("processed_video/mean_space_combined.png")
        plt.close()
    
    def display_mean_space(self):
        """显示空间域均值分析结果"""
        # 清除旧图
        for widget in self.tab_mean_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_mean_frame, 2, 1)
        
        # 加载并显示保存的结果
        if os.path.exists("processed_video/mean_space_separate.png"):
            img1 = plt.imread("processed_video/mean_space_separate.png")
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.imshow(img1)
            ax1.axis('off')
        
        if os.path.exists("processed_video/mean_space_combined.png"):
            img2 = plt.imread("processed_video/mean_space_combined.png")
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.imshow(img2)
            ax2.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_mean)
    
    def process_variance_time(self):
        """在时间轴上统计方差函数，进行运动区域检测（只保存，不显示）"""
        # 计算两个视频的时间方差
        var_frames = []
        for i, frames in enumerate(self.video_frames):
            var_frame = np.var(frames, axis=0).astype(np.uint8)
            var_frames.append(var_frame)
            
            # 保存结果
            output_path = f"processed_video/variance_time_video{i+1}.png"
            plt.imsave(output_path, var_frame, cmap='jet')
        
        # 特别处理第一个视频，标记运动区域
        var_threshold = np.mean(var_frames[0]) + np.std(var_frames[0]) * 2
        motion_mask = var_frames[0] > var_threshold
        
        # 显示带有运动区域标记的原始视频第一帧
        motion_frame = self.video_frames[0][0].copy()
        
        # 寻找轮廓（修复类型问题）
        motion_mask_uint8 = motion_mask.astype(np.uint8) * 255
        # 确保图像是单通道的灰度图
        if len(motion_mask_uint8.shape) > 2:
            motion_mask_uint8 = cv2.cvtColor(motion_mask_uint8, cv2.COLOR_RGB2GRAY)
            
        contours, _ = cv2.findContours(motion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(motion_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 保存运动区域结果
        plt.figure()
        plt.imshow(motion_frame)
        plt.title("静态暗视频中检测到的运动区域")
        plt.axis('off')
        plt.savefig("processed_video/motion_detection.png")
        plt.close()
    
    def display_variance_time(self):
        """显示时间域方差分析结果"""
        # 清除旧图
        for widget in self.tab_variance_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_variance_frame, 1, 2)
        
        # 加载并显示保存的结果
        for i in range(2):
            img_path = f"processed_video/variance_time_video{i+1}.png"
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax = fig.add_subplot(1, 2, i+1)
                ax.imshow(img)
                ax.set_title(f"视频{i+1}的时间域方差")
                ax.axis('off')
        
        # 添加运动检测结果
        if os.path.exists("processed_video/motion_detection.png"):
            new_fig = Figure(figsize=(8, 6), dpi=100)
            new_canvas = FigureCanvasTkAgg(new_fig, master=self.tab_variance_frame)
            new_ax = new_fig.add_subplot(1, 1, 1)
            
            motion_img = plt.imread("processed_video/motion_detection.png")
            new_ax.imshow(motion_img)
            new_ax.set_title("静态暗视频中检测到的运动区域")
            new_ax.axis('off')
            
            new_fig.tight_layout()
            new_canvas.draw()
            new_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_variance)
    
    def process_variance_space(self):
        """在空间上求方差函数来度量内容复杂度，探测镜头切换（只保存，不显示）"""
        # 计算每帧的空间方差
        spatial_variances = []
        for frames in self.video_frames:
            variances = [np.var(frame) for frame in frames]
            spatial_variances.append(variances)
        
        # 保存结果
        for i, variances in enumerate(spatial_variances):
            plt.figure(figsize=(10, 6))
            plt.plot(variances)
            
            # 检测镜头切换
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            threshold = mean_var + 2 * std_var
            
            scene_changes = []
            for j, var in enumerate(variances):
                if j > 0 and var > threshold and variances[j-1] < threshold:
                    scene_changes.append(j)
                    plt.axvline(x=j, color='r', linestyle='--')
            
            plt.title(f"视频{i+1}的空间域方差（检测到{len(scene_changes)}个镜头切换）")
            plt.xlabel("帧")
            plt.ylabel("方差")
            plt.savefig(f"processed_video/variance_space_video{i+1}.png")
            plt.close()
    
    def display_variance_space(self):
        """显示空间域方差分析结果"""
        # 清除旧图
        for widget in self.tab_variance_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_variance_frame, 2, 1)
        
        # 加载并显示保存的结果
        for i in range(2):
            img_path = f"processed_video/variance_space_video{i+1}.png"
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax = fig.add_subplot(2, 1, i+1)
                ax.imshow(img)
                ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_variance)
    
    def process_correlation_time(self):
        """在时间轴上计算自相关函数，体现场景变换特性（只保存，不显示）"""
        # 对每个视频计算时间域自相关
        for i, frames in enumerate(self.video_frames):
            # 选择帧数量的一半作为最大延迟
            max_lag = min(30, len(frames) // 2)
            
            # 使用中心点像素值计算自相关
            h, w, _ = frames[0].shape
            center_values = [frame[h//2, w//2, 0] for frame in frames]  # 使用红色通道
            
            # 计算自相关
            autocorr = np.correlate(center_values, center_values, mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + max_lag+1]
            autocorr = autocorr / autocorr[0]  # 归一化
            
            # 保存结果
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(autocorr)), autocorr)
            plt.title(f"视频{i+1}的时间域自相关函数")
            plt.xlabel("延迟")
            plt.ylabel("自相关系数")
            plt.savefig(f"processed_video/correlation_time_video{i+1}.png")
            plt.close()
    
    def display_correlation_time(self):
        """显示时间域自相关分析结果"""
        # 清除旧图
        for widget in self.tab_correlation_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_correlation_frame, 2, 1)
        
        # 加载并显示保存的结果
        for i in range(2):
            img_path = f"processed_video/correlation_time_video{i+1}.png"
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax = fig.add_subplot(2, 1, i+1)
                ax.imshow(img)
                ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_correlation)
    
    def process_correlation_space(self):
        """在空间上计算自相关函数，体现纹理的空间重复性（只保存，不显示）"""
        # 对每个视频的第一帧计算空间自相关
        for i, frames in enumerate(self.video_frames):
            # 选取第一帧
            frame = frames[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 计算2D自相关
            result = cv2.matchTemplate(gray, gray, cv2.TM_CCORR_NORMED)
            
            # 保存结果
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(frame)
            plt.title(f"视频{i+1}第一帧")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(result, cmap='jet')
            plt.title(f"视频{i+1}的空间自相关")
            plt.axis('off')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f"processed_video/correlation_space_video{i+1}.png")
            plt.close()
    
    def display_correlation_space(self):
        """显示空间域自相关分析结果"""
        # 清除旧图
        for widget in self.tab_correlation_frame.winfo_children():
            widget.destroy()
        
        fig, canvas = self.create_figure(self.tab_correlation_frame, 2, 1)
        
        # 加载并显示保存的结果
        for i in range(2):
            img_path = f"processed_video/correlation_space_video{i+1}.png"
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax = fig.add_subplot(2, 1, i+1)
                ax.imshow(img)
                ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tab_correlation)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSequenceProcessor(root)
    root.mainloop() 