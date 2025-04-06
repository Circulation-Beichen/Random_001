import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import time
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from scipy.stats import linregress
# 导入 Analyze_withNoise 中的函数
from Analyze_withNoise import add_gaussian_noise, add_salt_pepper_noise, calculate_stats

# --- 配置 Matplotlib 支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams.update({'font.size': 10}) # 调整字体大小
# ---

# --- 从 Analyze_withoutNoise.py 引入或重定义的参数和函数 ---
OUTPUT_DIR = 'video_analysis_output_gui'
PCA_N_COMPONENTS = 5
MOTION_VAR_THRESHOLD = 500
ACF_PIXEL_COORDS = [(50, 50), (100, 150)]

# 噪声分析输出目录
OUTPUT_DIR_NOISE = 'noise_analysis_output_gui' # 使用不同的目录以防与脚本冲突
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR_NOISE):
    os.makedirs(OUTPUT_DIR_NOISE)

# --- GUI 应用类 ---
class VideoAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频统计分析 GUI")
        self.root.geometry("1000x750")

        self.video_paths = ['video_move_light.mp4', 'video_stay_dark.mp4']
        self.current_video_path = None
        self.video_data = {}

        self.noise_params = {
            'gaussian_var': tk.DoubleVar(value=0.01),
            'salt_pepper_amount': tk.DoubleVar(value=0.04)
        }
        self.noise_type_var = tk.StringVar(value="高斯")
        self.noise_frame_index_var = tk.IntVar(value=50)
        self.selected_noise_video_path = tk.StringVar()

        self.create_widgets()
        self.load_all_video_data()
        # 设置默认选择的噪声分析视频
        if self.video_paths:
            self.selected_noise_video_path.set(self.video_paths[0])

    def create_widgets(self):
        # --- Top Frame: Video Selection ---
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="选择视频:").pack(side=tk.LEFT, padx=(0, 5))
        self.video_combobox = ttk.Combobox(top_frame, values=self.video_paths, state="readonly", width=30)
        self.video_combobox.pack(side=tk.LEFT, padx=5)
        self.video_combobox.bind("<<ComboboxSelected>>", self.on_video_selected)
        if self.video_paths:
            self.video_combobox.current(0)
            self.current_video_path = self.video_paths[0]

        # --- Main Frame: Tabs for Analysis ---
        self.notebook = ttk.Notebook(self.root, padding="5")
        self.notebook.pack(expand=True, fill=tk.BOTH)

        # 创建各个分析的标签页和画布区域
        self.tabs = {}
        self.canvases = {}
        self.figures = {}
        analysis_types = [
            "空间均值 (亮度)", "时间均值 (平均帧)", "空间方差 (复杂度)",
            "时间方差 (运动)", "运动检测框", "像素ACF", "PCA方差",
            "PCA重构", "SSIM差异", "灰度PDF/CDF", "直方图均衡化",
            "相邻像素联合PDF", "RGB联合PDF",
            "平稳性分析", "各态历经性估计"
        ]

        for name in analysis_types:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=name)
            self.tabs[name] = tab
            self.figures[name] = None
            self.canvases[name] = None

        # --- 添加噪声分析标签页 ---
        self.tab_noise = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_noise, text="噪声分析")
        self.tabs["噪声分析"] = self.tab_noise
        self.figures["噪声分析"] = None
        self.canvases["噪声分析"] = None

        # 噪声分析控制框架
        noise_control_frame = ttk.LabelFrame(self.tab_noise, text="噪声参数与控制", padding="10")
        noise_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 选择视频进行噪声分析
        ttk.Label(noise_control_frame, text="选择视频:").grid(row=0, column=0, sticky=tk.W, pady=2)
        noise_video_combo = ttk.Combobox(noise_control_frame, textvariable=self.selected_noise_video_path,
                                         values=self.video_paths, state="readonly", width=25)
        noise_video_combo.grid(row=0, column=1, sticky=tk.EW, pady=2)

        # 选择帧索引
        ttk.Label(noise_control_frame, text="帧索引:").grid(row=1, column=0, sticky=tk.W, pady=2)
        noise_frame_spinbox = ttk.Spinbox(noise_control_frame, from_=0, to=1000, # 设定一个上限
                                          textvariable=self.noise_frame_index_var, width=8)
        noise_frame_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2)

        # 选择噪声类型
        ttk.Label(noise_control_frame, text="噪声类型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        noise_type_combo = ttk.Combobox(noise_control_frame, textvariable=self.noise_type_var,
                                        values=["高斯", "椒盐"], state="readonly", width=10)
        noise_type_combo.grid(row=2, column=1, sticky=tk.W, pady=2)

        # 高斯噪声参数
        gaussian_frame = ttk.Frame(noise_control_frame)
        gaussian_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=5)
        ttk.Label(gaussian_frame, text="高斯方差:").pack(side=tk.LEFT, padx=(0,5))
        gaussian_scale = ttk.Scale(gaussian_frame, from_=0.001, to=0.1, orient=tk.HORIZONTAL,
                                   variable=self.noise_params['gaussian_var'], length=150)
        gaussian_scale.pack(side=tk.LEFT, expand=True, fill=tk.X)
        gaussian_label = ttk.Label(gaussian_frame, textvariable=self.noise_params['gaussian_var'], width=5)
        gaussian_label.pack(side=tk.LEFT)
        self.noise_params['gaussian_var'].trace_add("write", lambda *args: gaussian_label.config(text=f"{self.noise_params['gaussian_var'].get():.3f}"))

        # 椒盐噪声参数
        sp_frame = ttk.Frame(noise_control_frame)
        sp_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        ttk.Label(sp_frame, text="椒盐比例:").pack(side=tk.LEFT, padx=(0,5))
        sp_scale = ttk.Scale(sp_frame, from_=0.01, to=0.2, orient=tk.HORIZONTAL,
                             variable=self.noise_params['salt_pepper_amount'], length=150)
        sp_scale.pack(side=tk.LEFT, expand=True, fill=tk.X)
        sp_label = ttk.Label(sp_frame, textvariable=self.noise_params['salt_pepper_amount'], width=5)
        sp_label.pack(side=tk.LEFT)
        self.noise_params['salt_pepper_amount'].trace_add("write", lambda *args: sp_label.config(text=f"{self.noise_params['salt_pepper_amount'].get():.3f}"))

        # 运行噪声分析按钮
        run_noise_button = ttk.Button(noise_control_frame, text="运行噪声分析", command=self.run_noise_analysis)
        run_noise_button.grid(row=5, column=0, columnspan=2, pady=15)

        # 噪声分析结果显示区域 (画布将放在这里)
        self.noise_display_frame = ttk.Frame(self.tab_noise)
        self.noise_display_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # --- Bottom Frame: Status Bar and Buttons ---
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)

        self.run_analysis_button = ttk.Button(bottom_frame, text="运行当前视频分析", command=self.run_current_video_analysis)
        self.run_analysis_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(bottom_frame, text="保存当前图表", command=self.save_current_figure)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_var.set("就绪。请选择视频并运行分析。")

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_video_data(self, video_path):
        """加载单个视频文件的帧数据"""
        if video_path in self.video_data and self.video_data[video_path]:
            self.set_status(f"视频 {os.path.basename(video_path)} 数据已加载。")
            return True

        self.set_status(f"正在加载视频: {os.path.basename(video_path)}...")
        if not os.path.exists(video_path):
            messagebox.showerror("错误", f"视频文件未找到: {video_path}")
            self.set_status(f"错误：找不到视频文件 {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频文件: {video_path}")
            self.set_status(f"错误：无法打开视频 {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_gray_list = []
        frames_rgb_list = []
        pixel_time_series = {coord: [] for coord in ACF_PIXEL_COORDS}
        spatial_means = []
        spatial_vars = []

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_gray_list.append(gray_frame)
            if frame_num < PCA_N_COMPONENTS: # 存储用于 PCA 的帧
                 frames_rgb_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            spatial_means.append(np.mean(gray_frame))
            spatial_vars.append(np.var(gray_frame))

            for coord in ACF_PIXEL_COORDS:
                if 0 <= coord[0] < height and 0 <= coord[1] < width:
                     pixel_time_series[coord].append(gray_frame[coord[0], coord[1]])

            frame_num += 1
            if frame_num % 100 == 0:
                 self.set_status(f"加载 {os.path.basename(video_path)}: {frame_num}/{frame_count} 帧...")

        cap.release()

        if frame_num == 0:
            messagebox.showerror("错误", f"未能从视频读取任何帧: {video_path}")
            self.set_status(f"错误：未能读取 {video_path} 的帧")
            return False

        self.video_data[video_path] = {
            'frames_gray': np.array(frames_gray_list),
            'frames_rgb_pca': frames_rgb_list, # 只存少量 RGB 用于 PCA
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_num,
            'spatial_means': np.array(spatial_means),
            'spatial_vars': np.array(spatial_vars),
            'pixel_time_series': pixel_time_series
        }
        self.set_status(f"视频 {os.path.basename(video_path)} 加载完成 ({frame_num} 帧)。")
        return True

    def load_all_video_data(self):
        """加载所有视频数据"""
        all_loaded = True
        for path in self.video_paths:
            if not self.load_video_data(path):
                all_loaded = False
        if all_loaded:
             self.set_status("所有视频数据加载完成。请选择视频并运行分析。")
        else:
             self.set_status("部分视频加载失败。请检查文件路径。")

    def on_video_selected(self, event=None):
        self.current_video_path = self.video_combobox.get()
        self.set_status(f"已选择视频: {os.path.basename(self.current_video_path)}. 点击按钮运行分析。")
        # 清除旧的图表
        for name in self.canvases:
            if self.canvases[name]:
                self.canvases[name].get_tk_widget().destroy()
                # 显式销毁工具栏
                for widget in self.tabs[name].winfo_children():
                    if isinstance(widget, tk.Frame): # NavigationToolbar2Tk is inside a Frame
                        widget.destroy()
            self.canvases[name] = None
            self.figures[name] = None

    def run_current_video_analysis(self):
        if not self.current_video_path:
            messagebox.showwarning("提示", "请先选择一个视频。")
            return

        if self.current_video_path not in self.video_data or not self.video_data[self.current_video_path]:
            self.set_status(f"正在加载视频数据: {os.path.basename(self.current_video_path)}...")
            if not self.load_video_data(self.current_video_path):
                 messagebox.showerror("错误", f"无法加载视频数据: {self.current_video_path}")
                 return
            self.set_status(f"视频数据加载完成。开始分析...")
        else:
            self.set_status(f"开始分析视频: {os.path.basename(self.current_video_path)}...")

        data = self.video_data[self.current_video_path]
        video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]

        analysis_functions = {
            "空间均值 (亮度)": self.plot_spatial_mean,
            "时间均值 (平均帧)": self.plot_temporal_mean,
            "空间方差 (复杂度)": self.plot_spatial_variance,
            "时间方差 (运动)": self.plot_temporal_variance,
            "运动检测框": self.plot_motion_detection,
            "像素ACF": self.plot_pixel_acf,
            "PCA方差": self.plot_pca_variance,
            "PCA重构": self.plot_pca_reconstruction,
            "SSIM差异": self.plot_ssim_difference,
            "灰度PDF/CDF": self.plot_histogram_cdf,
            "直方图均衡化": self.plot_histogram_equalization,
            "相邻像素联合PDF": self.plot_joint_pdf_adjacent,
            "RGB联合PDF": self.plot_joint_pdf_rgb,
            "平稳性分析": self.plot_stationarity_analysis,
            "各态历经性估计": self.plot_ergodicity_analysis
        }

        # 逐个执行分析并显示
        for name, func in analysis_functions.items():
            self.set_status(f"正在计算: {name}...")
            try:
                fig = func(data, video_name)
                if fig:
                    self.display_figure(fig, name)
                else:
                    # 清理可能存在的旧画布
                    if self.canvases[name]:
                        self.canvases[name].get_tk_widget().destroy()
                        for widget in self.tabs[name].winfo_children():
                            if isinstance(widget, tk.Frame):
                                widget.destroy()
                        self.canvases[name] = None
                        self.figures[name] = None
                    # 添加提示标签
                    label = ttk.Label(self.tabs[name], text=f"{name} 分析不适用于或无法计算。")
                    label.pack(padx=20, pady=20)

            except Exception as e:
                print(f"计算 {name} 时出错: {e}")
                messagebox.showerror("分析错误", f"计算 {name} 时出错:\n{e}")
                # 添加错误提示标签
                if self.canvases[name]:
                    self.canvases[name].get_tk_widget().destroy()
                    for widget in self.tabs[name].winfo_children():
                        if isinstance(widget, tk.Frame):
                           widget.destroy()
                    self.canvases[name] = None
                    self.figures[name] = None
                label = ttk.Label(self.tabs[name], text=f"计算 {name} 时出错。")
                label.pack(padx=20, pady=20)


        self.set_status(f"视频 {os.path.basename(self.current_video_path)} 分析完成。")

    def display_figure(self, fig, tab_name, target_frame=None):
        """在指定的标签页或框架中显示 Matplotlib 图表 (更新以支持 target_frame)"""
        # 如果未指定目标框架，则使用标签页
        if target_frame is None:
            tab = self.tabs.get(tab_name)
            if tab is None:
                print(f"错误: 找不到标签页 '{tab_name}'")
                return
            target_widget = tab
        else:
            target_widget = target_frame

        # 清除旧的画布和工具栏 (在目标 widget 内查找)
        canvas_to_destroy = self.canvases.get(tab_name) # 获取与此标签名关联的旧画布
        if canvas_to_destroy and canvas_to_destroy.get_tk_widget().master == target_widget:
             canvas_to_destroy.get_tk_widget().pack_forget()
             canvas_to_destroy.get_tk_widget().destroy()
             # 尝试查找并销毁对应的工具栏 Frame
             for widget in target_widget.winfo_children():
                 if isinstance(widget, tk.Frame) and isinstance(widget.winfo_children()[0], NavigationToolbar2Tk):
                     widget.destroy()
                     break # 假设只有一个工具栏

        # 清除可能存在的提示标签
        for widget in target_widget.winfo_children():
            if isinstance(widget, ttk.Label) and widget.winfo_class() == 'TLabel':
                widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=target_widget)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建工具栏，并将其放入一个单独的 Frame 中以便管理
        toolbar_frame = ttk.Frame(target_widget)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # 更新 figure 和 canvas 引用
        self.figures[tab_name] = fig
        self.canvases[tab_name] = canvas

    def save_current_figure(self):
        # ... (保存逻辑基本不变，但需要考虑噪声分析标签页)
        current_tab_index = self.notebook.index(self.notebook.select())
        tab_name = self.notebook.tab(current_tab_index, "text")

        if tab_name in self.figures and self.figures[tab_name]:
            fig = self.figures[tab_name]
            # 获取文件名基础部分
            if tab_name == "噪声分析":
                base_name = os.path.splitext(os.path.basename(self.selected_noise_video_path.get()))[0]
                noise_suffix = self.noise_type_var.get().lower()
                frame_suffix = f"frame{self.noise_frame_index_var.get()}"
                default_filename_base = f"{base_name}_{frame_suffix}_noise_{noise_suffix}"
            elif self.current_video_path:
                 default_filename_base = os.path.splitext(os.path.basename(self.current_video_path))[0]
                 safe_tab_name = tab_name.replace("/", "_").replace(" ", "_")
                 default_filename_base += f"_{safe_tab_name}"
            else:
                 default_filename_base = f"analysis_{tab_name.replace(' ', '_')}"

            default_filename = f"{default_filename_base}.png"
            output_dir_to_use = OUTPUT_DIR_NOISE if tab_name == "噪声分析" else OUTPUT_DIR

            filepath = filedialog.asksaveasfilename(
                parent=self.root,
                title="保存图表",
                initialdir=output_dir_to_use,
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[("PNG 文件", "*.png"), ("JPEG 文件", "*.jpg"), ("所有文件", "*.*")]
            )
            if filepath:
                 try:
                     fig.savefig(filepath, dpi=150)
                     self.set_status(f"图表已保存到: {filepath}")
                     messagebox.showinfo("保存成功", f"图表已保存到:\n{filepath}")
                 except Exception as e:
                     messagebox.showerror("保存失败", f"无法保存图表:\n{e}")
                     self.set_status(f"错误：无法保存图表 {filepath}")
        else:
             messagebox.showwarning("提示", "当前标签页没有可保存的图表或未选择主分析视频。")

    # --- 分析绘图函数 (从 analyze_video 修改而来) ---
    # 每个函数接收 data 字典和 video_name，返回 matplotlib Figure 对象

    def plot_spatial_mean(self, data, video_name):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(data['frame_count']) / data['fps'], data['spatial_means'])
        ax.set_title(f'{video_name} - 帧平均亮度随时间变化')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('平均像素值')
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_temporal_mean(self, data, video_name):
        temporal_mean_frame = np.mean(data['frames_gray'], axis=0).astype(np.uint8)
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(temporal_mean_frame, cmap='gray')
        ax.set_title(f'{video_name} - 时间平均帧')
        ax.axis('off')
        fig.tight_layout()
        return fig

    def plot_spatial_variance(self, data, video_name):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(data['frame_count']) / data['fps'], data['spatial_vars'])
        ax.set_title(f'{video_name} - 帧内方差 (复杂度/镜头移动指示)')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('像素值方差')
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_temporal_variance(self, data, video_name):
        temporal_variance_frame = np.var(data['frames_gray'], axis=0)
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(temporal_variance_frame, cmap='hot')
        ax.set_title(f'{video_name} - 像素时间方差')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def plot_motion_detection(self, data, video_name):
        temporal_variance_frame = np.var(data['frames_gray'], axis=0)
        motion_mask = (temporal_variance_frame > MOTION_VAR_THRESHOLD).astype(np.uint8) * 255
        temporal_mean_frame = np.mean(data['frames_gray'], axis=0).astype(np.uint8)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 使用第一帧彩色图（如果加载了）或者时间平均灰度图作为背景
        if data['frames_rgb_pca']:
            background_frame_color = data['frames_rgb_pca'][0].copy() # 使用加载的第一帧RGB
        else:
            # 如果没加载RGB，用平均灰度图转 BGR
            background_frame_color = cv2.cvtColor(temporal_mean_frame, cv2.COLOR_GRAY2BGR)

        frame_with_boxes = background_frame_color
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2) # BGR Red box

        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)) # Convert back to RGB for display
        ax.set_title(f'{video_name} - 检测到的运动区域 (红框)')
        ax.axis('off')
        fig.tight_layout()
        return fig

    def plot_pixel_acf(self, data, video_name):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        for coord, series in data['pixel_time_series'].items():
            if len(series) > 1:
                series_norm = series - np.mean(series)
                max_lag = min(len(series_norm) - 1, 100)
                acf = np.correlate(series_norm, series_norm, mode='full')
                acf = acf[len(series_norm)-1 : len(series_norm)-1 + max_lag + 1]
                if acf[0] != 0:
                     acf /= acf[0]
                else: # Handle zero variance case
                     acf = np.zeros_like(acf)
                ax.plot(np.arange(max_lag + 1) / data['fps'], acf, label=f'像素 {coord}')

        ax.set_title(f'{video_name} - 像素点灰度值的时间自相关函数 (ACF)')
        ax.set_xlabel('时间延迟 (秒)')
        ax.set_ylabel('归一化自相关系数')
        if data['pixel_time_series']: # Only show legend if there's data
             ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_pca_variance(self, data, video_name):
        if not data['frames_rgb_pca']:
             print(f"警告：{video_name} 没有加载足够的 RGB 帧用于 PCA 分析。")
             return None # 返回 None 表示无法生成图表

        pca_data_frames = [f.flatten() for f in data['frames_rgb_pca']]
        pca_data = np.array(pca_data_frames)

        n_samples = pca_data.shape[0]
        if n_samples == 0:
            print(f"警告：{video_name} PCA 数据为空。")
            return None

        # 动态调整 n_components
        actual_n_components = min(PCA_N_COMPONENTS, n_samples)
        if actual_n_components == 0: # No samples to fit PCA
            print(f"警告：{video_name} 没有样本用于 PCA。")
            return None

        pca = PCA(n_components=actual_n_components)
        try:
             pca.fit(pca_data)
        except ValueError as e:
             print(f"PCA 拟合时出错 for {video_name}: {e}")
             messagebox.showerror("PCA 错误", f"视频 {video_name} PCA 计算出错: {e}")
             return None

        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1, actual_n_components + 1), np.cumsum(pca.explained_variance_ratio_))
        ax.set_title(f'{video_name} - PCA 累积解释方差 (前 {n_samples} 帧)')
        ax.set_xlabel('主成分数量')
        ax.set_ylabel('累积解释方差比例')
        if actual_n_components > 0:
            ax.set_xticks(np.arange(1, actual_n_components + 1))
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_pca_reconstruction(self, data, video_name):
        if not data['frames_rgb_pca']:
            print(f"警告：{video_name} 没有加载足够的 RGB 帧用于 PCA 重构。")
            return None

        pca_data_frames = [f.flatten() for f in data['frames_rgb_pca']]
        pca_data = np.array(pca_data_frames)

        n_samples = pca_data.shape[0]
        if n_samples == 0:
            print(f"警告：{video_name} PCA 数据为空。")
            return None

        actual_n_components = min(PCA_N_COMPONENTS, n_samples)
        if actual_n_components == 0:
             print(f"警告：{video_name} 没有样本用于 PCA。")
             return None

        pca = PCA(n_components=actual_n_components)
        try:
             pca.fit(pca_data)
             transformed_frame = pca.transform(pca_data[0:1])
             reconstructed_frame_flat = pca.inverse_transform(transformed_frame)
             original_shape = data['frames_rgb_pca'][0].shape
             reconstructed_frame = reconstructed_frame_flat.reshape(original_shape).astype(np.uint8)
             original_frame = data['frames_rgb_pca'][0]
        except ValueError as e:
             print(f"PCA 变换/重构时出错 for {video_name}: {e}")
             messagebox.showerror("PCA 错误", f"视频 {video_name} PCA 重构出错: {e}")
             return None

        fig = Figure(figsize=(10, 5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax1.imshow(original_frame)
        ax1.set_title('原始第一帧')
        ax1.axis('off')

        ax2 = fig.add_subplot(122)
        ax2.imshow(reconstructed_frame)
        ax2.set_title(f'PCA 重构 ({actual_n_components} 分量)')
        ax2.axis('off')

        fig.suptitle(f'{video_name} - PCA 重构对比')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_ssim_difference(self, data, video_name):
        if data['frame_count'] < 2:
             print(f"警告：{video_name} 帧数不足，无法计算 SSIM。")
             return None
        frame1 = data['frames_gray'][0]
        frame2 = data['frames_gray'][1]
        ssim_val, ssim_diff = ssim(frame1, frame2, full=True)

        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(ssim_diff, cmap='gray')
        ax.set_title(f'{video_name} - 前两帧 SSIM 差异图 (SSIM: {ssim_val:.4f})')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def plot_histogram_cdf(self, data, video_name):
        if data['frame_count'] == 0:
             return None
        sample_gray_frame = data['frames_gray'][data['frame_count'] // 2]
        hist, bins = np.histogram(sample_gray_frame.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        if cdf.max() > 0:
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
        else:
            cdf_normalized = np.zeros_like(cdf)

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(cdf_normalized, color='b', label='CDF')
        ax.hist(sample_gray_frame.flatten(), 256, [0, 256], color='r', alpha=0.7, label='PDF (直方图)')
        ax.set_title(f'{video_name} - 中间帧灰度 PDF 与 CDF')
        ax.set_xlabel('像素灰度值')
        ax.set_ylabel('频率 / 累积频率')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_histogram_equalization(self, data, video_name):
        if data['frame_count'] == 0:
            return None
        sample_gray_frame = data['frames_gray'][data['frame_count'] // 2]
        equ_frame = cv2.equalizeHist(sample_gray_frame)

        fig = Figure(figsize=(10, 8), dpi=100)
        # Image plots
        ax1 = fig.add_subplot(221)
        ax1.imshow(sample_gray_frame, cmap='gray')
        ax1.set_title('原始中间帧')
        ax1.axis('off')
        ax2 = fig.add_subplot(222)
        ax2.imshow(equ_frame, cmap='gray')
        ax2.set_title('直方图均衡化后')
        ax2.axis('off')
        # Histogram plots
        ax3 = fig.add_subplot(223)
        ax3.hist(sample_gray_frame.flatten(), 256, [0, 256], color='r')
        ax3.set_title('原始直方图')
        ax3.set_xlabel('灰度值')
        ax3.set_ylabel('频率')
        ax3.grid(True)
        ax4 = fig.add_subplot(224)
        ax4.hist(equ_frame.flatten(), 256, [0, 256], color='r')
        ax4.set_title('均衡化后直方图')
        ax4.set_xlabel('灰度值')
        ax4.set_ylabel('频率')
        ax4.grid(True)

        fig.suptitle(f'{video_name} - 直方图均衡化效果')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_joint_pdf_adjacent(self, data, video_name):
        if data['frame_count'] == 0:
            return None
        sample_gray_frame = data['frames_gray'][data['frame_count'] // 2]
        if sample_gray_frame.shape[1] < 2: # Need at least 2 columns
            return None

        pixels_current = sample_gray_frame[:, :-1].flatten()
        pixels_right = sample_gray_frame[:, 1:].flatten()
        hist2d_adj, xedges_adj, yedges_adj = np.histogram2d(pixels_current, pixels_right, bins=64, range=[[0, 255], [0, 255]])

        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log1p(hist2d_adj), cmap='hot', origin='lower',
                       extent=[xedges_adj[0], xedges_adj[-1], yedges_adj[0], yedges_adj[-1]], aspect='auto')
        fig.colorbar(im, ax=ax, label='log(1 + 频率)')
        ax.set_xlabel('当前像素灰度值')
        ax.set_ylabel('右侧相邻像素灰度值')
        ax.set_title(f'{video_name} - 水平相邻像素联合灰度分布')
        fig.tight_layout()
        return fig

    def plot_joint_pdf_rgb(self, data, video_name):
        if not data['frames_rgb_pca']: # Need RGB frames
             print(f"警告：{video_name} 没有加载 RGB 帧用于联合 PDF 分析。")
             return None

        sample_frame_rgb = data['frames_rgb_pca'][0]
        r_channel = sample_frame_rgb[:, :, 0].flatten() # RGB -> R is index 0
        g_channel = sample_frame_rgb[:, :, 1].flatten() # RGB -> G is index 1
        b_channel = sample_frame_rgb[:, :, 2].flatten() # RGB -> B is index 2

        fig = Figure(figsize=(15, 5), dpi=100)
        # R vs G
        ax1 = fig.add_subplot(131)
        hist2d_rg, xedges_rg, yedges_rg = np.histogram2d(r_channel, g_channel, bins=64, range=[[0, 255], [0, 255]])
        im1 = ax1.imshow(np.log1p(hist2d_rg), cmap='hot', origin='lower', extent=[xedges_rg[0], xedges_rg[-1], yedges_rg[0], yedges_rg[-1]], aspect='auto')
        fig.colorbar(im1, ax=ax1, label='log(1 + 频率)')
        ax1.set_xlabel('R 通道值')
        ax1.set_ylabel('G 通道值')
        ax1.set_title('R-G 联合分布')
        # G vs B
        ax2 = fig.add_subplot(132)
        hist2d_gb, xedges_gb, yedges_gb = np.histogram2d(g_channel, b_channel, bins=64, range=[[0, 255], [0, 255]])
        im2 = ax2.imshow(np.log1p(hist2d_gb), cmap='hot', origin='lower', extent=[xedges_gb[0], xedges_gb[-1], yedges_gb[0], yedges_gb[-1]], aspect='auto')
        fig.colorbar(im2, ax=ax2, label='log(1 + 频率)')
        ax2.set_xlabel('G 通道值')
        ax2.set_ylabel('B 通道值')
        ax2.set_title('G-B 联合分布')
        # R vs B
        ax3 = fig.add_subplot(133)
        hist2d_rb, xedges_rb, yedges_rb = np.histogram2d(r_channel, b_channel, bins=64, range=[[0, 255], [0, 255]])
        im3 = ax3.imshow(np.log1p(hist2d_rb), cmap='hot', origin='lower', extent=[xedges_rb[0], xedges_rb[-1], yedges_rb[0], yedges_rb[-1]], aspect='auto')
        fig.colorbar(im3, ax=ax3, label='log(1 + 频率)')
        ax3.set_xlabel('R 通道值')
        ax3.set_ylabel('B 通道值')
        ax3.set_title('R-B 联合分布')

        fig.suptitle(f'{video_name} - RGB 通道间联合灰度分布 (Log Scale, 第一帧)', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_stationarity_analysis(self, data, video_name):
        """分析空间均值的时间变化以判断平稳性"""
        spatial_means = data['spatial_means']
        fps = data['fps']
        time_axis = np.arange(data['frame_count']) / fps

        if len(spatial_means) < 2:
            return None # 无法计算差分

        # 计算一阶差分
        mean_diff = np.diff(spatial_means)
        # 计算差分的绝对值的平均值，作为变化速度指标
        change_speed = np.mean(np.abs(mean_diff)) * fps # 单位: 亮度单位/秒

        fig = Figure(figsize=(9, 6), dpi=100)

        # 绘制原始空间均值
        ax1 = fig.add_subplot(211)
        ax1.plot(time_axis, spatial_means, label='空间均值')
        ax1.set_title(f'{video_name} - 空间均值随时间变化')
        ax1.set_ylabel('平均像素值')
        ax1.grid(True)
        ax1.legend()

        # 绘制一阶差分（变化率）
        ax2 = fig.add_subplot(212)
        # 注意时间轴需要对齐差分数据
        ax2.plot(time_axis[1:], mean_diff * fps, label='均值变化率 (差分)', color='orange')
        ax2.set_title(f'空间均值变化率 (大致镜头改变速度估计: {change_speed:.2f} /秒)')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('均值变化率 (值/秒)')
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        return fig

    def plot_ergodicity_analysis(self, data, video_name):
        """根据空间均值波动估计各态历经性"""
        spatial_means = data['spatial_means']

        if len(spatial_means) < 2:
             # 无法可靠估计，显示提示信息
             fig = Figure(figsize=(6, 3), dpi=100)
             ax = fig.add_subplot(111)
             ax.text(0.5, 0.5, '帧数不足，无法估计各态历经性',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='red')
             ax.axis('off')
             fig.tight_layout()
             return fig

        overall_mean = np.mean(spatial_means)
        std_dev = np.std(spatial_means)
        threshold = 0.03 * overall_mean # 3% 阈值

        is_ergodic = std_dev <= threshold

        # 修正 f-string 语法错误
        ergodicity_status = '具有各态历经性 (波动 <= 3%)' if is_ergodic else '不具有各态历经性 (波动 > 3%)'
        result_text = f"空间均值标准差: {std_dev:.3f}\n"
        result_text += f"整体均值: {overall_mean:.3f}\n"
        result_text += f"3% 波动阈值: {threshold:.3f}\n\n"
        result_text += f"结论: {ergodicity_status}" # 使用变量

        text_color = 'green' if is_ergodic else 'red' # 颜色可以稍后用于文本，但 text() 不直接支持

        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        # 在图表中显示文本结果
        ax.text(0.5, 0.5, result_text,
                horizontalalignment='center', verticalalignment='center',
                fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        ax.set_title(f'{video_name} - 各态历经性估计 (基于空间均值波动)')
        ax.axis('off') # 不显示坐标轴
        fig.tight_layout()
        return fig

    def run_noise_analysis(self):
        """执行单帧噪声分析并在 GUI 中显示结果"""
        video_path = self.selected_noise_video_path.get()
        frame_index = self.noise_frame_index_var.get()
        noise_type = self.noise_type_var.get()

        if not video_path:
            messagebox.showwarning("提示", "请先选择用于噪声分析的视频。")
            return

        self.set_status(f"开始噪声分析: 视频 '{os.path.basename(video_path)}', 帧 {frame_index}, 类型 '{noise_type}'")

        # --- 读取指定帧 --- 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频: {video_path}")
            self.set_status("错误：无法打开视频进行噪声分析")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= frame_count:
            messagebox.showerror("错误", f"帧索引 {frame_index} 超出视频总帧数 {frame_count}！")
            cap.release()
            self.set_status("错误：帧索引无效")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, original_frame_color = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("错误", f"无法读取视频 {video_path} 的第 {frame_index} 帧！")
            self.set_status("错误：无法读取指定帧")
            return

        original_frame_gray = cv2.cvtColor(original_frame_color, cv2.COLOR_BGR2GRAY)

        # --- 添加噪声 --- 
        noisy_img = None
        if noise_type == "高斯":
            gauss_var = self.noise_params['gaussian_var'].get()
            noisy_img = add_gaussian_noise(original_frame_gray, var=gauss_var)
            param_info = f"方差={gauss_var:.3f}"
        elif noise_type == "椒盐":
            sp_amount = self.noise_params['salt_pepper_amount'].get()
            noisy_img = add_salt_pepper_noise(original_frame_gray, amount=sp_amount)
            param_info = f"比例={sp_amount:.3f}"
        else:
            messagebox.showerror("错误", f"未知的噪声类型: {noise_type}")
            return

        # --- 计算统计量并绘图 (使用修改后的绘图函数) --- 
        try:
            fig = self.plot_noise_comparison_figure(original_frame_gray, noisy_img, f"{noise_type} ({param_info})")
            if fig:
                # 在噪声分析标签页显示图表
                self.display_figure(fig, "噪声分析", target_frame=self.noise_display_frame)
                self.set_status(f"噪声分析完成: {noise_type}")
            else:
                 messagebox.showerror("错误", "无法生成噪声对比图。")
        except Exception as e:
            messagebox.showerror("绘图错误", f"生成噪声对比图时出错:\n{e}")
            print(f"Error plotting noise comparison: {e}")
            self.set_status("错误：生成噪声图时出错")

    def plot_noise_comparison_figure(self, original_img, noisy_img, noise_type_label):
        """修改自 Analyze_withNoise.plot_image_and_stats，返回 Figure 对象"""
        stats_orig = calculate_stats(original_img)
        stats_noisy = calculate_stats(noisy_img)

        fig = Figure(figsize=(12, 8), dpi=100) # 调整画布大小
        fig.suptitle(f'原始图像 vs {noise_type_label}噪声', fontsize=14)

        # 绘制图像
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('原始图像 (帧 {})'.format(self.noise_frame_index_var.get()))
        ax1.axis('off')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(noisy_img, cmap='gray')
        ax2.set_title(f'{noise_type_label}噪声图像')
        ax2.axis('off')

        # 绘制直方图
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(stats_orig['histogram'], color='b', label=f'原始 (μ={stats_orig["mean"]:.2f}, σ²={stats_orig["variance"]:.2f})')
        ax3.plot(stats_noisy['histogram'], color='r', label=f'{noise_type_label} (μ={stats_noisy["mean"]:.2f}, σ²={stats_noisy["variance"]:.2f})', alpha=0.7)
        ax3.set_title('灰度直方图对比')
        ax3.set_xlabel('灰度值')
        ax3.set_ylabel('频率')
        ax3.legend()
        ax3.grid(True)

        # 文本显示统计量对比
        ax4 = fig.add_subplot(2, 2, 4)
        text_content = f"--- 统计量对比 ({noise_type_label}) ---\n\n"
        text_content += f"原始均值 (μ): {stats_orig['mean']:.3f}\n"
        text_content += f"噪声均值 (μ): {stats_noisy['mean']:.3f}\n"
        text_content += f"均值变化 (Δμ): {stats_noisy['mean'] - stats_orig['mean']:.3f}\n\n"
        text_content += f"原始方差 (σ²): {stats_orig['variance']:.3f}\n"
        text_content += f"噪声方差 (σ²): {stats_noisy['variance']:.3f}\n"
        text_content += f"方差变化 (Δσ²): {stats_noisy['variance'] - stats_orig['variance']:.3f}"
        ax4.text(0.1, 0.5, text_content, fontsize=9, verticalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))
        ax4.axis('off')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

# --- 主程序入口 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerGUI(root)
    root.mainloop() 