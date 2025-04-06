# 数字图像序列处理实验程序

## 功能说明
本程序是一个数字图像序列处理的GUI应用，实现了以下功能：
- 时间域和空间域均值分析：计算平均帧图像，以及闪烁检测
- 时间域和空间域方差分析：运动区域检测，镜头切换探测
- 时间域和空间域自相关分析：场景变换特性和纹理空间重复性分析

## 使用方法
1. 确保安装了所需的Python库：
```
pip install numpy opencv-python matplotlib pillow
```

2. 运行程序：
```
python digital_image_sequence_processor.py
```

3. 界面说明：
   - 程序会自动加载视频"video_stay_dark.mp4"和"video_move_light.mp4"
   - 程序启动时会自动处理所有数据并将结果保存在processed_video目录下
   - 点击界面下方的按钮可以在不同标签页中查看各种分析结果

## 注意事项
- 需要将两个视频文件（video_stay_dark.mp4和video_move_light.mp4）放在程序同一目录下
- 处理后的所有结果文件会自动保存在processed_video目录下
