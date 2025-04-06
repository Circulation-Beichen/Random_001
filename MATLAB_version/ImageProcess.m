%图像准备
folder = 'F:/MATLAB/video/'; 
files = dir(fullfile(folder, '*.jpg'));
num_frames = length(files);
image_Buff = [];%高，宽，长

for k = 1:num_frames
    filename = fullfile(folder, files(k).name);
    img = imread(filename);
    if size(img, 3) == 3%彩转灰
        img = rgb2gray(img); 
    end
    image_Buff(:, :, k) = im2double(img); % 高啊，宽，长度，归一化
end

% 显示原始图像
figure;
imshow(image_Buff(:, :, 1)); % 第一帧
title('原始图像');

%统计特征计算 

%均值函数
mean_image = mean(image_Buff, 3);
%显示均值图像
figure;
imshow(mean_image);
title('均值图像');
colorbar;

%方差函数
image_Var = var(image_Buff, 0, 3); 
%显示方差图像
figure;
imshow(image_Var, []);
title('方差图像');
colorbar;

%自相关函数
pixel_x = 100; 
pixel_y = 150;
image_Tim = squeeze(image_Buff(pixel_x, pixel_y, :));
[R, lags] = xcorr(image_Tim, 'unbiased');
figure;
plot(lags, R);
xlabel('延迟 \tau');
ylabel('自相关 R(\tau)');
title('自相关函数');

%协方差函数
% 添加高斯噪声,均值为0，方差0.1
noisy_Buff = imnoise(image_Buff, 'gaussian', 0, 0.1);
image_Cov = cov(noisy_Buff(:), image_Buff(:));
disp('协方差矩阵:');
disp(image_Cov);

%一维密度函数
figure;
histogram(image_Buff(:, :, 1), 'Normalization', 'pdf');
xlabel('像素值');
ylabel('概率密度');
title('第一帧的概率密度函数');

%二维密度/分布



%特性分析 
%平稳性分析

%各态历经性分析

disp('时间平均与集合平均是否一致:');
disp(isequal(mean_image, mean(image_Buff, 3))); % 应输出1（True）


%噪声模型影响

noisy_Mean = mean(noisy_Buff, 3);
noisy_Var = var(noisy_Buff, 0, 3);

% 比较噪声前后的方差
figure;
subplot(1, 2, 1);
imshow(variance_image, []);
title('原始方差');
subplot(1, 2, 2);
imshow(noisy_Var, []);
title('加噪后方差');
sgtitle('噪声对方差的影响');
%比较分析 
% SNR（信噪比）
psnr_value = psnr(noisy_Buff, image_Buff);
disp(['PSNR = ', num2str(psnr_value), ' dB']);


%对比无噪声和有噪声情况下的统计特征
%分析噪声对图像序列统计特征的具体影响
%
%
%
%
%