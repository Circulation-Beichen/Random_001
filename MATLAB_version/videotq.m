video = VideoReader('video1.mp4');

frame_idx = 1;
while hasFrame(video)
    frame = readFrame(video);
    imwrite(frame, fullfile('F:/MATLAB/video/', sprintf('frame_%04d.jpg', frame_idx)));
    frame_idx = frame_idx + 1;
end