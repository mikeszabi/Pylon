
v = VideoReader('D:\tmp\IMG_3962.m4v');

h=v.Height;
w=v.Width;
scale=1080/min(h,w);

mov = struct('cdata',zeros(int32(h*scale),int32(w*scale),3,'uint8'),'colormap',[]);

k = 1;
while hasFrame(v)
    frame = readFrame(v);
    frame_small=imresize(frame,scale);
    [bboxes, scores, labels] = detect(detector, frame_small);
    if size(bboxes,1)>0
        frame_small = insertObjectAnnotation(frame_small, 'rectangle', bboxes, cellstr(labels));
    end
%    imshow(frame_small)
    mov(k).cdata = frame_small;
    k = k+1
end

% hf = figure;
% set(hf,'position',[150 150 int32(w*scale) int32(h*scale)]);
% 
% movie(hf,mov,1,v.FrameRate);

v = VideoWriter('detect_VB_long_1080_2160_3840.mp4','MPEG-4');
open(v)
writeVideo(v,mov)
close(v)