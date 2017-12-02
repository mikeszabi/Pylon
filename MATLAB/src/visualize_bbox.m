load(detector_file)

i_obj=20;

% [I_out]=load_one_image(T.imageFileName{i_obj},200);
% inf=getexif(T.imageFileName{i_obj});
% inf=imfinfo(T.imageFileName{i_obj})

I = imread(testData.imageFileName{i_obj});

[h,w,d]=size(I);
scale=640/min(h,w);

I_small=imresize(I,scale);

% Insert the ROI labels.
% I_small = insertShape(I_small, 'Rectangle', testData.concretepylon{i_obj}*scale,'color','cyan','LineWidth', 10);
% I_small = insertShape(I_small, 'Rectangle', testData.woodpylon{i_obj}*scale,'color','red','LineWidth', 10);
% I_small = insertShape(I_small, 'Rectangle', testData.metalpylon{i_obj}*scale,'color','black','LineWidth', 10);


[bboxes, scores, labels] = detect(detector, I_small);
%I_small = insertShape(I_small, 'Rectangle', bboxes,'color','blue','LineWidth', 5);
if size(bboxes,1)>0
    I_small = insertShape(I_small, 'Rectangle', bboxes,'color','blue','LineWidth', 5);
    I_small = insertObjectAnnotation(I_small, 'rectangle', bboxes, scores);
end
figure
imshow(I_small)
