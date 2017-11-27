i_obj=300

I = imread(T.imageFileName{i_obj});

[h,w,d]=size(I);
scale=600/min(h,w);

I_small=imresize(I,scale);

% Insert the ROI labels.
I_small = insertShape(I_small, 'Rectangle', T.concretepylon{i_obj}*scale,'color','cyan','LineWidth', 10);
I_small = insertShape(I_small, 'Rectangle', T.woodpylon{i_obj}*scale,'color','red','LineWidth', 10);
I_small = insertShape(I_small, 'Rectangle', T.metalpylon{i_obj}*scale,'color','black','LineWidth', 10);


%[bboxes, scores, labels] = detect(detector, I_small);
%I_small = insertShape(I_small, 'Rectangle', bboxes,'color','blue','LineWidth', 5);
%I_small = insertObjectAnnotation(I_small, 'rectangle', bboxes, scores);
figure
imshow(I_small)

figure
imshow(I)