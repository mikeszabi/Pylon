load('C:\Users\fodrasz\OneDrive\Annotation\IDB_Pylon\pylon_detector.mat')

%https://uk.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html
resultsStruct = struct([]);
for i = 1:height(testData)

    % Read the image.
    I = imread(testData.imageFileName{i});
    [h,w,d]=size(I);
    scale=1000/min(h,w);

    I_small=imresize(I,scale);

    % Run the detector.
    tic
    [bboxes, scores, labels] = detect(detector, I_small);
    toc
    
    % Collect the results.
    resultsStruct(i).Boxes = bboxes;
    resultsStruct(i).Scores = scores;
    resultsStruct(i).Labels = labels;
end

% Convert the results into a table.
results = struct2table(resultsStruct);

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

figure
plot(recall{3}, precision{3})
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.1f', ap))