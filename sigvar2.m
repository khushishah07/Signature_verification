% Define paths to your training and testing dataset folders
trainDatasetPath = 'train/';
testDatasetPath = 'test/';

% Load the training and testing datasets
trainSignatureImages = imageDatastore(trainDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.png');
testSignatureImages = imageDatastore(testDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.png');

% Pre-process and extract features from training images
trainFeatures = [];
trainLabels = [];
for i = 1:numel(trainSignatureImages.Files)
    img = readimage(trainSignatureImages, i);
    img = rgb2gray(img); % Convert to grayscale
    img = imresize(img, [150 150]); % Normalize size
    [featureVector, ~] = extractHOGFeatures(img);
    trainFeatures = [trainFeatures; featureVector];
    trainLabels = [trainLabels; trainSignatureImages.Labels(i)];
end

% Pre-process and extract features from testing images
testFeatures = [];
testLabels = [];
for i = 1:numel(testSignatureImages.Files)
    img = readimage(testSignatureImages, i);
    img = rgb2gray(img); % Convert to grayscale
    img = imresize(img, [150 150]); % Normalize size
    [featureVector, ~] = extractHOGFeatures(img);
    testFeatures = [testFeatures; featureVector];
    testLabels = [testLabels; testSignatureImages.Labels(i)];
end

% Train a classifier (example using SVM)
classifier = fitcecoc(trainFeatures, trainLabels);

% Test the classifier
predictedLabels = predict(classifier, testFeatures);
accuracy = mean(predictedLabels == testLabels);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);


% Train different classifiers
svmClassifier = fitcecoc(trainFeatures, trainLabels);
knnClassifier = fitcknn(trainFeatures, trainLabels);
decisionTreeClassifier = fitctree(trainFeatures, trainLabels);

% Test the classifiers
svmPredictedLabels = predict(svmClassifier, testFeatures);
knnPredictedLabels = predict(knnClassifier, testFeatures);
decisionTreePredictedLabels = predict(decisionTreeClassifier, testFeatures);

% Calculate accuracy
svmAccuracy = mean(svmPredictedLabels == testLabels);
knnAccuracy = mean(knnPredictedLabels == testLabels);
decisionTreeAccuracy = mean(decisionTreePredictedLabels == testLabels);

% Display accuracy for each classifier
fprintf('SVM Accuracy: %.2f%%\n', svmAccuracy * 100);
fprintf('k-NN Accuracy: %.2f%%\n', knnAccuracy * 100);
fprintf('Decision Tree Accuracy: %.2f%%\n', decisionTreeAccuracy * 100);

% Confusion Matrix
%figure;
%confMat = confusionmat(testLabels, predictedLabels);
%confMatChart = confusionchart(confMat);
%confMatChart.Title = 'Confusion Matrix for Signature Verification';
%confMatChart.RowSummary = 'row-normalized';
%confMatChart.ColumnSummary = 'column-normalized';

