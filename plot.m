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

% Remove features with zero variance
variances = var(trainFeatures);
featuresToKeep = variances > 0;
trainFeaturesFiltered = trainFeatures(:, featuresToKeep);
testFeaturesFiltered = testFeatures(:, featuresToKeep);

% Train various classifiers with filtered features
svmClassifier = fitcecoc(trainFeaturesFiltered, trainLabels);
knnClassifier = fitcknn(trainFeaturesFiltered, trainLabels);
decisionTreeClassifier = fitctree(trainFeaturesFiltered, trainLabels);
randomForestClassifier = TreeBagger(50, trainFeaturesFiltered, trainLabels, 'Method', 'classification');

neuralNet = patternnet(10); % A simple neural network with 10 hidden nodes
neuralNet = train(neuralNet, trainFeaturesFiltered', dummyvar(trainLabels)');

% Test the classifiers
svmPredictedLabels = predict(svmClassifier, testFeaturesFiltered);
knnPredictedLabels = predict(knnClassifier, testFeaturesFiltered);
decisionTreePredictedLabels = predict(decisionTreeClassifier, testFeaturesFiltered);
randomForestPredictedLabels = predict(randomForestClassifier, testFeaturesFiltered);

neuralNetPredictedLabels = neuralNet(testFeaturesFiltered')';
[~, neuralNetPredictedLabels] = max(neuralNetPredictedLabels, [], 2); % Convert output to labels

% Calculate accuracy
svmAccuracy = mean(svmPredictedLabels == testLabels);
knnAccuracy = mean(knnPredictedLabels == testLabels);
decisionTreeAccuracy = mean(decisionTreePredictedLabels == testLabels);
randomForestAccuracy = mean(categorical(randomForestPredictedLabels) == testLabels);

neuralNetAccuracy = mean(neuralNetPredictedLabels == grp2idx(testLabels));

% Display accuracy for each classifier
fprintf('SVM Accuracy: %.2f%%\n', svmAccuracy * 100);
fprintf('k-NN Accuracy: %.2f%%\n', knnAccuracy * 100);
fprintf('Decision Tree Accuracy: %.2f%%\n', decisionTreeAccuracy * 100);
fprintf('Random Forest Accuracy: %.2f%%\n', randomForestAccuracy * 100);

fprintf('Neural Network Accuracy: %.2f%%\n', neuralNetAccuracy * 100);

% Calculate confusion matrices
svmCM = confusionmat(testLabels, svmPredictedLabels);
knnCM = confusionmat(testLabels, knnPredictedLabels);
decisionTreeCM = confusionmat(testLabels, decisionTreePredictedLabels);
randomForestCM = confusionmat(testLabels, categorical(randomForestPredictedLabels));

neuralNetCM = confusionmat(grp2idx(testLabels), neuralNetPredictedLabels);
% Plot confusion matrices
figure;

% SVM
subplot(3, 2, 1);
confusionchart(svmCM, 'Title', 'SVM Confusion Matrix');

% k-NN
subplot(3, 2, 2);
confusionchart(knnCM, 'Title', 'k-NN Confusion Matrix');

% Decision Tree
subplot(3, 2, 3);
confusionchart(decisionTreeCM, 'Title', 'Decision Tree Confusion Matrix');

% Random Forest
subplot(3, 2, 4);
confusionchart(randomForestCM, 'Title', 'Random Forest Confusion Matrix');

% Neural Network
subplot(3, 2, 5);
confusionchart(neuralNetCM, 'Title', 'Neural Network Confusion Matrix');

% Calculate accuracy for each classifier
accuracies = [svmAccuracy, knnAccuracy, decisionTreeAccuracy, randomForestAccuracy, neuralNetAccuracy];
classifiers = {'SVM', 'k-NN', 'Decision Tree', 'Random Forest', 'Neural Network'};

% Plot bar chart
figure;
bar(accuracies);
set(gca, 'XTickLabel', classifiers);
xlabel('Classifier');
ylabel('Accuracy');
title('Comparison of Classifier Accuracies');
