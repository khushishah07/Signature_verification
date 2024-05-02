% Define path to your new signature image
newSignaturePath = 'new2.png';

% Load and pre-process the new signature image
newImg = imread(newSignaturePath);
newImg = rgb2gray(newImg); % Convert to grayscale
newImg = imresize(newImg, [150 150]); % Normalize size

% Extract features from the new signature image
[newFeatureVector, ~] = extractHOGFeatures(newImg);


% Predict the label of the new signature and get the scores
[predictedLabel, score] = predict(classifier, newFeatureVector);

% Find the score for the predicted label
[~, maxScoreIndex] = max(score, [], 2);
matchingScore = score(maxScoreIndex);

fprintf('The predicted label for the new signature is: %s\n', string(predictedLabel));
fprintf('Matching score: %.2f\n', matchingScore);

