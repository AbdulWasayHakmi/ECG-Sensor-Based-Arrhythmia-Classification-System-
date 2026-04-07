load('trainedRandomForestModel.mat');

% Load the test dataset
testEcgData = readtable("mitbih_test.csv");

testLabels = testEcgData{:, end};  % Extract labels from the last column of the table

testLabels = categorical(testLabels);

ecgSignals = testEcgData{:, 1:end-1};  % Extract the ECG signals (all columns except the last one)

normalizedData = (ecgSignals - min(ecgSignals)) ./ (max(ecgSignals) - min(ecgSignals));

% Reshape test data to fit the CNN input dimensions
numSamples = size(normalizedData, 1);
numFeatures = size(normalizedData, 2);

disp(['Number of samples: ', num2str(numSamples)]);
disp(['Number of features: ', num2str(numFeatures)]);

%testData = reshape(normalizedData', [1, 187]);
%disp(size(testData))

predictions = predict(model, normalizedData);  % net is your trained model

accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);