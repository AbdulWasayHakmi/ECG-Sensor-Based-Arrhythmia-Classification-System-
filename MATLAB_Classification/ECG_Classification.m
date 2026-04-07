clc;
trainEcgData = readtable("mitbih_train.csv");
%head(trainData);

plot(trainEcgData{1,:}); % Plot the first signal in the dataset
title('Sample ECG Signal');
xlabel('Time');
ylabel('Amplitude');

function [trainData, testData, trainLabels, testLabels] = splitData(data, labels, trainRatio)
    % This function splits data into training and testing sets
    % data: input ECG signals
    % labels: corresponding labels for the data
    % trainRatio: ratio of data to be used for training (e.g., 0.8 for 80%)
    
    % Get the number of samples
    numSamples = size(data, 1);
    
    % Calculate the number of training samples
    numTrain = round(trainRatio * numSamples);
    
    % Generate random indices for shuffling the data
    randIndices = randperm(numSamples);
    
    % Split the data into training and testing sets
    trainData = data(randIndices(1:numTrain), :);
    testData = data(randIndices(numTrain+1:end), :);
    
    % Split the labels into training and testing sets
    trainLabels = labels(randIndices(1:numTrain), :);
    testLabels = labels(randIndices(numTrain+1:end), :);
end

% Assuming the last column contains the labels
labels = trainEcgData{:, end};  % Extract labels (last column of the table)
ecgSignals = trainEcgData{:, 1:end-1};  % Extract the ECG signals (all columns except the last one)

normalizedData = (ecgSignals - min(ecgSignals)) ./ (max(ecgSignals) - min(ecgSignals));

[trainData, testData, trainLabels, testLabels] = splitData(normalizedData, labels, 0.8);

% Convert trainData and testData from table to matrix
trainData = table2array(trainData);
testData = table2array(testData);

% Assuming trainData is now a matrix (after table2array conversion)
numSamples = size(trainData, 1);  % Number of samples
numFeatures = size(trainData, 2); % Should be 188

% Reshape trainData into 4D array [numFeatures, 1, 1, numSamples]
trainData = reshape(trainData', [numFeatures, 1, 1, numSamples]);

% Similarly reshape testData
testData = reshape(testData', [numFeatures, 1, 1, size(testData, 1)]);

trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% --- Undersample the majority class (class 0) ---
% Find the indices of each class
classCounts = histcounts(trainLabels);
[~, majorityClass] = max(classCounts);  % Identify the majority class (class 0)

% Find indices of samples from class 0 (majority class)
majorityClassIndices = find(trainLabels == majorityClass);

% Find indices of minority classes (non-majority)
minorityClassIndices = find(trainLabels ~= majorityClass);

% Determine how many samples to keep (size of minority class)
numMinoritySamples = numel(minorityClassIndices);

% Randomly select a subset of the majority class to match the size of the minority class
undersampledMajorityClassIndices = randsample(majorityClassIndices, numMinoritySamples);

% Combine the indices of the minority class and the undersampled majority class
undersampledIndices = [minorityClassIndices; undersampledMajorityClassIndices];

% Create the new training data and labels
trainData = trainData(undersampledIndices, :);
trainLabels = trainLabels(undersampledIndices);

% Check the new class distribution after undersampling
classDistribution = histcounts(trainLabels);
disp('Class Distribution After Undersampling:');
disp(classDistribution);

layers = [
    imageInputLayer([187, 1, 1])  % Reduced input size
    convolution2dLayer([5, 1], 4, 'Padding', 'same')  % Reduced filters to 4
    reluLayer
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])
    fullyConnectedLayer(4)  % Reduced fully connected layer size
    reluLayer
    fullyConnectedLayer(5)  % Output layer for 5 classes
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', 'MaxEpochs', 5, 'MiniBatchSize', 32, 'Shuffle', 'every-epoch');
net = trainNetwork(trainData, trainLabels, layers, options);

predictions = classify(net, testData);
accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Create confusion matrix
confMat = confusionmat(testLabels, predictions);

% Display the confusion matrix
figure;
confusionchart(confMat);

save('trainedECGModel.mat', 'net');