clc;
trainEcgData = readtable("mitbih_train.csv");

function [trainData, testData, trainLabels, testLabels] = splitData(data, labels, trainRatio)
    % Split data into training and testing sets
    numSamples = size(data, 1);
    numTrain = round(trainRatio * numSamples);
    randIndices = randperm(numSamples);
    trainData = data(randIndices(1:numTrain), :);
    testData = data(randIndices(numTrain+1:end), :);
    trainLabels = labels(randIndices(1:numTrain), :);
    testLabels = labels(randIndices(numTrain+1:end), :);
end

% Assuming the last column contains the labels
labels = trainEcgData{:, end};  % Extract labels (last column of the table)
ecgSignals = trainEcgData{:, 1:end-1};  % Extract the ECG signals (all columns except the last one)

normalizedData = (ecgSignals - min(ecgSignals)) ./ (max(ecgSignals) - min(ecgSignals));

% Split data into training and testing sets
[trainData, testData, trainLabels, testLabels] = splitData(normalizedData, labels, 0.8);

% Reshape trainData and testData into 4D arrays
numSamples = size(trainData, 1);  % Number of samples
numFeatures = size(trainData, 2); % Number of features (should be 188)
trainData = reshape(trainData', [numFeatures, 1, 1, numSamples]);
testData = reshape(testData', [numFeatures, 1, 1, size(testData, 1)]);

trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

uniqueLabels = unique(trainLabels);
disp('Unique Class Labels:');
disp(uniqueLabels);

% --- Class Distribution Before Undersampling ---
disp('Class Distribution Before Undersampling:');
disp(histcounts(trainLabels));

% --- Undersampling Class 0 ---
desiredClass0Size = 8000;  % Specify the desired size for class 0

% Convert categorical labels to numeric for comparison

% Get the indices of class 0 samples
class0Indices = find(trainLabels == '0');

% Check how many class 0 samples exist
numClass0 = length(class0Indices);
disp(['Number of class 0 samples: ', num2str(numClass0)]);

% If there are fewer than desiredClass0Size samples, adjust the desired size
if numClass0 < desiredClass0Size
    desiredClass0Size = numClass0;
    disp(['Reducing undersampling size to the number of available class 0 samples: ', num2str(desiredClass0Size)]);
end

% Randomly sample the desired number of class 0 samples
undersampledClass0Indices = randsample(class0Indices, desiredClass0Size);

% Get indices of other classes (1 to 4)
classOtherIndices = find(trainLabels ~= '0');

% Combine the undersampled class 0 with all other classes
undersampledIndices = [undersampledClass0Indices; classOtherIndices];

% Convert undersampled indices back into the data
trainDataUndersampled = trainData(:, :, :, undersampledIndices);
trainLabelsUndersampled = trainLabels(undersampledIndices);

% Check the new class distribution after undersampling
classDistribution = histcounts(trainLabelsUndersampled);
disp('Class Distribution After Undersampling:');
disp(classDistribution);

% Define the layers for the CNN model
layers = [
    imageInputLayer([187, 1, 1])  % Input size
    convolution2dLayer([5, 1], 4, 'Padding', 'same')  % Reduced filters to 4
    reluLayer
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])
    fullyConnectedLayer(4)  % Reduced fully connected layer size
    reluLayer
    fullyConnectedLayer(5)  % Output layer for 5 classes
    softmaxLayer
    classificationLayer
];

% Training options with Adam optimizer
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch');

% Train the model with undersampled data
net = trainNetwork(trainDataUndersampled, trainLabelsUndersampled, layers, options);

% Predict on the test data
predictions = classify(net, testData);
accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Create confusion matrix
confMat = confusionmat(testLabels, predictions);

% Display the confusion matrix
figure;
confusionchart(confMat);

save('trainedECGModel1.mat', 'net');