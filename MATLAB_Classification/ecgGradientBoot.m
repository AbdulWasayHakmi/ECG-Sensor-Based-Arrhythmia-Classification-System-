clc;
trainEcgData = readtable("mitbih_train.csv");
%head(trainData);


function [trainData, testData, trainLabels, testLabels] = splitData(data, labels, trainRatio)
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

% Extract features and labels
labels = trainEcgData{:, end};  % Extract labels (last column of the table)
ecgSignals = trainEcgData{:, 1:end-1};  % Extract the ECG signals (all columns except the last one)

% Normalize the data
normalizedData = (ecgSignals - min(ecgSignals)) ./ (max(ecgSignals) - min(ecgSignals));

% Split the data into training and testing sets
[trainData, testData, trainLabels, testLabels] = splitData(normalizedData, labels, 0.8);

trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

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
trainDataUndersampled = trainData(undersampledIndices, :);
trainLabelsUndersampled = trainLabels(undersampledIndices);

% Check the new class distribution after undersampling
classDistribution = histcounts(trainLabelsUndersampled);
disp('Class Distribution After Undersampling:');
disp(classDistribution);


% Fit a Random Forest model (using fitcensemble)
numTrees = 100;  % Number of trees in the forest
model = fitcensemble(trainData, trainLabels, 'Method', 'Bag', 'NumLearningCycles', numTrees, 'Learners', 'tree');

% Evaluate the Random Forest model on test data
predictions = predict(model, testData);
accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Convert categorical data to numeric for confusion matrix calculation
trueLabelsNumeric = double(testLabels);
predictedLabelsNumeric = double(predictions);

% Create a confusion matrix
confMat = confusionmat(trueLabelsNumeric, predictedLabelsNumeric);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Save the trained model
save('trainedRandomForestModel.mat', 'model');