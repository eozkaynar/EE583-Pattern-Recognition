clc; clear; close all;
%% Question 2 %%
rng(100);
load fisheriris
X = meas(:,3:4);
Y = species;

% Partition the data into two halves
cv = cvpartition(Y, 'Holdout', 0.5); % Split data into training and testing sets

idxX1 = training(cv);                % Indices for X1
idxX2 = test(cv);                    % Indices for X2
idxY1 = training(cv);                % Indices for Y1
idxY2 = test(cv);                    % Indices for Y2

X1 = X(idxX1, :);
Y1 = Y(idxY1);
X2 = X(idxX2, :);
Y2 = Y(idxY2);

weak_learner = templateTree("MaxNumSplits",1);
% First half of the observation
Mdl = fitcensemble(X1,Y1,'Method','AdaBoostM2','NumLearningCycles',25,'Learners',weak_learner);
% Plot the tree
view(Mdl.Trained{1},'Mode','graph')

% Predict the labels from second half of the data using ensemble model
PredictYensemble = predict(Mdl,X2);

% Predict the labels from second half of the data using first tree
PredictYfirsttree = predict(Mdl.Trained{1},X2);

% Calculate classification accuracy for both
accuracyEnsemble    = 0;
accuracyFirstTree   = 0;
for i = 1:numel(Y2)
    accuracyEnsemble  = accuracyEnsemble  + strcmp(PredictYensemble{i},Y2{i});
    accuracyFirstTree = accuracyFirstTree + strcmp(PredictYfirsttree{i}, Y2{i});
end
accuracyEnsemble = accuracyEnsemble / numel(Y2);
accuracyFirstTree = accuracyFirstTree / numel(Y2);
% Display classification accuracies
disp(['Accuracy of AdaBoost ensemble: ', num2str(accuracyEnsemble)]);
disp(['Accuracy of the first tree: ', num2str(accuracyFirstTree)]);

% LearnRate Parameters

lr = 0.01:0.01:1;
acc = zeros(size(lr));
for i = 1:length(lr)
    weak_learner = templateTree("MaxNumSplits",1);
    Mdl = fitcensemble(X1,Y1,'Method','AdaBoostM2','NumLearningCycles',25,'Learners',weak_learner,"LearnRate",lr(i));    
    Yensemble = predict(Mdl,X2);
    Yfirsttree = predict(Mdl.Trained{1},X2);
    
    % Calculate classification accuracy
    accEnsemble    = 0;
    for ii = 1:numel(Y2)
        accEnsemble  = accEnsemble  + strcmp(Yensemble{ii},Y2{ii});
    end
    accEnsemble = accEnsemble / numel(Y2);
    acc(i) = accEnsemble;
end

figure;
plot(lr,acc)
xlabel("Learning Rate")
ylabel("Accuracy")
title("Accuracy vs. Learning Rate")
