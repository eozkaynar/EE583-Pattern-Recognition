clc; clear; close all;
%% Question 3 %%
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
% rng(100);

Mdl = TreeBagger(25,X1,Y1,'SampleWithReplacement','on','OOBPrediction',"on","Method","classification");

view(Mdl.Trees{1},'Mode','graph')
%Prediction 
% Predict the labels from second half of the data using ensemble model
PredictYensemble = predict(Mdl,X2);

% Predict the labels from second half of the data using first tree
PredictYfirsttree = predict(Mdl.Trees{1},X2);


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

plot(oobError(Mdl))
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Error")
