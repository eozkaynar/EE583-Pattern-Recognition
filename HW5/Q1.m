clear; clc; close all;
%% Question 1 %%

load fisheriris
X = meas(:,3:4);
Y = species;

rng(1); % For reproducibility

% Scatter plot of the original data
figure;
gscatter(X(:,1), X(:,2), species);
title('Original Data Scatter Plot');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');

% Train a classification tree with default parameters
MdlDefault = fitctree(X,Y,'CrossVal','on');

% Plot the tree
view(MdlDefault.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorDefault = kfoldLoss(MdlDefault);

% Train a classification tree with maximum number of splits at 7
Mdl7 = fitctree(X,Y,'MaxNumSplits',7,'CrossVal','on');

% Plot the tree
view(Mdl7.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorMdl7 = kfoldLoss(Mdl7);

% Train a classification tree with maximum number of splits at 7
MdlSplit = fitctree(X,Y,'SplitCriterion','deviance','CrossVal','on');

% Plot the tree
view(MdlSplit.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorMdlSplit = kfoldLoss(MdlSplit);
