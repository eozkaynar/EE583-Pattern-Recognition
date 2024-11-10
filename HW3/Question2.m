clear; close all; clc;
%% Question 2 - Cross-Validate SVM %%

% Load dataset
load fisheriris
% Choose setosa and versicolor
inds = ~strcmp(species,'virginica');
% Use all 4 features
X = meas(inds,:);
y = species(inds);

Y= categorical(species(inds)); % Convert species to categorical for coloring


SVMModel = fitcsvm(X,y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

% 10-Fold Cross-Validation
CVSVMModel1 = crossval(SVMModel,'Kfold',10,'Leaveout','off');
classLoss = kfoldLoss(CVSVMModel1);

% Leave-One-Out Cross-Validation (LOOCV)
CVSVMModel2         = crossval(SVMModel,'Leaveout','on');
leave_one_out_cv    = kfoldLoss(CVSVMModel2);

% Initialize arrays to store cross-validation results
results = [];

% Loop through each feature individually
for i = 1:4
    % Select a single feature
    X_single = X(:, i);
    
    % Train SVM model with a single feature
    SVMModel = fitcsvm(X_single, y, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto');
    
    % 10-Fold Cross-Validation
    CVSVMModel1 = crossval(SVMModel, 'Kfold', 10);
    classLoss_10fold = kfoldLoss(CVSVMModel1);
    
    % Leave-One-Out Cross-Validation (LOOCV)
    CVSVMModel2 = crossval(SVMModel, 'Leaveout', 'on');
    classLoss_LOOCV = kfoldLoss(CVSVMModel2);
    
    % Store the results
    results = [results; {['Feature ', num2str(i)], classLoss_10fold, classLoss_LOOCV}];
end

% Display results
resultsTable = cell2table(results, 'VariableNames', {'Feature', '10FoldError', 'LOOCVError'});
disp(resultsTable);
y_categorical = categorical(y);

% Loop through each feature individually
for i = 1:4
    % Select a single feature
    X_single = X(:, i);
    
    % Train SVM model with a single feature
    SVMModel = fitcsvm(X_single, y, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto');
    
    % Create a range of values for plotting decision boundaries
    x_min = min(X_single) - 1;
    x_max = max(X_single) + 1;
    x_range = linspace(x_min, x_max, 100)';
    
    % Predict scores over the range to find the decision boundary
    [~, scores] = predict(SVMModel, x_range);
    
    % Plot the data and the decision boundary
    figure
    gscatter(X_single, zeros(size(X_single)), y_categorical, 'rb', 'xo');
    hold on
    % plot(x_range, scores(:, 2), 'k-', 'LineWidth', 2); % Decision boundary
    plot(x_range, zeros(size(x_range)), 'k--'); % Decision line at score 0
    title(['Decision Boundary for Feature ', num2str(i)])
    xlabel(['Feature ', num2str(i)])
    ylabel('Decision Score')
    legend('Setosa', 'Versicolor', 'Decision Boundary', 'Location', 'Best')
    hold off
end