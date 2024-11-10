clear; close all; clc;

%% Question 1%%

% Feature pairs (1,2)

% Load dataset X and ys
load fisheriris
% Choose setosa and versicolor
inds = ~strcmp(species,'virginica');
X = meas(inds,1:2);
y = species(inds);

% Feature Pairs (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
feature_pairs = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 3, 4];

% For each feature pairs
for i = 1:size(feature_pairs, 1)
    % Choose one pair
    X = meas(inds, feature_pairs(i, :));
    
    % Train SVM model
    SVMModel = fitcsvm(X, y);
    
    % Get SVs
    sv      = SVMModel.SupportVectors;
    beta    = SVMModel.Beta; % Linear predictor coefficients
    b       = SVMModel.Bias; % Bias term
    
    % Plot
    figure
    gscatter(X(:, 1), X(:, 2), y)
    hold on
    plot(sv(:, 1), sv(:, 2), 'ko', 'MarkerSize', 10)
    X1 = linspace(min(X(:,1)),max(X(:,1)),100);
    X2 = -(beta(1)/beta(2)*X1)-b/beta(2);
    plot(X1,X2,'-')
    m = 1/sqrt(beta(1)^2 + beta(2)^2);  % Margin half-width
    X1margin_low = X1+beta(1)*m^2;
    X2margin_low = X2+beta(2)*m^2;
    X1margin_high = X1-beta(1)*m^2;
    X2margin_high = X2-beta(2)*m^2;
    plot(X1margin_high,X2margin_high,'b--')
    plot(X1margin_low,X2margin_low,'r--')
    legend('versicolor', 'setosa', 'Support Vector')
    title("Features " + int2str(feature_pairs(i, 1)) + " & " + int2str(feature_pairs(i, 2)) + ...
          " - Number of SVs = " + int2str(length(sv)))
    hold off
end