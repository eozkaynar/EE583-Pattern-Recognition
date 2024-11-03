clear; clc; close all;

%% Question 3 %% Minimum error-rate classifier
rng(55)  % For reproducibility

% Mean vectors of classes
mu1     = [-1, -1]';
mu2     = [1, 1]';

% Covarience matrix 
sigma   = [1.4 .2; .2 .28];

% Generate sets
omega1  = mvnrnd(mu1,sigma,500);
omega2  = mvnrnd(mu2,sigma,500);


% Randomly select 250 indices for training
indices1     = randperm(500, 250); % Randomly select 250 indices from 1 to 500
indices2     = randperm(500, 250); % For omega2 as well

% Split train-test sets using the random indices
omega1_train = omega1(indices1, :);
omega1_test  = omega1(setdiff(1:500, indices1), :); % Remaining points for testing
omega2_train = omega2(indices2, :);
omega2_test  = omega2(setdiff(1:500, indices2), :);

% Create a new figure for the plot
figure;

% Plot training data
scatter(omega1_train(:,1), omega1_train(:,2), 25, 'r', 'filled'); % Class 1 training data
hold on;
scatter(omega2_train(:,1), omega2_train(:,2), 25, 'b', 'filled'); % Class 2 training data
hold off;
title('Training Data (Class 1 and Class 2)');
xlabel('x1');
ylabel('x2');
legend('Class 1 Training', 'Class 2 Training');
grid on;

% Plot test data
figure;
scatter(omega1_test(:,1), omega1_test(:,2), 25, 'r', 'o'); % Class 1 test data
hold on
scatter(omega2_test(:,1), omega2_test(:,2), 25, 'b', 'o'); % Class 2 test data
hold off

title('Test Data');
xlabel('x1');
ylabel('x2');
legend( 'Class 1 Test', 'Class 2 Test');
grid on;


% Decision boundary from eq REFF 
n   = (mu1 - mu2)'; % normal of the boundary line
x0  = (mu1 + mu2)/2; 

% Decision boundary function
decision_boundary  = @(x1, x2) n * inv(sigma) * ([x1; x2] - x0);

% Classification and Error Calculation for Train Data
train_data           = [omega1_train; omega2_train];
train_labels         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels1    = zeros(size(train_labels));

for i = 1:length(train_data)
    x = train_data(i, :)';
    if decision_boundary(x(1), x(2)) > 0
        predicted_labels1(i) = 1; % Assign to class 1
    else
        predicted_labels1(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate1 = sum(predicted_labels1 ~= train_labels) / length(train_labels);
fprintf('Error Rate for Train Data: %.2f%%\n', error_rate1 * 100);

% Classification and Error Calculation for Test Data
test_data           = [omega1_test; omega2_test];
test_labels         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels    = zeros(size(test_labels));

for i = 1:length(test_data)
    x = test_data(i, :)';
    if decision_boundary(x(1), x(2)) > 0
        predicted_labels(i) = 1; % Assign to class 1
    else
        predicted_labels(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate = sum(predicted_labels ~= test_labels) / length(test_labels);
fprintf('Error Rate for Test Data: %.2f%%\n', error_rate * 100);

% Plot decision boundary on the train data
figure;
scatter(omega1_train(:, 1), omega1_train(:, 2), 25, 'r', 'filled'); % Class 1 test data
hold on;
scatter(omega2_train(:, 1), omega2_train(:, 2), 25, 'b', 'filled'); % Class 2 test data
fimplicit(@(x1, x2) decision_boundary(x1, x2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Train Data with Decision Boundary (Error Rate: %.2f%%)', error_rate1 * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Train', 'Class 2 Train', 'Decision Boundary');
grid on;

% Plot decision boundary on the test data
figure;
scatter(omega1_test(:, 1), omega1_test(:, 2), 25, 'r', 'o'); % Class 1 test data
hold on;
scatter(omega2_test(:, 1), omega2_test(:, 2), 25, 'b', 'o'); % Class 2 test data
fimplicit(@(x1, x2) decision_boundary(x1, x2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Test Data with Decision Boundary (Error Rate: %.2f%%)', error_rate * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Test', 'Class 2 Test', 'Decision Boundary');
grid on;

%% Question 3 iii)

% Maximum Likelihood estimator
N1              = size(omega1_train,1);
N2              = size(omega2_train,1);

mu1_estimator   = sum(omega1_train,1)/N1;
mu2_estimator   = sum(omega2_train,1)/N2;

% Initialize outer product

outer_product1 = zeros(size(2, 2));  % Initialize to the correct size

for i=1:N1
    outer_product1 = outer_product1 + (omega1_train(i,:)' - mu1_estimator')*(transpose(omega1_train(i,:)' - mu1_estimator'));
end

variance_estimator1 = outer_product1/N1;

outer_product2 = zeros(size(2, 2));  % Initialize to the correct size

for i=1:N2
    outer_product2  = outer_product2 + (omega2_train(i,:)' - mu2_estimator')*(transpose(omega2_train(i,:)' - mu2_estimator'));
end

variance_estimator2 = outer_product2/N2;

% Decision boundary from eq REFF 
n_2   = (mu1_estimator - mu2_estimator); % normal of the boundary line
x0_2  = (mu1_estimator + mu2_estimator)'/2; 

% Decision boundary function
decision_boundary_2  = @(x1_2, x2_2) n_2 * inv(variance_estimator1) * ([x1_2; x2_2] - x0_2);

% Classification and Error Calculation for Train Data
train_data_2           = [omega1_train; omega2_train];
train_labels_2         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels1_2     = zeros(size(train_labels));

for i = 1:length(train_data_2)
    x = train_data_2(i, :)';
    if decision_boundary_2(x(1), x(2)) > 0
        predicted_labels1_2(i) = 1; % Assign to class 1
    else
        predicted_labels1_2(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate1_2 = sum(predicted_labels1_2 ~= train_labels_2) / length(train_labels_2);
fprintf('Error Rate for Train Data ML: %.2f%%\n', error_rate1_2 * 100);

% Classification and Error Calculation for Test Data
test_data_2           = [omega1_test; omega2_test];
test_labels_2         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels_2    = zeros(size(test_labels));

for i = 1:length(test_data_2)
    x = test_data_2(i, :)';
    if decision_boundary_2(x(1), x(2)) > 0
        predicted_labels_2(i) = 1; % Assign to class 1
    else
        predicted_labels_2(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate_2 = sum(predicted_labels_2 ~= test_labels_2) / length(test_labels_2);
fprintf('Error Rate for Test Data ML: %.2f%%\n', error_rate_2 * 100);

% Plot decision boundary on the train data
figure;
scatter(omega1_train(:, 1), omega1_train(:, 2), 25, 'r', 'filled'); % Class 1 test data
hold on;
scatter(omega2_train(:, 1), omega2_train(:, 2), 25, 'b', 'filled'); % Class 2 test data
fimplicit(@(x1_2, x2_2) decision_boundary_2(x1_2, x2_2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Train Data with Decision Boundary ML (Error Rate: %.2f%%)', error_rate1_2 * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Train', 'Class 2 Train', 'Decision Boundary');
grid on;

% Plot decision boundary on the test data
figure;
scatter(omega1_test(:, 1), omega1_test(:, 2), 25, 'r', 'o'); % Class 1 test data
hold on;
scatter(omega2_test(:, 1), omega2_test(:, 2), 25, 'b', 'o'); % Class 2 test data
fimplicit(@(x1_2, x2_2) decision_boundary_2(x1_2, x2_2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Test Data with Decision Boundary ML (Error Rate: %.2f%%)', error_rate_2 * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Test', 'Class 2 Test', 'Decision Boundary');
grid on;
%% Bonus
% Mean vectors and covariance matrices for classes i and j
mu1_estimator = mu1_estimator';
mu2_estimator = mu2_estimator';
% Define the quadratic decision boundary function
decision_boundary_3 = @(x1, x2) ([x1; x2]' * (inv(variance_estimator2) - inv(variance_estimator1)) * [x1; x2]) + ...
                               2 * ((mu1_estimator' * inv(variance_estimator1)) - (mu2_estimator' * inv(variance_estimator2))) * [x1; x2] + ...
                               (mu2_estimator' * inv(variance_estimator2) * mu2_estimator) - (mu1_estimator' * inv(variance_estimator1) * mu1_estimator) + ...
                               log(det(variance_estimator2) / det(variance_estimator1));

% Classification and Error Calculation for Train Data
train_data_2           = [omega1_train; omega2_train];
train_labels_2         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels1_2     = zeros(size(train_labels));

for i = 1:length(train_data_2)
    x = train_data_2(i, :)';
    if decision_boundary_3(x(1), x(2)) > 0
        predicted_labels1_2(i) = 1; % Assign to class 1
    else
        predicted_labels1_2(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate1_2 = sum(predicted_labels1_2 ~= train_labels_2) / length(train_labels_2);
fprintf('Error Rate for Train Data ML Quadratic: %.2f%%\n', error_rate1_2 * 100);

% Classification and Error Calculation for Test Data
test_data_2           = [omega1_test; omega2_test];
test_labels_2         = [ones(250, 1); 2 * ones(250, 1)]; % 1 for class 1, 2 for class 2
predicted_labels_2    = zeros(size(test_labels));

for i = 1:length(test_data_2)
    x = test_data_2(i, :)';
    if decision_boundary_3(x(1), x(2)) > 0
        predicted_labels_2(i) = 1; % Assign to class 1
    else
        predicted_labels_2(i) = 2; % Assign to class 2
    end
end

% Calculate error rate
error_rate_2 = sum(predicted_labels_2 ~= test_labels_2) / length(test_labels_2);
fprintf('Error Rate for Test Data ML Quadratic: %.2f%%\n', error_rate_2 * 100);

% Plot decision boundary on the train data
figure;
scatter(omega1_train(:, 1), omega1_train(:, 2), 25, 'r', 'filled'); % Class 1 test data
hold on;
scatter(omega2_train(:, 1), omega2_train(:, 2), 25, 'b', 'filled'); % Class 2 test data
fimplicit(@(x1_2, x2_2) decision_boundary_3(x1_2, x2_2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Train Data with Quadratic Decision Boundary ML (Error Rate: %.2f%%)', error_rate1_2 * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Train', 'Class 2 Train', 'Decision Boundary');
grid on;

% Plot decision boundary on the test data
figure;
scatter(omega1_test(:, 1), omega1_test(:, 2), 25, 'r', 'o'); % Class 1 test data
hold on;
scatter(omega2_test(:, 1), omega2_test(:, 2), 25, 'b', 'o'); % Class 2 test data
fimplicit(@(x1_2, x2_2) decision_boundary_3(x1_2, x2_2), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
hold off;

title(sprintf('Test Data with Quadratic Decision Boundary ML (Error Rate: %.2f%%)', error_rate_2 * 100));
xlabel('x1');
ylabel('x2');
legend('Class 1 Test', 'Class 2 Test', 'Decision Boundary');
grid on;