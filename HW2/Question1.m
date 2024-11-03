clear; clc; close all;

%% Question 1 %% Maximum Likelihood

rng('default')  % For reproducibility

mu1     = [-.75 .5];
Sigma1  = [.5 .3 ; .3 .8];


X_10 = mvnrnd(mu1,Sigma1,10);

% % Visualize the data
figure;
plot(X_10(:,1),X_10(:,2),'+')
title('Scatter Plot of Normal Distribution for 10 samples');
xlabel('x1')
ylabel('x2')

% Maximum likelihood estimation for 10 samples

N_10               = size(X_10,1);      % Number of samples
mean_estimator_10  = sum(X_10,1)/N_10;  % Summing along the first dimension (rows)

% Initialize outer product
outer_product1     = zeros(size(2, 2));  % Initialize to the correct size

for i=1:N_10
    outer_product1 = outer_product1 + (transpose(X_10(i,:)) - transpose(mean_estimator_10))*transpose(transpose(X_10(i,:)) - transpose(mean_estimator_10));
end

varience_estimator_10 = outer_product1/N_10;

% Maximum likelihood estimation for 1000 samples
X_1000 = mvnrnd(mu1,Sigma1,1000);

% Visualize the data
figure;
plot(X_1000(:,1),X_1000(:,2),'+')
title('Scatter Plot of Normal Distribution for 1000 samples');
xlabel('x1')
ylabel('x2')

N_1000               = size(X_1000,1);    % Number of samples
mean_estimator_1000  = sum(X_1000,1)/N_1000;   % Summing along the first dimension (rows)

% Initialize outer product
outer_product2 = zeros(size(2, 2));  % Initialize to the correct size

for i=1:N_1000
    outer_product2  = outer_product2 + (X_1000(i,:)' - mean_estimator_1000')*(transpose(X_1000(i,:)' - mean_estimator_1000'));
end

varience_estimator_1000 = outer_product2/N_1000;