clear; clc; close all;

% Load Fisher Iris Dataset
load fisheriris
X = meas(:,3:4);

% Scatter plot of the original data
figure;
gscatter(X(:,1), X(:,2), species);
title('Original Data Scatter Plot');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');

% Compute distance matrix and similarity matrix
dist_temp = pdist(X);
dist = squareform(dist_temp);

% Similarity matrix
S = exp(-dist.^2);
disp(['Is Similarity Matrix Symmetric: ', num2str(issymmetric(S))]);
% Compute degree matrix
D = diag(sum(S, 2));

% Compute Laplacian Matrices
% 1. Unnormalized Laplacian
L_unnormalized = D - S;

% 2. Symmetric Normalized Laplacian
L_sym = eye(size(S)) - D^(-1/2) * S * D^(-1/2);

% Visualize Laplacian Matrices
figure;
imagesc(L_unnormalized);
colorbar;
title('Unnormalized Laplacian Matrix (L)');
xlabel('Nodes');
ylabel('Nodes');

figure;
imagesc(L_sym);
colorbar;
title('Symmetric Normalized Laplacian Matrix (L_{sym})');
xlabel('Nodes');
ylabel('Nodes');

figure;
imagesc(S);
colorbar;
title('Similarity Matrix (S)');
xlabel('Nodes');
ylabel('Nodes');

% Spectral clustering
k = 3; % Number of clusters
rng('default') % For reproducibility
idx = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');

% Spectral clustering with unnormalized Laplacian
idx_unnormalized = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'none');

% Plot results with symmetric Laplacian
figure;
gscatter(X(:,1), X(:,2), idx);
title('Spectral Clustering with Normalized Laplacian');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');

% Plot results with unnormalized Laplacian
figure;
gscatter(X(:,1), X(:,2), idx_unnormalized);
title('Spectral Clustering with Unnormalized Laplacian');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');

% Mahalanobis Distance
[idx_mahalanobis, ~] = spectralcluster(X, k,'NumNeighbors', size(X,1), 'Distance', 'mahalanobis');
figure;
gscatter(X(:,1), X(:,2), idx_mahalanobis);
title('Spectral Clustering with Mahalanobis Distance');

% KernelScale = 0.1
idx1 = spectralcluster(X, k,'NumNeighbors', size(X,1), 'KernelScale', 0.1, 'LaplacianNormalization', 'symmetric');

% KernelScale = 1
idx2 = spectralcluster(X, k,'NumNeighbors', size(X,1), 'KernelScale', 1, 'LaplacianNormalization', 'symmetric');

% KernelScale = 10
idx3 = spectralcluster(X, k,'NumNeighbors', size(X,1), 'KernelScale', 10, 'LaplacianNormalization', 'symmetric');

% KernelScale = 15
idx4 = spectralcluster(X, k,'NumNeighbors', size(X,1), 'KernelScale', 15, 'LaplacianNormalization', 'symmetric');

% Visualize the results
figure;

gscatter(X(:,1), X(:,2), idx1);
title('KernelScale = 0.1');
xlabel('X1');
ylabel('X2');
figure;
gscatter(X(:,1), X(:,2), idx2);
title('KernelScale = 1');
xlabel('X1');
ylabel('X2');
figure;
gscatter(X(:,1), X(:,2), idx3);
title('KernelScale = 10');
xlabel('X1');
ylabel('X2');

figure;
gscatter(X(:,1), X(:,2), idx4);
title('KernelScale = 15');
xlabel('X1');
ylabel('X2');

% Convert species to numerical labels for comparison
true_labels = grp2idx(species); % 1 = setosa, 2 = versicolor, 3 = virginica

% Q_default = correct_classification(true_labels,idx,k);
% Q_unnormalized = correct_classification(true_labels,idx_unnormalized,k);
% Q_mahalanobis = correct_classification(true_labels,idx_mahalanobis,k);
% Q_kernel01 = correct_classification(true_labels,idx1,k);
% Q_kernel1 = correct_classification(true_labels,idx2,k);
% Q_kernel10 = correct_classification(true_labels,idx3,k);


% Compute distance matrix
dist_temp = pdist(X);
dist = squareform(dist_temp);

% Define different KernelScale values
kernel_scales = [0.1, 1, 10, 15];

% Plot similarity matrices for each KernelScale
figure;
for i = 1:length(kernel_scales)
    % Compute similarity matrix with given KernelScale
    kernel_scale = kernel_scales(i);
    S = exp(-dist.^2 / (2 * kernel_scale^2));
    
    % Plot the similarity matrix
    subplot(2, 2, i);
    imagesc(S);
    colorbar;
    title(['Similarity Matrix (KernelScale = ', num2str(kernel_scale), ')']);
    xlabel('Data Points');
    ylabel('Data Points');
end


function [correct_class1, correct_class2,correct_class3] = correct_classification(true_labels,idx,k)

    % Initialize counts
    correct_class1 = 0;
    correct_class2 = 0;
    correct_class3 = 0;
    
    % Compare true labels with predicted clusters
    for i = 1:k
        % Find majority cluster for each class
        class_indices = (true_labels == i);
        predicted_clusters = idx(class_indices);
        
        % Majority voting: most frequent cluster in true class
        most_frequent_cluster = mode(predicted_clusters);
        correct_count = sum(predicted_clusters == most_frequent_cluster);
        
        % Store correct counts
        if i == 1
            correct_class1 = correct_count;
        elseif i == 2
            correct_class2 = correct_count;
        elseif i == 3
            correct_class3 = correct_count;
        end
    end
    
    % Display results
    fprintf('Correct classifications:\n');
    fprintf('Class 1 (Setosa): %d / %d\n', correct_class1, sum(true_labels == 1));
    fprintf('Class 2 (Versicolor): %d / %d\n', correct_class2, sum(true_labels == 2));
    fprintf('Class 3 (Virginica): %d / %d\n', correct_class3, sum(true_labels == 3));
end


