clear; clc; close all;
% Define the data matrix (each row is an observation, each column is a variable)
X = [0 1 2 3; 
     1 0 4 5; 
     2 4 0 6; 
     3 5 6 0];

Z = linkage(X, 'centroid');
% Mahalanobis Distance
D_mahalanobis = pdist(X, 'mahalanobis'); % Precompute Mahalanobis distances
Z_mahalanobis = linkage(D_mahalanobis, 'centroid'); % Perform hierarchical clustering

% Minkowski Distance (with p = 3)
D_minkowski = pdist(X, 'minkowski', 1); % p = 1
Z_minkowski = linkage(D_minkowski, 'centroid');

% Standardized Euclidean Distance
D_seuclidean = pdist(X, 'seuclidean'); % Precompute Standardized Euclidean distances
Z_seuclidean = linkage(D_seuclidean, 'centroid');

% cosine Distance
D_cosine = pdist(X, 'cosine'); % Precompute Standardized Euclidean distances
Z_cosine = linkage(D_cosine, 'centroid');

% correlation Distance
D_corr = pdist(X, 'correlation'); % Precompute Standardized Euclidean distances
Z_correlation = linkage(D_corr, 'centroid');

% Display the linkage matrices
disp('Linkage Matrix (Mahalanobis):');
disp(Z_mahalanobis);
disp('Linkage Matrix (Minkowski):');
disp(Z_minkowski);
disp('Linkage Matrix (Standardized Euclidean):');
disp(Z_seuclidean);

% Plot dendrograms for each distance metric

figure;
subplot(2,3,1);
dendrogram(Z);
title('Dendrogram (Euclidean)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance');

subplot(2,3,2);
dendrogram(Z_mahalanobis);
title('Dendrogram (Mahalanobis)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance');

subplot(2,3,3);
dendrogram(Z_minkowski);
title('Dendrogram (Minkowski, p=1)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance');

subplot(2,3,4);
dendrogram(Z_seuclidean);
title('Dendrogram (Standardized Euclidean)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance'); 

subplot(2,3,5);
dendrogram(Z_cosine);
title('Dendrogram (Cosine)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance'); 

subplot(2,3,6);
dendrogram(Z_correlation);
title('Dendrogram (Correlation)');
xlabel('Leaf Nodes');
ylabel('Linkage Distance'); 

