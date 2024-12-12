clear; clc; close all;

% Load Fisher's iris data set. Use the petal lengths and widths as predictors.
load fisheriris
X = meas(:,3:4);

figure;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)'; 
ylabel 'Petal Widths (cm)';

rng(1); % For reproducibility

C = rand(3,2); % Randomly initialized centroids
x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

% K-Means with random centroids
idx2Region = kmeans(XGrid,3,'MaxIter',25,'Start',C);
[idx,C_1,sumd] = kmeans(X,3,"Distance","sqeuclidean","Start",C);

% Assigns each node in the grid to the closest centroid
figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title(['Fisher''s Iris Data Random Mean Values (C = [', num2str(C(:)'), '])']);
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

% Measure Clustering Quality
Quality = class_quality(X,idx,C);

% K-Means with better initial centroids
C_better = [4.5, 1.5; 6, 2; 1.5, 0.25];
idx2Region_better = kmeans(XGrid,3,'MaxIter',1,'Start',C_better);
[idx2,C_better1,sumd2] = kmeans(X,3,'Start',C_better,'Distance','sqeuclidean');

% Assigns each node in the grid to the closest centroid
figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region_better,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title(['Fisher''s Iris Data Better Initialized Mean Values (C_{better} = [', num2str(C_better(:)'), '])']);
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

% Measure Clustering Quality
Quality_better = class_quality(X,idx2,C_better1);

% K-Means with centroids based on class means
C_better2 = [mean(X(1:50,:)); mean(X(51:100,:)); mean(X(101:end,:))];
[idx3,C_better3,sumd3] = kmeans(X,3,'Start',C_better2,'Distance','sqeuclidean');
idx2Region_better2 = kmeans(XGrid,3,'MaxIter',1,'Start',C_better2);

% Assigns each node in the grid to the closest centroid
figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region_better2,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title(['Fisher''s Iris Data Better Initialized Mean Values 2 (C_{better2} = [', num2str(C_better2(:)'), '])']);
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

% Measure Clustering Quality
Quality_better2 = class_quality(X,idx3,C_better3);
Quality_random = class_quality(X,idx3,C_1);

% Function to compute clustering quality
function Quality = class_quality(X,idx,C)
Quality = 0;
    for i = 1:length(X)
        Quality = Quality + norm(X(i,:) - C(idx(i),:)).^2;
    end
end
