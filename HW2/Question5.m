clear; clc; close all;

%%
% Load dataset
load fisheriris

% Extract feature and labels
X = meas(:,1:2);    % Features, take length and width
Y = species;        % Labels

figure;
gscatter(X(:,1),X(:,2),species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
% PCA
[coeff, score, eigenvalues] = pca(X);

% Take basis and normal vectors
basis   = coeff(:,1);
normal  = coeff(:,2);

% Mean vector of data
m       = mean(X,1);

% Scores give the projection of each point onto the line
[n,p]   = size(X);

% Fitting lines
Xfit    = repmat(m,n,1) + score(:,1)*coeff(:,1)'; % Most principle
Xfit2   = repmat(m,n,1) + score(:,2)*coeff(:,2)'; % Least principle


% Error
error   = abs((X - repmat(m,n,1))*normal);
sse     = sum(error.^2);

% Plot PCA
figure;
t = [min(score(:,1)) - 0.2, max(score(:,1)) + 0.2];
endpts = [m + t(1) * basis'; m + t(2) * basis'];
plot(endpts(:,1), endpts(:,2), 'k-', 'LineWidth', 1.5);
hold on;
colors = 'rgb';
classes = unique(Y);

idx1 = strcmp(Y, classes{1});
idx2 = strcmp(Y, classes{2});
idx3 = strcmp(Y, classes{3});

plot(X(idx1,1), X(idx1,2), 'ro', 'MarkerFaceColor', 'none');  % Blue diamond outline with no fill
plot(X(idx2,1), X(idx2,2), 'gs', 'MarkerFaceColor', 'none');  
plot(X(idx3,1), X(idx3,2), 'bd', 'MarkerFaceColor', 'none');  


for i = 1:length(classes)
    % Select data points for the current class
    idx = strcmp(Y, classes{i});
    % Plot projections on the PCA line
    X1 = [X(idx,1) Xfit(idx,1) nan*ones(sum(idx),1)];
    X2 = [X(idx,2) Xfit(idx,2) nan*ones(sum(idx),1)];
    plot(X1', X2', '-', 'Color', colors(i));
end

hold off;
legend('Fitting Line','Setosa','versicolor','virginica', 'Location', 'best');
xlabel('Sepal length');
ylabel('Sepal width');
title('Orthogonal Regression using PCA with Projections by Class');


% Plot of most principle direction
X_setosa        = Xfit(1:50,1);
X_versicolor    = Xfit(50:100,1);
X_virginica     = Xfit(100:150,1);

figure;
subplot(1,2,1)
histogram(X_setosa, 10); % 10 bins for Setosa
hold on;
histogram(X_versicolor, 10); % 10 bins for Versicolor
histogram(X_virginica, 10); % 10 bins for Virginica
hold off;
title("Histogram of of the PCA-reduced 1D data on most principle direction")
xlabel('Principal Component 1');
ylabel('Frequency');

% Plot of least principle direction
X_setosa2        = Xfit2(1:50,1);
X_versicolor2    = Xfit2(50:100,1);
X_virginica2     = Xfit2(100:150,1);

subplot(1,2,2)
histogram(X_setosa2, 10); % 10 bins for Setosa
hold on;
histogram(X_versicolor2, 10); % 10 bins for Versicolor
histogram(X_virginica2, 10); % 10 bins for Virginica
hold off;
title("Histogram of of the PCA-reduced 1D data on least principle direction")
xlabel('Principal Component 2');
ylabel('Frequency');
