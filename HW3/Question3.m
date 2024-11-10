clear; clc; close all;

%% Question 3 Multiclass SVM %% 

% Load data
load fisheriris
% Choose features 1 and 2 
X = meas(:,1:2);
Y = species;

SVMModels   = cell(3,1);
classes     = unique(Y);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx            = strcmp(Y,classes(j)); % Create binary classes for each classifier
    SVMModels{j}    = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',false,...
        'KernelFunction','linear','BoxConstraint',1);
end

% Examine scatter plot of the data
figure
gscatter(X(:,1),X(:,2),Y);
hold on 
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Sepal Length (cm)');
ylabel('Sepal Width (cm)');
legend('Location','Northwest');
hold off

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);

for j = 1:numel(classes)
    sv = SVMModels{j}.SupportVectors;
    h(6+j) = plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Support Vectors');
end
title('{\bf Iris Classification Regions}');
xlabel('Sepal Length (cm)');
ylabel('Sepal Width (cm)');
legend(h,{'setosa region','versicolor region','virginica region',...
    'observed setosa','observed versicolor','observed virginica','Support Vectors'},...
    'Location','Northwest');
axis tight
hold off