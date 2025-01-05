clear; clc; close all;
%% Question 1 %%

load fisheriris
X = meas(:,3:4);
Y = species;

rng(1); % For reproducibility



% Train a classification tree with default parameters
MdlDefault = fitctree(X,Y,'CrossVal','on');

% Plot the tree
view(MdlDefault.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorDefault = kfoldLoss(MdlDefault);

% Classification Region Plot
cutPredictors   = MdlDefault.Trained{1}.CutPredictor;     % Cut Predictors
cutPoints       = MdlDefault.Trained{1}.CutPoint;         % Cut points
% Scatter plot of the original data
figure;
gscatter(X(:,1), X(:,2), species);
title('Original Data Scatter Plot and Decion Boundaries');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
hold on; % Mevcut scatter plot üzerine çizim yap

for i = 1:length(cutPoints)
    if ~isnan(cutPoints(i)) % Eğer bir bölme noktası varsa
        if strcmp(cutPredictors{i}, 'x1') % Petal Length (x1)
            xline(cutPoints(i), '--k', 'LineWidth', 1.5); % Dikey çizgi
        elseif strcmp(cutPredictors{i}, 'x2') % Petal Width (x2)
            yline(cutPoints(i), '--k', 'LineWidth', 1.5); % Yatay çizgi
        end
    end
end

legend('Setosa', 'Versicolor', 'Virginica', 'Decision Boundaries');
hold off;

% Train a classification tree with maximum number of splits at 7
Mdl7 = fitctree(X,Y,'MaxNumSplits',7);

% Plot the tree
view(Mdl7.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorMdl7 = kfoldLoss(Mdl7);

% Classification Region Plot
cutPredictors   = Mdl7.Trained{1}.CutPredictor;     % Cut Predictors
cutPoints       = Mdl7.Trained{1}.CutPoint;         % Cut points
% Scatter plot of the original data
figure;
gscatter(X(:,1), X(:,2), species);
title('Original Data Scatter Plot and Decion Boundaries');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
hold on; % Mevcut scatter plot üzerine çizim yap

for i = 1:length(cutPoints)
    if ~isnan(cutPoints(i)) % Eğer bir bölme noktası varsa
        if strcmp(cutPredictors{i}, 'x1') % Petal Length (x1)
            xline(cutPoints(i), '--k', 'LineWidth', 1.5); % Dikey çizgi
        elseif strcmp(cutPredictors{i}, 'x2') % Petal Width (x2)
            yline(cutPoints(i), '--k', 'LineWidth', 1.5); % Yatay çizgi
        end
    end
end

legend('Setosa', 'Versicolor', 'Virginica', 'Decision Boundaries');
hold off;


% Train a classification tree with Entropy SplitCriterion
MdlSplit = fitctree(X,Y,'SplitCriterion','deviance','CrossVal','on');

% Plot the tree
view(MdlSplit.Trained{1},'Mode','graph')

% Cross validation classification error
classErrorMdlSplit = kfoldLoss(MdlSplit);

% Classification Region Plot
cutPredictors   = MdlSplit.Trained{1}.CutPredictor;     % Cut Predictors
cutPoints       = MdlSplit.Trained{1}.CutPoint;         % Cut points
% Scatter plot of the original data
figure;
gscatter(X(:,1), X(:,2), species);
title('Original Data Scatter Plot and Decion Boundaries');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
hold on; % Mevcut scatter plot üzerine çizim yap

for i = 1:length(cutPoints)
    if ~isnan(cutPoints(i)) % Eğer bir bölme noktası varsa
        if strcmp(cutPredictors{i}, 'x1') % Petal Length (x1)
            xline(cutPoints(i), '--k', 'LineWidth', 1.5); % Dikey çizgi
        elseif strcmp(cutPredictors{i}, 'x2') % Petal Width (x2)
            yline(cutPoints(i), '--k', 'LineWidth', 1.5); % Yatay çizgi
        end
    end
end

legend('Setosa', 'Versicolor', 'Virginica', 'Decision Boundaries');
hold off;
