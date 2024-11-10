clear; clc; close all;

%% Question 4 %% Optimize an SVM classifier

% Generate data
rng('default') % For reproducibility
grnpop = mvnrnd([1,0],eye(2),10);
redpop = mvnrnd([0,1],eye(2),10);

% View the base points
figure;
plot(grnpop(:,1),grnpop(:,2),'go')
hold on
plot(redpop(:,1),redpop(:,2),'ro')
hold off

% Generate 100 data points
redpts = zeros(100,2);
grnpts = redpts;
for i = 1:100
    grnpts(i,:) = mvnrnd(grnpop(randi(10),:),eye(2)*0.02);
    redpts(i,:) = mvnrnd(redpop(randi(10),:),eye(2)*0.02);
end

% View the data points
figure
plot(grnpts(:,1),grnpts(:,2),'go')
hold on
plot(redpts(:,1),redpts(:,2),'ro')
hold off

% Prepare data for classification

cdata = [grnpts;redpts];
grp = ones(200,1);
grp(101:200) = -1;

% Prepare cross validation
c = cvpartition(200,'KFold',10);

% Optimize fit
opts = struct('CVPartition',c,'AcquisitionFunctionName', ...
    'expected-improvement-plus');
Mdl = fitcsvm(cdata,grp,'KernelFunction','rbf', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
