clc; clear; close all;
%% Question 4 Jackknife sampling %% 

% First compute the sample correlation on the data
load lawdata
rhohat = corr(lsat,gpa);

plot(lsat,gpa,'+')
lsline
xlabel('LSAT Scores')   % Label x-axis
ylabel('GPA Scores')    % Label y-axis
title('LSAT vs GPA')    % Plot title
grid on                 % Add grid for better visualization

% Next compute the correlations for jackknife samples, and compute their mean
rng default;  % For reproducibility
jackrho = jackknife(@corr,lsat,gpa);
meanrho = mean(jackrho);

% Compute an estimate of the bias
n = length(lsat);
biasrho = (n-1) * (meanrho-rhohat);

% histogram(jackrho,30,'FaceColor',[.8 .8 1])

% Median Calculation for LSAT
medhat_lsat = median(lsat);
jackmed_lsat = jackknife(@median,lsat);

biasmed_lsat = (n-1) * (mean(jackmed_lsat)- (medhat_lsat));

% Median Calculation for GPA
medhat_gpa = median(gpa);
jackmed_gpa = jackknife(@median,gpa);

biasmed_gpa = (n-1) * (mean(jackmed_gpa)- mean(medhat_gpa));

disp(['Sample Mean : ', num2str((rhohat))]);
disp(['Mean Jackknife Correlation Coefficient: ', num2str(meanrho)]);
disp(['Jackknife Estimate of Bias: ', num2str(biasrho)]);


disp(['Sample Median LSAT: ', num2str((medhat_lsat))]);
disp(['Jackknife Average Median LSAT: ', num2str(mean(jackmed_lsat))]);
disp(['Jackknife Bias LSAT: ', num2str(biasmed_lsat)]);

disp(['Sample Median GPA: ', num2str(medhat_gpa)]);
disp(['Jackknife Average Median GPA: ', num2str(mean(jackmed_gpa))]);
disp(['Jackknife Bias GPA: ', num2str(biasmed_gpa)]);
