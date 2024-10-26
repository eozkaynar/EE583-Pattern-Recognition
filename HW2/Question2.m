%% Question 2 %% Bayesian Parameter Estimation:
rng('default') % For reproducibility

sigma = 0.7;
% i)
mu1     = 3;
N_25    = 25;
x_i     = normrnd(mu1,sigma,[N_25,1]);


% define values for y-axis
y       = zeros(length(x_i),1);
group   = ones(length(x_i), 1);  % All samples in one group
% Plot
figure;
gscatter(x_i, y, group, 'br', '.',18);

% Maximum Likelihood Estimation

mean_estimator_i = sum(x_i)/N_25;

%ii)
mu_mu       = 2.8; % mean parameter of random variable mean
sigma_mu    = .8; % variance parameter of random variable mean

w1_ii       = (N_25*sigma_mu^2)/(N_25*sigma_mu^2 + sigma^2);
w2_ii       = (sigma^2)/(N_25*sigma_mu^2 + sigma^2);

mu_map_ii   = w1_ii * mean_estimator_i + w2_ii * mu_mu;
    
% iii)

N_1000  = 1000;
x_iii   = normrnd(mu1,sigma,[N_1000,1]);

% define values for y-axis
y       = zeros(length(x_iii),1);
group   = ones(length(x_iii), 1);  % All samples in one group
% Plot
figure;
gscatter(x_iii, y, group, 'br', '.',18);

% Maximum Likelihood Estimation
mean_estimator_iii = sum(x_iii)/N_1000;

% MAP Estimation
w1_iii          = (N_1000*sigma_mu^2)/(N_1000*sigma_mu^2 + sigma^2);
w2_iii          = (sigma^2)/(N_1000*sigma_mu^2 + sigma^2);

mu_map_iii      = w1_iii*mean_estimator_iii + w2_iii * mu_mu; % Prior infoya daha yakın yorum yazarken aklında bulundurursun kıps :D

