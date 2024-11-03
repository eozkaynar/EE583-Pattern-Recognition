clear; clc; close all;

%% Question 4 %%  Non-parametric Density Estimation
% Parameters of normal distributed feature vector
mu          = [-1.5; 1.5];
sigma       = [.8 .2; .2 .6];

% Number of samples
N           = 50;  % Reduced sample size for faster computation

% Grid for plotting the Gaussian distribution
[X1, X2]    = meshgrid(linspace(-6, 6, 500), linspace(-6, 6, 500));
X           = [X1(:) X2(:)];

% True distribution
gaus_true   = mvnpdf([X1(:) X2(:)], mu', sigma);
gaus_true   = reshape(gaus_true, length(X2), length(X1));

% Plot true distribution
figure;
surf(X1, X2, gaus_true, 'EdgeColor', 'interp');
title('True Distribution')
xlabel('X1')
ylabel('X2')
zlabel('PDF')

% Generate samples from the distribution
X_samples = mvnrnd(mu, sigma, N);

% Different h1 values for testing
h1_values = [0.5, 1, 2, 3, 5, 10];
num_h1 = length(h1_values);

% Loop over each h1 value and perform Parzen window density estimation

% Gaussian Parzen window estimation 
figure;
for idx = 1:num_h1
    h1  = h1_values(idx);
    % Initial volume
    Vo  = h1^3;
    % Adaptive volume and h
    V   = Vo / sqrt(N);
    h   = V^(1/3);
    
    % Estimate density with Gaussian Parzen window
    p_gaussian = zeros(size(X,1), 1);   % Initialize density function
    for k = 1:size(X,1)
        % Take one feature from true distribution
        x   = X(k, :);
      
        sum = 0;
        for n = 1:N
            % Calculate each samples effect 
            xk  = X_samples(n, :);
            sum = sum + parzen_window((x - xk) / h);

        end
        p_gaussian(k) = sum / (N * V);
    end
    p_gaussian = reshape(p_gaussian, length(X2), length(X1));

    % Plot the Gaussian Parzen window estimate for the current h1
    subplot(2, 3, idx);  % Arrange in a 2x3 grid
    surf(X1, X2, p_gaussian, 'EdgeColor', 'interp');
    title(['Gaussian Parzen (h1 = ', num2str(h1), ')(V = ', num2str(V), ')']);
    xlabel('X1');
    ylabel('X2');
    zlabel('PDF');
end

% Cubic Parzen window estimation
figure;
for idx = 1:num_h1
    h1 = h1_values(idx);
    Vo = h1^3;
    V = Vo / sqrt(N);
    h = V^(1/3);

    % Estimate density with cubic Parzen window
    p_cubic = zeros(size(X,1), 1);
    for k = 1:size(X,1)
        x = X(k, :);
        sum = 0;
        for n = 1:N
            xk = X_samples(n, :);
            sum = sum + parzen_window_cub((x - xk) / h);
        end
        p_cubic(k) = sum / (N * V);
    end
    p_cubic = reshape(p_cubic, length(X2), length(X1));

    % Plot the Cubic Parzen window estimate for the current h1
    subplot(2, 3, idx);  % Arrange in a 2x3 grid
    surf(X1, X2, p_cubic, 'EdgeColor', 'interp');
    title(['Cubic Parzen (h1 = ', num2str(h1), ')(V = ', num2str(V), ')']);
    xlabel('X1');
    ylabel('X2');
    zlabel('PDF');
end

% Gaussian shape Parzen window function
function f = parzen_window(u)
    u = norm(u);
    f = (1/sqrt(2 * pi)) * exp(-0.5 * u^2);
end

% Cubic Parzen window function
function f_cub = parzen_window_cub(u)
    if norm(u) <= 1/2
        f_cub = 1;
    else
        f_cub = 0;
    end
end
