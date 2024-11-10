% Step 1: Load Fisher Iris Data
load fisheriris
inds = ~strcmp(species, 'setosa');  % Exclude 'virginica' class
X = meas(inds, :);                     % Use all 4 features
y = species(inds);                     % Only 'setosa' and 'versicolor' labels

% Step 2: 10-Fold Cross-Validation
SVMModel_10fold = fitcsvm(X, y, 'Standardize', true, 'Kfold', 10);
cvError_10fold = kfoldLoss(SVMModel_10fold);  % Misclassification error for 10-fold CV

% Display 10-fold cross-validation error
fprintf('10-Fold Cross-Validation Error: %.4f\n', cvError_10fold);

% Step 3: Leave-One-Out Cross-Validation (LOOCV)
SVMModel_LOOCV = fitcsvm(X, y, 'Standardize', true, 'Leaveout', 'on');
cvError_LOOCV = kfoldLoss(SVMModel_LOOCV);   % Misclassification error for LOOCV

% Display LOOCV error
fprintf('Leave-One-Out Cross-Validation Error: %.4f\n', cvError_LOOCV);

% Step 4: Compare Results
fprintf('\nComparison of Cross-Validation Results:\n');
fprintf('10-Fold Cross-Validation Error: %.4f\n', cvError_10fold);
fprintf('Leave-One-Out Cross-Validation Error: %.4f\n', cvError_LOOCV);