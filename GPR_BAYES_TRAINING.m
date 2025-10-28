%% GPR_TRAINING_BAYESOPT.m

%% Load data
data = load('cleaned_data.mat'); % Load cleaned dataset
data = data.data;
X = data{:, 1:end-1};
y = data{:, end};
n_samples = size(X,1);

rng(42);

% 85-15 train-test split
cv_holdout = cvpartition(n_samples, 'HoldOut', 0.15);
X_train = X(training(cv_holdout), :);
y_train = y(training(cv_holdout));
X_test  = X(test(cv_holdout), :);
y_test  = y(test(cv_holdout));

fprintf('Training set: %d samples, Test set: %d samples\n', size(X_train,1), size(X_test,1));

%% KERNEL FUNCTIONS EXPLANATION
fprintf('\n========================================\n');
fprintf('KERNEL FUNCTIONS BEING TESTED:\n');
fprintf('========================================\n\n');

fprintf('1. SQUARED EXPONENTIAL (RBF/Gaussian):\n');
fprintf('   - Infinitely differentiable, very smooth predictions\n');
fprintf('   - Best for: Smooth, continuous relationships in data\n');
fprintf('   - Formula: k(x,x'') = exp(-||x-x''||^2 / (2*sigma^2))\n');
fprintf('   - Use case: General-purpose, when you expect smooth trends\n\n');

fprintf('2. MATERN 3/2:\n');
fprintf('   - Once differentiable, moderately smooth\n');
fprintf('   - Best for: Data with some roughness/noise\n');
fprintf('   - Formula: k(x,x'') = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)\n');
fprintf('   - Use case: Engineering data with moderate noise\n\n');

fprintf('3. MATERN 5/2:\n');
fprintf('   - Twice differentiable, smoother than Matern 3/2\n');
fprintf('   - Best for: Balance between smoothness and flexibility\n');
fprintf('   - Formula: k(x,x'') = (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)\n');
fprintf('   - Use case: FEA/simulation data (RECOMMENDED for your case)\n\n');

fprintf('4. EXPONENTIAL:\n');
fprintf('   - Not differentiable, rough predictions\n');
fprintf('   - Best for: Very noisy, irregular data\n');
fprintf('   - Formula: k(x,x'') = exp(-||x-x''|| / sigma)\n');
fprintf('   - Use case: High-frequency variations, signal processing\n\n');

fprintf('5. RATIONAL QUADRATIC:\n');
fprintf('   - Mixture of SE kernels, multi-scale modeling\n');
fprintf('   - Best for: Data with features at multiple length scales\n');
fprintf('   - Formula: k(x,x'') = (1 + ||x-x''||^2 / (2*alpha*sigma^2))^(-alpha)\n');
fprintf('   - Use case: Complex systems with multiple characteristic scales\n\n');

fprintf('For FEA surrogate modeling, Matern 5/2 is often optimal as it\n');
fprintf('captures the smooth but not perfectly smooth nature of physical simulations.\n');
fprintf('========================================\n\n');

%% Setup for Bayesian Optimization
% Define hyperparameters to optimize with MORE CONSERVATIVE bounds
% Note: Standardize is ALWAYS TRUE (not tuned) for numerical stability
optimVars = [
    optimizableVariable('KernelFunction', {'squaredexponential', 'matern32', 'matern52', 'exponential', 'rationalquadratic'}, 'Type', 'categorical')
    optimizableVariable('Sigma', [1e-2, 10], 'Transform', 'log')  % Increased lower bound
    optimizableVariable('SigmaLowerBound', [1e-5, 1e-3], 'Transform', 'log')  % Increased bounds
];

%% Objective function for Bayesian Optimization
objectiveFcn = @(params)gprObjective(params, X_train, y_train);

%% Run Bayesian Optimization
fprintf('\nStarting Bayesian Optimization...\n');
fprintf('Note: Standardization is FORCED to TRUE for numerical stability\n');
fprintf('Random seed: 42 (for reproducibility)\n\n');

% Reset random seed RIGHT BEFORE bayesopt to ensure reproducibility
rng(42, 'twister');

results = bayesopt(objectiveFcn, optimVars, ...
    'Verbose', 1, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...
    'IsObjectiveDeterministic', true, ...  % Changed to true since CV is deterministic
    'UseParallel', false);

%% Extract best hyperparameters
best_params = results.XAtMinObjective;
fprintf('\n===== BEST HYPERPARAMETERS =====\n');
fprintf('Kernel Function: %s\n', char(best_params.KernelFunction));
fprintf('Sigma: %.4e\n', best_params.Sigma);
fprintf('SigmaLowerBound: %.4e\n', best_params.SigmaLowerBound);
fprintf('Standardize: TRUE (forced)\n');
fprintf('Best CV RMSE: %.4f\n', results.MinObjective);

%% Evaluate best config with detailed fold metrics
fprintf('\n===== DETAILED CV METRICS FOR BEST CONFIG =====\n');
cv = cvpartition(size(X_train,1), 'KFold', 5);
cv_rmse = zeros(cv.NumTestSets, 1);
cv_mae  = zeros(cv.NumTestSets, 1);
cv_r2   = zeros(cv.NumTestSets, 1);

for fold = 1:cv.NumTestSets
    X_cv_train = X_train(training(cv, fold), :);
    y_cv_train = y_train(training(cv, fold));
    X_cv_val   = X_train(test(cv, fold), :);
    y_cv_val   = y_train(test(cv, fold));
    
    gpr_model = fitrgp(X_cv_train, y_cv_train, ...
        'KernelFunction', char(best_params.KernelFunction), ...
        'Sigma', best_params.Sigma, ...
        'SigmaLowerBound', best_params.SigmaLowerBound, ...
        'Standardize', true, ...
        'ConstantSigma', true, ...
        'OptimizeHyperparameters', 'none');
    
    y_pred_val = predict(gpr_model, X_cv_val);
    
    cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred_val).^2));
    cv_mae(fold)  = mean(abs(y_cv_val - y_pred_val));
    cv_r2(fold)   = 1 - sum((y_cv_val - y_pred_val).^2)/sum((y_cv_val - mean(y_cv_val)).^2);
    
    fprintf('Fold %d: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', fold, cv_rmse(fold), cv_mae(fold), cv_r2(fold));
end

fprintf('\nCV metrics (mean ± std): RMSE=%.3f ± %.3f, MAE=%.3f ± %.3f, R²=%.3f ± %.3f\n', ...
    mean(cv_rmse), std(cv_rmse), mean(cv_mae), std(cv_mae), mean(cv_r2), std(cv_r2));

%% Retrain best model on full training set
fprintf('\nRetraining best GPR model on full training set...\n');
tic;
best_gpr_model = fitrgp(X_train, y_train, ...
    'KernelFunction', char(best_params.KernelFunction), ...
    'Sigma', best_params.Sigma, ...
    'SigmaLowerBound', best_params.SigmaLowerBound, ...
    'Standardize', true, ...
    'ConstantSigma', true, ...
    'OptimizeHyperparameters', 'none');
train_time = toc;
fprintf('Training completed in %.2f seconds\n', train_time);

%% Evaluate metrics on TRAIN and TEST
fprintf('\nEvaluating model performance...\n');

% Training set predictions
y_pred_train = predict(best_gpr_model, X_train);
train_rmse = sqrt(mean((y_train - y_pred_train).^2));
train_mae  = mean(abs(y_train - y_pred_train));
train_r2   = 1 - sum((y_train - y_pred_train).^2)/sum((y_train - mean(y_train)).^2);

% Test set predictions
y_pred_test  = predict(best_gpr_model, X_test);
test_rmse = sqrt(mean((y_test - y_pred_test).^2));
test_mae  = mean(abs(y_test - y_pred_test));
test_r2   = 1 - sum((y_test - y_pred_test).^2)/sum((y_test - mean(y_test)).^2);

fprintf('\n===== TRAIN METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', train_rmse, train_mae, train_r2);
fprintf('\n===== TEST METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', test_rmse, test_mae, test_r2);

%% Save model and metrics
output_folder = '../fea-surrogate-model/Models';
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

model_filename = fullfile(output_folder, ...
    sprintf('best_gpr_%s_sigma%.2e.mat', char(best_params.KernelFunction), best_params.Sigma));

% Store comprehensive results
save(model_filename, 'best_gpr_model', 'best_params', 'results', ...
    'train_rmse', 'train_mae', 'train_r2', ...
    'test_rmse', 'test_mae', 'test_r2', 'train_time', ...
    'cv_rmse', 'cv_mae', 'cv_r2');

fprintf('\nModel and metrics saved at: %s\n', model_filename);

%% Objective function for cross-validation
function rmse = gprObjective(params, X, y)
    % 5-fold cross-validation
    cv = cvpartition(size(X,1), 'KFold', 5);
    cv_rmse = zeros(cv.NumTestSets, 1);
    cv_mae  = zeros(cv.NumTestSets, 1);
    cv_r2   = zeros(cv.NumTestSets, 1);
    
    fprintf('\n========================================\n');
    fprintf('Testing GPR Configuration:\n');
    fprintf('  Kernel: %s\n', char(params.KernelFunction));
    fprintf('  Sigma: %.4e\n', params.Sigma);
    fprintf('  SigmaLowerBound: %.4e\n', params.SigmaLowerBound);
    fprintf('  Standardize: TRUE\n');
    fprintf('----------------------------------------\n');
    tic;
    
    failed_folds = 0;
    
    for fold = 1:cv.NumTestSets
        % Split data
        X_cv_train = X(training(cv, fold), :);
        y_cv_train = y(training(cv, fold));
        X_cv_val   = X(test(cv, fold), :);
        y_cv_val   = y(test(cv, fold));
        
        % Train GPR model with stability settings
        try
            gpr_model = fitrgp(X_cv_train, y_cv_train, ...
                'KernelFunction', char(params.KernelFunction), ...
                'Sigma', params.Sigma, ...
                'SigmaLowerBound', params.SigmaLowerBound, ...
                'Standardize', true, ...
                'ConstantSigma', true, ...
                'OptimizeHyperparameters', 'none');
            
            % Predict and compute metrics
            y_pred_val = predict(gpr_model, X_cv_val);
            cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred_val).^2));
            cv_mae(fold)  = mean(abs(y_cv_val - y_pred_val));
            cv_r2(fold)   = 1 - sum((y_cv_val - y_pred_val).^2)/sum((y_cv_val - mean(y_cv_val)).^2);
            
            fprintf('  Fold %d: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', fold, cv_rmse(fold), cv_mae(fold), cv_r2(fold));
            
        catch ME
            % If model training fails, return high penalty
            cv_rmse(fold) = 1e6;
            cv_mae(fold)  = 1e6;
            cv_r2(fold)   = -1e6;
            failed_folds = failed_folds + 1;
            fprintf('  Fold %d: FAILED - %s\n', fold, ME.message);
        end
    end
    
    time_elapsed = toc;
    
    % Return mean RMSE across folds
    valid_folds = cv_rmse < 1e6;
    
    if sum(valid_folds) == 0
        rmse = 1e6;
        fprintf('----------------------------------------\n');
        fprintf('ALL FOLDS FAILED! Returning penalty.\n');
        fprintf('========================================\n');
    else
        rmse = mean(cv_rmse(valid_folds));
        mae_mean = mean(cv_mae(valid_folds));
        r2_mean = mean(cv_r2(valid_folds));
        
        fprintf('----------------------------------------\n');
        fprintf('Summary (%d/%d folds successful):\n', sum(valid_folds), cv.NumTestSets);
        fprintf('  Mean RMSE: %.3f ± %.3f\n', rmse, std(cv_rmse(valid_folds)));
        fprintf('  Mean MAE:  %.3f ± %.3f\n', mae_mean, std(cv_mae(valid_folds)));
        fprintf('  Mean R²:   %.3f ± %.3f\n', r2_mean, std(cv_r2(valid_folds)));
        fprintf('  Time: %.2fs\n', time_elapsed);
        fprintf('========================================\n');
    end
end