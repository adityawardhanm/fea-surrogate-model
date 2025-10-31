%% SVR_TRAINING_BAYESOPT.m
% SVR TRAINING USING BAYESIAN OPTIMIZATION AND FIVE-FOLD CROSS-VALIDATION

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

%% Setup for Bayesian Optimization
% Define hyperparameters to optimize for SVR
% Note: Standardize is ALWAYS TRUE (not tuned) for numerical stability
optimVars = [
    optimizableVariable('KernelFunction', {'linear', 'gaussian', 'polynomial'}, 'Type', 'categorical')
    optimizableVariable('BoxConstraint', [1e-2, 1e3], 'Transform', 'log')
    optimizableVariable('Epsilon', [1e-3, 1], 'Transform', 'log')
    optimizableVariable('KernelScale', [1e-1, 1e2], 'Transform', 'log')
];

%% Objective function for Bayesian Optimization
objectiveFcn = @(params)svrObjective(params, X_train, y_train);

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
    'IsObjectiveDeterministic', true, ...
    'UseParallel', false);

%% Extract best hyperparameters
best_params = results.XAtMinObjective;
fprintf('\n===== BEST HYPERPARAMETERS =====\n');
fprintf('Kernel Function: %s\n', char(best_params.KernelFunction));
fprintf('BoxConstraint: %.4e\n', best_params.BoxConstraint);
fprintf('Epsilon: %.4e\n', best_params.Epsilon);
fprintf('KernelScale: %.4e\n', best_params.KernelScale);
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
    
    svr_model = fitrsvm(X_cv_train, y_cv_train, ...
        'KernelFunction', char(best_params.KernelFunction), ...
        'BoxConstraint', best_params.BoxConstraint, ...
        'Epsilon', best_params.Epsilon, ...
        'KernelScale', best_params.KernelScale, ...
        'Standardize', true);
    
    y_pred_val = predict(svr_model, X_cv_val);
    
    cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred_val).^2));
    cv_mae(fold)  = mean(abs(y_cv_val - y_pred_val));
    cv_r2(fold)   = 1 - sum((y_cv_val - y_pred_val).^2)/sum((y_cv_val - mean(y_cv_val)).^2);
    
    fprintf('Fold %d: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', fold, cv_rmse(fold), cv_mae(fold), cv_r2(fold));
end

fprintf('\nCV metrics (mean ± std): RMSE=%.3f ± %.3f, MAE=%.3f ± %.3f, R²=%.3f ± %.3f\n', ...
    mean(cv_rmse), std(cv_rmse), mean(cv_mae), std(cv_mae), mean(cv_r2), std(cv_r2));

%% Learning Curve Analysis
fprintf('\n===== GENERATING LEARNING CURVES =====\n');
train_sizes = round(linspace(0.1*size(X_train,1), size(X_train,1), 10));
lc_train_rmse = zeros(length(train_sizes), 1);
lc_val_rmse = zeros(length(train_sizes), 1);
lc_train_r2 = zeros(length(train_sizes), 1);
lc_val_r2 = zeros(length(train_sizes), 1);

for i = 1:length(train_sizes)
    n_train = train_sizes(i);
    
    % Use subset of training data
    idx_subset = randperm(size(X_train,1), n_train);
    X_subset = X_train(idx_subset, :);
    y_subset = y_train(idx_subset);
    
    % Train model on subset
    model_temp = fitrsvm(X_subset, y_subset, ...
        'KernelFunction', char(best_params.KernelFunction), ...
        'BoxConstraint', best_params.BoxConstraint, ...
        'Epsilon', best_params.Epsilon, ...
        'KernelScale', best_params.KernelScale, ...
        'Standardize', true);
    
    % Evaluate on training subset
    y_pred_train_subset = predict(model_temp, X_subset);
    lc_train_rmse(i) = sqrt(mean((y_subset - y_pred_train_subset).^2));
    lc_train_r2(i) = 1 - sum((y_subset - y_pred_train_subset).^2)/sum((y_subset - mean(y_subset)).^2);
    
    % Evaluate on full validation set (test set)
    y_pred_val = predict(model_temp, X_test);
    lc_val_rmse(i) = sqrt(mean((y_test - y_pred_val).^2));
    lc_val_r2(i) = 1 - sum((y_test - y_pred_val).^2)/sum((y_test - mean(y_test)).^2);
    
    fprintf('Train size: %d, Train RMSE: %.3f, Val RMSE: %.3f\n', n_train, lc_train_rmse(i), lc_val_rmse(i));
end

% Plot learning curves
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
plot(train_sizes, lc_train_rmse, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Training');
hold on;
plot(train_sizes, lc_val_rmse, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Validation');
xlabel('Training Set Size', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('RMSE', 'FontSize', 12, 'FontWeight', 'bold');
title('Learning Curve - RMSE', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11);

subplot(1, 2, 2);
plot(train_sizes, lc_train_r2, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Training');
hold on;
plot(train_sizes, lc_val_r2, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Validation');
xlabel('Training Set Size', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('R²', 'FontSize', 12, 'FontWeight', 'bold');
title('Learning Curve - R²', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11);

% Save learning curve figure
output_folder = '../fea-surrogate-model/Models';
saveas(gcf, fullfile(output_folder, 'svr_learning_curves.png'));
fprintf('Learning curves saved.\n');

%% Retrain best model on full training set
fprintf('\nRetraining best SVR model on full training set...\n');
tic;
best_svr_model = fitrsvm(X_train, y_train, ...
    'KernelFunction', char(best_params.KernelFunction), ...
    'BoxConstraint', best_params.BoxConstraint, ...
    'Epsilon', best_params.Epsilon, ...
    'KernelScale', best_params.KernelScale, ...
    'Standardize', true);
train_time = toc;
fprintf('Training completed in %.2f seconds\n', train_time);

%% Evaluate metrics on TRAIN and TEST
fprintf('\nEvaluating model performance...\n');

% Training set predictions
y_pred_train = predict(best_svr_model, X_train);
train_rmse = sqrt(mean((y_train - y_pred_train).^2));
train_mae  = mean(abs(y_train - y_pred_train));
train_r2   = 1 - sum((y_train - y_pred_train).^2)/sum((y_train - mean(y_train)).^2);

% Test set predictions
y_pred_test  = predict(best_svr_model, X_test);
test_rmse = sqrt(mean((y_test - y_pred_test).^2));
test_mae  = mean(abs(y_test - y_pred_test));
test_r2   = 1 - sum((y_test - y_pred_test).^2)/sum((y_test - mean(y_test)).^2);

fprintf('\n===== TRAIN METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', train_rmse, train_mae, train_r2);
fprintf('\n===== TEST METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', test_rmse, test_mae, test_r2);

%% Save model and metrics
model_filename = fullfile(output_folder, ...
    sprintf('best_svr_model.mat' ));

% Store comprehensive results including learning curve data
save(model_filename, 'best_svr_model', 'best_params', 'results', ...
    'train_rmse', 'train_mae', 'train_r2', ...
    'test_rmse', 'test_mae', 'test_r2', 'train_time', ...
    'cv_rmse', 'cv_mae', 'cv_r2', ...
    'train_sizes', 'lc_train_rmse', 'lc_val_rmse', 'lc_train_r2', 'lc_val_r2');

fprintf('\nModel and metrics saved at: %s\n', model_filename);

%% Objective function for cross-validation
function rmse = svrObjective(params, X, y)
    % 5-fold cross-validation
    cv = cvpartition(size(X,1), 'KFold', 5);
    cv_rmse = zeros(cv.NumTestSets, 1);
    cv_mae  = zeros(cv.NumTestSets, 1);
    cv_r2   = zeros(cv.NumTestSets, 1);
    
    fprintf('\n========================================\n');
    fprintf('Testing SVR Configuration:\n');
    fprintf('  Kernel: %s\n', char(params.KernelFunction));
    fprintf('  BoxConstraint (C): %.4e\n', params.BoxConstraint);
    fprintf('  Epsilon: %.4e\n', params.Epsilon);
    fprintf('  KernelScale: %.4e\n', params.KernelScale);
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
        
        % Train SVR model
        try
            svr_model = fitrsvm(X_cv_train, y_cv_train, ...
                'KernelFunction', char(params.KernelFunction), ...
                'BoxConstraint', params.BoxConstraint, ...
                'Epsilon', params.Epsilon, ...
                'KernelScale', params.KernelScale, ...
                'Standardize', true);
            
            % Predict and compute metrics
            y_pred_val = predict(svr_model, X_cv_val);
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