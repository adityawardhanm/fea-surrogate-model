%% GPR_TRAINING_COMPLETE.m
% Complete GPR training using built-in Bayesian optimization with all required analysis

%% Load data
data = load('cleaned_data.mat'); % Load cleaned dataset
data = data.data;
X = data{:, 1:end-1};
y = data{:, end};
n_samples = size(X,1);
feature_names = data.Properties.VariableNames(1:end-1);

rng(42);

% 85-15 train-test split
cv_holdout = cvpartition(n_samples, 'HoldOut', 0.15);
X_train = X(training(cv_holdout), :);
y_train = y(training(cv_holdout));
X_test  = X(test(cv_holdout), :);
y_test  = y(test(cv_holdout));

fprintf('Training set: %d samples, Test set: %d samples\n', size(X_train,1), size(X_test,1));

%% Create output folders
models_folder = '../fea-surrogate-model/Models';
figures_folder = '../fea-surrogate-model/Figures';
if ~exist(models_folder,'dir')
    mkdir(models_folder);
end
if ~exist(figures_folder,'dir')
    mkdir(figures_folder);
end

%% Train GPR with Built-in Bayesian Optimization
fprintf('\n========== HYPERPARAMETER TUNING ==========\n');
fprintf('Using MATLAB built-in Bayesian Optimization\n');
fprintf('Kernel: Matern 5/2 (fixed based on prior testing)\n');
fprintf('Optimizing: Sigma, KernelScale, and Standardize\n');
fprintf('Random seed: 42 (for reproducibility)\n\n');

% Configure Bayesian optimization options
bayesopt_options = struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...
    'ShowPlots', false, ...
    'Verbose', 1, ...
    'CVPartition', cvpartition(size(X_train,1), 'KFold', 5), ...
    'Repartition', false, ...
    'UseParallel', false);

% Train GPR with automatic hyperparameter optimization
fprintf('Starting Bayesian Optimization...\n');
tic;
best_gpr_model = fitrgp(X_train, y_train, ...
    'KernelFunction', 'matern52', ...
    'Standardize', true, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', bayesopt_options);
train_time = toc;

fprintf('\n========== OPTIMIZATION COMPLETE ==========\n');
fprintf('Training completed in %.2f seconds\n', train_time);
fprintf('\nBest Hyperparameters:\n');
fprintf('  Kernel Function: Matern 5/2\n');
fprintf('  Sigma: %.4e\n', best_gpr_model.Sigma);
kernel_params = best_gpr_model.KernelInformation.KernelParameters;
if length(kernel_params) == 1
    fprintf('  Kernel Scale: %.4f\n', kernel_params(1));
else
    fprintf('  Kernel Parameters: [%.4f, %.4f]\n', kernel_params(1), kernel_params(2));
end
fprintf('  Standardize: TRUE (applied during training)\n');

%% Evaluate with detailed CV metrics on best model
fprintf('\n========== CROSS-VALIDATION METRICS ==========\n');
cv_kfold = cvpartition(size(X_train,1), 'KFold', 5);
cv_rmse = zeros(1, cv_kfold.NumTestSets);
cv_mae  = zeros(1, cv_kfold.NumTestSets);
cv_r2   = zeros(1, cv_kfold.NumTestSets);

for fold = 1:cv_kfold.NumTestSets
    X_cv_train = X_train(training(cv_kfold, fold), :);
    y_cv_train = y_train(training(cv_kfold, fold));
    X_cv_val   = X_train(test(cv_kfold, fold), :);
    y_cv_val   = y_train(test(cv_kfold, fold));
    
    % Train with best hyperparameters
    gpr_cv = fitrgp(X_cv_train, y_cv_train, ...
        'KernelFunction', 'matern52', ...
        'Sigma', best_gpr_model.Sigma, ...
        'KernelParameters', best_gpr_model.KernelInformation.KernelParameters, ...
        'Standardize', true);
    
    y_pred_val = predict(gpr_cv, X_cv_val);
    
    cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred_val).^2));
    cv_mae(fold)  = mean(abs(y_cv_val - y_pred_val));
    cv_r2(fold)   = 1 - sum((y_cv_val - y_pred_val).^2)/sum((y_cv_val - mean(y_cv_val)).^2);
    
    fprintf('Fold %d: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', fold, cv_rmse(fold), cv_mae(fold), cv_r2(fold));
end

fprintf('\nCV metrics (mean ± std): RMSE=%.3f ± %.3f, MAE=%.3f ± %.3f, R²=%.3f ± %.3f\n', ...
    mean(cv_rmse), std(cv_rmse), mean(cv_mae), std(cv_mae), mean(cv_r2), std(cv_r2));

%% Evaluate metrics on TRAIN and TEST
fprintf('\n========== FINAL MODEL EVALUATION ==========\n');

% Training set predictions
y_pred_train = predict(best_gpr_model, X_train);
train_rmse = sqrt(mean((y_train - y_pred_train).^2));
train_mae  = mean(abs(y_train - y_pred_train));
train_r2   = 1 - sum((y_train - y_pred_train).^2)/sum((y_train - mean(y_train)).^2);

% Test set predictions with uncertainty
[y_pred_test, y_std_test] = predict(best_gpr_model, X_test);
test_rmse = sqrt(mean((y_test - y_pred_test).^2));
test_mae  = mean(abs(y_test - y_pred_test));
test_r2   = 1 - sum((y_test - y_pred_test).^2)/sum((y_test - mean(y_test)).^2);

fprintf('\n===== TRAIN METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', train_rmse, train_mae, train_r2);
fprintf('\n===== TEST METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', test_rmse, test_mae, test_r2);
fprintf('Mean Prediction Std Dev: %.3f\n', mean(y_std_test));

%% ========================================
%% REQUIRED: RESIDUAL ANALYSIS
%% ========================================
fprintf('\n========== RESIDUAL ANALYSIS ==========\n');
residuals_train = y_train - y_pred_train;
residuals_test = y_test - y_pred_test;

figure('Position', [100, 100, 1400, 900]);

% Test Set Residuals
subplot(2,3,1);
scatter(y_pred_test, residuals_test, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
yline(0, 'r--', 'LineWidth', 2);
xlabel('Predicted Values');
ylabel('Residuals');
title('Test Set: Residuals vs Predicted');
grid on;

subplot(2,3,2);
histogram(residuals_test, 30, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
hold on;
mu = mean(residuals_test);
sigma = std(residuals_test);
x_norm = linspace(min(residuals_test), max(residuals_test), 100);
plot(x_norm, normpdf(x_norm, mu, sigma), 'r-', 'LineWidth', 2);
xlabel('Residuals');
ylabel('Density');
title(sprintf('Test Set: Residual Distribution (μ=%.3f, σ=%.3f)', mu, sigma));
legend('Residuals', 'Normal Fit');
grid on;

subplot(2,3,3);
qqplot(residuals_test);
title('Test Set: Q-Q Plot');
grid on;

% Train Set Residuals
subplot(2,3,4);
scatter(y_pred_train, residuals_train, 50, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
yline(0, 'r--', 'LineWidth', 2);
xlabel('Predicted Values');
ylabel('Residuals');
title('Train Set: Residuals vs Predicted');
grid on;

subplot(2,3,5);
histogram(residuals_train, 30, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
hold on;
mu_train = mean(residuals_train);
sigma_train = std(residuals_train);
x_norm_train = linspace(min(residuals_train), max(residuals_train), 100);
plot(x_norm_train, normpdf(x_norm_train, mu_train, sigma_train), 'r-', 'LineWidth', 2);
xlabel('Residuals');
ylabel('Density');
title(sprintf('Train Set: Residual Distribution (μ=%.3f, σ=%.3f)', mu_train, sigma_train));
legend('Residuals', 'Normal Fit');
grid on;

subplot(2,3,6);
qqplot(residuals_train);
title('Train Set: Q-Q Plot');
grid on;

sgtitle('Residual Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'gpr_residual_analysis.png'));
fprintf('Residual analysis plots saved.\n');

%% ========================================
%% GPR-SPECIFIC: PREDICTIVE UNCERTAINTY (Mean ± 2×SD)
%% ========================================
fprintf('\n========== PREDICTIVE UNCERTAINTY ANALYSIS ==========\n');

% Calculate prediction intervals (mean ± 2*SD ≈ 95% confidence)
y_lower = y_pred_test - 2*y_std_test;
y_upper = y_pred_test + 2*y_std_test;

% Calculate coverage (what % of true values fall within intervals)
coverage = mean((y_test >= y_lower) & (y_test <= y_upper));
fprintf('95%% Prediction Interval Coverage: %.2f%%\n', coverage * 100);
fprintf('Mean Uncertainty (±2SD): %.3f\n', mean(2*y_std_test));

% Plot 1: Prediction intervals for subset of test samples
num_samples_plot = min(100, length(y_test));
[~, sort_idx_plot] = sort(y_test);
idx_plot = sort_idx_plot(1:num_samples_plot);

figure('Position', [100, 100, 1400, 1000]);

subplot(2,2,1);
x_axis = 1:num_samples_plot;
fill([x_axis, fliplr(x_axis)], ...
     [y_lower(idx_plot)', fliplr(y_upper(idx_plot)')], ...
     [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
plot(x_axis, y_test(idx_plot), 'bo', 'MarkerSize', 6, 'DisplayName', 'True Values');
plot(x_axis, y_pred_test(idx_plot), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predictions');
xlabel('Sample Index (sorted by true value)');
ylabel('Target Value');
title(sprintf('95%% Prediction Intervals (Coverage: %.1f%%)', coverage * 100));
legend('95% PI', 'True Values', 'Predictions', 'Location', 'best');
grid on;

% Plot 2: Predicted vs True with error bars
subplot(2,2,2);
errorbar(y_test, y_pred_test, 2*y_std_test, 'o', 'MarkerSize', 4, ...
    'MarkerFaceColor', [0.2 0.6 0.8], 'MarkerEdgeColor', 'none', ...
    'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', 'LineWidth', 2);
xlabel('True Values');
ylabel('Predicted Values ± 2SD');
title('Predictions with Uncertainty');
grid on;
axis equal;

% Plot 3: Uncertainty vs Prediction Error
subplot(2,2,3);
abs_errors = abs(y_test - y_pred_test);
scatter(y_std_test, abs_errors, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Prediction Std Dev');
ylabel('Absolute Error');
title('Uncertainty vs Prediction Error');
grid on;

% Plot 4: Distribution of uncertainties
subplot(2,2,4);
histogram(y_std_test, 30, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Prediction Std Dev');
ylabel('Frequency');
title(sprintf('Distribution of Uncertainties (mean=%.3f)', mean(y_std_test)));
grid on;

sgtitle('GPR Predictive Uncertainty Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'gpr_predictive_uncertainty.png'));
fprintf('Predictive uncertainty plots saved.\n');

%% ========================================
%% PARTIAL DEPENDENCE PLOTS
%% ========================================
fprintf('\n========== PARTIAL DEPENDENCE PLOTS ==========\n');

% Get feature importance through sensitivity analysis
% For GPR, we compute how much variance in predictions comes from each feature
n_features = size(X_train, 2);
feature_sensitivity = zeros(n_features, 1);

for i = 1:n_features
    X_perturbed = X_train;
    X_perturbed(:, i) = mean(X_train(:, i)); % Set feature to mean
    y_pred_perturbed = predict(best_gpr_model, X_perturbed);
    feature_sensitivity(i) = var(y_pred_train - y_pred_perturbed);
end

% Normalize
feature_sensitivity = feature_sensitivity / sum(feature_sensitivity);

% Sort features by sensitivity
[sensitivity_sorted, sort_idx] = sort(feature_sensitivity, 'descend');
features_sorted = feature_names(sort_idx);

% Plot feature sensitivity
figure('Position', [100, 100, 1000, 600]);
bar(sensitivity_sorted, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:length(features_sorted), 'XTickLabel', features_sorted);
xtickangle(45);
ylabel('Sensitivity Score');
title('Feature Sensitivity Analysis (GPR)');
grid on;
saveas(gcf, fullfile(figures_folder, 'gpr_feature_sensitivity.png'));

fprintf('Top 5 most sensitive features:\n');
for i = 1:min(5, length(features_sorted))
    fprintf('  %d. %s: %.4f\n', i, features_sorted{i}, sensitivity_sorted(i));
end

% Plot PDPs for top 4 most sensitive features
num_pdp = min(4, length(features_sorted));
figure('Position', [100, 100, 1200, 800]);

for i = 1:num_pdp
    feat_idx = sort_idx(i);
    feat_name = features_sorted{i};
    
    subplot(2, 2, i);
    plotPartialDependence(best_gpr_model, feat_idx, X_train);
    title(sprintf('PDP: %s', feat_name));
    xlabel(feat_name);
    ylabel('Partial Effect on Prediction');
    grid on;
end

sgtitle('Partial Dependence Plots (Top 4 Features)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'gpr_partial_dependence_plots.png'));
fprintf('Partial dependence plots saved.\n');

%% 2D Partial Dependence Surface Plot (Top 2 Features)
fprintf('Creating 2D partial dependence surface plot...\n');

% Get top 2 features
top2_indices = sort_idx(1:2);
feat1_name = features_sorted{1};
feat2_name = features_sorted{2};

figure('Position', [100, 100, 900, 700]);
plotPartialDependence(best_gpr_model, top2_indices, X_train);
title(sprintf('2D Partial Dependence: %s vs %s', feat1_name, feat2_name), 'FontSize', 14, 'FontWeight', 'bold');
xlabel(feat1_name, 'FontSize', 12);
ylabel(feat2_name, 'FontSize', 12);
zlabel('Partial Effect on Prediction', 'FontSize', 12);
view(45, 30); % 3D view angle
colorbar;
grid on;

saveas(gcf, fullfile(figures_folder, 'gpr_2d_partial_dependence.png'));
fprintf('2D partial dependence surface plot saved.\n');

%% ========================================
%% SAVE MODEL AND ALL RESULTS
%% ========================================
fprintf('\n========== SAVING RESULTS ==========\n');

model_filename = fullfile(models_folder, 'best_gpr_model_matern52.mat');

% Save comprehensive results
save(model_filename, 'best_gpr_model', ...
    'train_rmse', 'train_mae', 'train_r2', ...
    'test_rmse', 'test_mae', 'test_r2', ...
    'cv_rmse', 'cv_mae', 'cv_r2', ...
    'train_time', 'coverage', ...
    'feature_sensitivity', 'feature_names', ...
    'y_pred_test', 'y_std_test', ...
    'residuals_train', 'residuals_test');

fprintf('Model and all metrics saved at: %s\n', model_filename);

%% ========================================
%% SUMMARY REPORT
%% ========================================
fprintf('\n========================================\n');
fprintf('         TRAINING COMPLETE\n');
fprintf('========================================\n');
fprintf('Best Configuration:\n');
fprintf('  - Kernel: Matern 5/2\n');
fprintf('  - Sigma: %.4e\n', best_gpr_model.Sigma);
kernel_params = best_gpr_model.KernelInformation.KernelParameters;
if isscalar(kernel_params)
    fprintf('  - Kernel Scale: %.4f\n', kernel_params(1));
else
    fprintf('  - Kernel Parameters: [%.4f, %.4f]\n', kernel_params(1), kernel_params(2));
end
fprintf('  - Standardize: TRUE\n');
fprintf('\nPerformance Metrics:\n');
fprintf('  Train: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', train_rmse, train_mae, train_r2);
fprintf('  Test:  RMSE=%.3f, MAE=%.3f, R²=%.3f\n', test_rmse, test_mae, test_r2);
fprintf('  CV:    RMSE=%.3f ± %.3f\n', mean(cv_rmse), std(cv_rmse));
fprintf('\nUncertainty Quantification:\n');
fprintf('  95%% Coverage: %.2f%%\n', coverage * 100);
fprintf('  Mean Uncertainty: ±%.3f\n', mean(2*y_std_test));
fprintf('\nTraining Time: %.2f seconds\n', train_time);
fprintf('\nModel saved to: %s\n', models_folder);
fprintf('Figures saved to: %s\n', figures_folder);
fprintf('========================================\n');