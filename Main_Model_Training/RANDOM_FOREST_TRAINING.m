%% RANDOM_FOREST_TRAINING_COMPLETE.m
% Complete Random Forest training script with all required analysis

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

%% 5-fold CV setup
cv_kfold = cvpartition(size(X_train,1), 'KFold', 5);

%% Hyperparameter grid
numTreesList = [50, 100, 200];
leafSizeList = [1, 5, 10];

best_rmse = inf;
best_params = struct();

% Storage for learning curves
learning_curve_data = struct();
param_idx = 1;

%% Grid search with CV metrics
fprintf('\n========== HYPERPARAMETER TUNING ==========\n');
for nTrees = numTreesList
    for leafSize = leafSizeList
        % Store metrics per fold
        cv_rmse = zeros(1, cv_kfold.NumTestSets);
        cv_mae  = zeros(1, cv_kfold.NumTestSets);
        cv_r2   = zeros(1, cv_kfold.NumTestSets);

        fprintf('Testing RF: %d trees, MinLeafSize=%d\n', nTrees, leafSize);
        tic;
        for fold = 1:cv_kfold.NumTestSets
            % Split fold
            X_cv_train = X_train(training(cv_kfold, fold), :);
            y_cv_train = y_train(training(cv_kfold, fold));
            X_cv_val   = X_train(test(cv_kfold, fold), :);
            y_cv_val   = y_train(test(cv_kfold, fold));

            % Train model
            t_rf = templateTree('MinLeafSize', leafSize, 'Surrogate', 'on');
            m = fitrensemble(X_cv_train, y_cv_train, ...
                'Method', 'Bag', ...
                'NumLearningCycles', nTrees, ...
                'Learners', t_rf);

            % Predict on validation
            y_pred_val = predict(m, X_cv_val);

            % Compute metrics
            cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred_val).^2));
            cv_mae(fold)  = mean(abs(y_cv_val - y_pred_val));
            cv_r2(fold)   = 1 - sum((y_cv_val - y_pred_val).^2) / sum((y_cv_val - mean(y_cv_val)).^2);
        end
        time_elapsed = toc;

        fprintf(' CV metrics (mean ± std): RMSE=%.3f ± %.3f, MAE=%.3f ± %.3f, R²=%.3f ± %.3f (%.2fs)\n', ...
            mean(cv_rmse), std(cv_rmse), mean(cv_mae), std(cv_mae), mean(cv_r2), std(cv_r2), time_elapsed);

        % Store for learning curves
        learning_curve_data(param_idx).nTrees = nTrees;
        learning_curve_data(param_idx).leafSize = leafSize;
        learning_curve_data(param_idx).cv_rmse_mean = mean(cv_rmse);
        learning_curve_data(param_idx).cv_rmse_std = std(cv_rmse);
        learning_curve_data(param_idx).cv_mae_mean = mean(cv_mae);
        learning_curve_data(param_idx).cv_r2_mean = mean(cv_r2);
        param_idx = param_idx + 1;

        % Track best model by mean CV RMSE
        if mean(cv_rmse) < best_rmse
            best_rmse = mean(cv_rmse);
            best_params.NumTrees    = nTrees;
            best_params.MinLeafSize = leafSize;
            best_params.CV_RMSE    = cv_rmse;
            best_params.CV_MAE     = cv_mae;
            best_params.CV_R2      = cv_r2;
        end
    end
end

fprintf('\n========== BEST CONFIGURATION ==========\n');
fprintf('Best RF config: %d trees, MinLeafSize=%d\n', ...
    best_params.NumTrees, best_params.MinLeafSize);
fprintf('Mean CV RMSE: %.3f ± %.3f\n', mean(best_params.CV_RMSE), std(best_params.CV_RMSE));

%% Retrain best model on full training set
fprintf('\n========== FINAL MODEL TRAINING ==========\n');
fprintf('Retraining best RF on full training set...\n');
t_final = templateTree('MinLeafSize', best_params.MinLeafSize, 'Surrogate','on');
best_rf_model = fitrensemble(X_train, y_train, ...
    'Method','Bag', ...
    'NumLearningCycles', best_params.NumTrees, ...
    'Learners', t_final);

%% Evaluate metrics on TRAIN and TEST
y_pred_train = predict(best_rf_model, X_train);
y_pred_test  = predict(best_rf_model, X_test);

train_rmse = sqrt(mean((y_train - y_pred_train).^2));
train_mae  = mean(abs(y_train - y_pred_train));
train_r2   = 1 - sum((y_train - y_pred_train).^2)/sum((y_train - mean(y_train)).^2);

test_rmse = sqrt(mean((y_test - y_pred_test).^2));
test_mae  = mean(abs(y_test - y_pred_test));
test_r2   = 1 - sum((y_test - y_pred_test).^2)/sum((y_test - mean(y_test)).^2);

fprintf('\n===== TRAIN METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', train_rmse, train_mae, train_r2);
fprintf('\n===== TEST METRICS =====\nRMSE: %.3f\nMAE: %.3f\nR²: %.3f\n', test_rmse, test_mae, test_r2);

%% Create output folders
models_folder = '../fea-surrogate-model/Models';
figures_folder = '../fea-surrogate-model/Figures';
if ~exist(models_folder,'dir')
    mkdir(models_folder);
end
if ~exist(figures_folder,'dir')
    mkdir(figures_folder);
end

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
saveas(gcf, fullfile(figures_folder, 'rf_residual_analysis.png'));
fprintf('Residual analysis plots saved.\n');

%% ========================================
%% REQUIRED: FEATURE IMPORTANCE
%% ========================================
fprintf('\n========== FEATURE IMPORTANCE ==========\n');
importance = predictorImportance(best_rf_model);

% Sort features by importance
[importance_sorted, sort_idx] = sort(importance, 'descend');
features_sorted = feature_names(sort_idx);

figure('Position', [100, 100, 1000, 600]);
bar(importance_sorted, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:length(features_sorted), 'XTickLabel', features_sorted);
xtickangle(45);
ylabel('Importance Score');
title('Feature Importance (Sorted)');
grid on;
saveas(gcf, fullfile(figures_folder, 'rf_feature_importance.png'));

fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(features_sorted))
    fprintf('  %d. %s: %.4f\n', i, features_sorted{i}, importance_sorted(i));
end

%% ========================================
%% REQUIRED: LEARNING CURVES
%% ========================================
fprintf('\n========== LEARNING CURVES ==========\n');

% Extract data for plotting
nTrees_vals = [learning_curve_data.nTrees];
leafSize_vals = [learning_curve_data.leafSize];
cv_rmse_vals = [learning_curve_data.cv_rmse_mean];
cv_rmse_std_vals = [learning_curve_data.cv_rmse_std];

figure('Position', [100, 100, 1400, 500]);

% Learning curve: RMSE vs Number of Trees
subplot(1,2,1);
unique_leafSizes = unique(leafSizeList);
for ls = unique_leafSizes
    idx = leafSize_vals == ls;
    trees = nTrees_vals(idx);
    rmse = cv_rmse_vals(idx);
    rmse_std = cv_rmse_std_vals(idx);
    
    errorbar(trees, rmse, rmse_std, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', sprintf('LeafSize=%d', ls));
    hold on;
end
xlabel('Number of Trees');
ylabel('CV RMSE');
title('Learning Curve: RMSE vs Number of Trees');
legend('Location', 'best');
grid on;

% Learning curve: RMSE vs Leaf Size
subplot(1,2,2);
unique_nTrees = unique(numTreesList);
for nt = unique_nTrees
    idx = nTrees_vals == nt;
    leafs = leafSize_vals(idx);
    rmse = cv_rmse_vals(idx);
    rmse_std = cv_rmse_std_vals(idx);
    
    errorbar(leafs, rmse, rmse_std, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', sprintf('NumTrees=%d', nt));
    hold on;
end
xlabel('Min Leaf Size');
ylabel('CV RMSE');
title('Learning Curve: RMSE vs Min Leaf Size');
legend('Location', 'best');
grid on;

sgtitle('Hyperparameter Learning Curves', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'rf_learning_curves.png'));
fprintf('Learning curves saved.\n');

%% ========================================
%% REQUIRED: PARTIAL DEPENDENCE PLOTS
%% ========================================
fprintf('\n========== PARTIAL DEPENDENCE PLOTS ==========\n');

% Plot PDPs for top 4 most important features
num_pdp = min(4, length(features_sorted));
figure('Position', [100, 100, 1200, 800]);

for i = 1:num_pdp
    feat_idx = sort_idx(i);
    feat_name = features_sorted{i};
    
    subplot(2, 2, i);
    
    % Create PDP
    plotPartialDependence(best_rf_model, feat_idx, X_train);
    title(sprintf('PDP: %s', feat_name));
    xlabel(feat_name);
    ylabel('Partial Effect on Prediction');
    grid on;
end

sgtitle('Partial Dependence Plots (Top 4 Features)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'rf_partial_dependence_plots.png'));
fprintf('Partial dependence plots saved.\n');

%% ========================================
%% REQUIRED: PREDICTION INTERVALS
%% ========================================
fprintf('\n========== PREDICTION INTERVALS ==========\n');

% Use out-of-bag predictions for uncertainty estimation
% For each test sample, get predictions from all trees
all_predictions = zeros(size(X_test, 1), best_params.NumTrees);
for i = 1:best_params.NumTrees
    all_predictions(:, i) = predict(best_rf_model.Trained{i}, X_test);
end

% Calculate prediction intervals (e.g., 95%)
pred_mean = mean(all_predictions, 2);
pred_std = std(all_predictions, 0, 2);
pred_lower = prctile(all_predictions, 2.5, 2);  % 2.5th percentile
pred_upper = prctile(all_predictions, 97.5, 2); % 97.5th percentile

% Calculate coverage (what % of true values fall within intervals)
coverage = mean((y_test >= pred_lower) & (y_test <= pred_upper));
fprintf('95%% Prediction Interval Coverage: %.2f%%\n', coverage * 100);

% Plot prediction intervals for a subset of test samples
num_samples_plot = min(100, length(y_test));
[~, sort_idx_plot] = sort(y_test);
idx_plot = sort_idx_plot(1:num_samples_plot);

figure('Position', [100, 100, 1200, 600]);
x_axis = 1:num_samples_plot;

fill([x_axis, fliplr(x_axis)], ...
     [pred_lower(idx_plot)', fliplr(pred_upper(idx_plot)')], ...
     [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
plot(x_axis, y_test(idx_plot), 'bo', 'MarkerSize', 6, 'DisplayName', 'True Values');
plot(x_axis, pred_mean(idx_plot), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predictions');
xlabel('Sample Index (sorted by true value)');
ylabel('Target Value');
title(sprintf('95%% Prediction Intervals (Coverage: %.1f%%)', coverage * 100));
legend('95% PI', 'True Values', 'Predictions', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_folder, 'rf_prediction_intervals.png'));
fprintf('Prediction intervals plot saved.\n');

%% ========================================
%% SAVE MODEL AND ALL RESULTS
%% ========================================
fprintf('\n========== SAVING RESULTS ==========\n');

model_filename = fullfile(models_folder, ...
    sprintf('best_random_forest_%dtree_leaf%d.mat', best_params.NumTrees, best_params.MinLeafSize));

% Save comprehensive results
save(model_filename, 'best_rf_model', 'best_params', ...
    'train_rmse', 'train_mae', 'train_r2', ...
    'test_rmse', 'test_mae', 'test_r2', ...
    'importance', 'feature_names', ...
    'learning_curve_data', 'coverage', ...
    'residuals_train', 'residuals_test');

fprintf('Model and all metrics saved at: %s\n', model_filename);

%% ========================================
%% SUMMARY REPORT
%% ========================================
fprintf('\n========================================\n');
fprintf('         TRAINING COMPLETE\n');
fprintf('========================================\n');
fprintf('Best Configuration:\n');
fprintf('  - Number of Trees: %d\n', best_params.NumTrees);
fprintf('  - Min Leaf Size: %d\n', best_params.MinLeafSize);
fprintf('\nPerformance Metrics:\n');
fprintf('  Train: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', train_rmse, train_mae, train_r2);
fprintf('  Test:  RMSE=%.3f, MAE=%.3f, R²=%.3f\n', test_rmse, test_mae, test_r2);
fprintf('  CV:    RMSE=%.3f ± %.3f\n', mean(best_params.CV_RMSE), std(best_params.CV_RMSE));
fprintf('\nPrediction Intervals:\n');
fprintf('  95%% Coverage: %.2f%%\n', coverage * 100);
fprintf('\nAll plots and results saved to: %s\n', figures_folder);
fprintf('========================================\n');