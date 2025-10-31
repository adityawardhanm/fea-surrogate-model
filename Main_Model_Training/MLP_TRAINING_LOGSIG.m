%% NN TRAINING WITH BAYESIAN OPTIMIZATION + COMPLETE ANALYSIS
% Uses BayesOpt to find best architecture, with full metrics and visualizations

%% Load and prepare data
data = load('cleaned_data.mat');
data = data.data;
X = data{:, 1:end-1};  % 4 parameters
y = data{:, end};       % Deformation
n_samples = size(X, 1);
feature_names = data.Properties.VariableNames(1:end-1);

fprintf('Dataset: %d samples, %d features → 1 target\n', n_samples, size(X,2));
fprintf('Target statistics: Min=%.3f, Max=%.3f, Mean=%.3f, Std=%.3f\n', ...
    min(y), max(y), mean(y), std(y));

rng(42);

% 85-15 train-test split
cv_holdout = cvpartition(n_samples, 'HoldOut', 0.15);
X_train = X(training(cv_holdout), :);
y_train = y(training(cv_holdout));
X_test  = X(test(cv_holdout), :);
y_test  = y(test(cv_holdout));

fprintf('Training set: %d samples, Test set: %d samples\n\n', size(X_train,1), size(X_test,1));

%% Create output folders
models_folder = '../fea-surrogate-model/Models';
figures_folder = '../fea-surrogate-model/Figures';
if ~exist(models_folder,'dir')
    mkdir(models_folder);
end
if ~exist(figures_folder,'dir')
    mkdir(figures_folder);
end

%% DEFINE BAYESOPT SEARCH SPACE (CONSTRAINED)
optimVars = [
    % Number of hidden layers (2 or 3)
    optimizableVariable('NumLayers', [2, 3], 'Type', 'integer')
    
    % Layer sizes (MUCH more constrained than before!)
    optimizableVariable('Layer1Size', [8, 64], 'Type', 'integer')
    optimizableVariable('Layer2Size', [4, 32], 'Type', 'integer')
    optimizableVariable('Layer3Size', [4, 16], 'Type', 'integer')
    
    % Regularization
    optimizableVariable('L2Reg', [1e-5, 1e-2], 'Transform', 'log')
];

% Fixed activation function (based on prior testing)
fixed_activation = 'logsig';

%% OBJECTIVE FUNCTION FOR BAYESOPT
% Pass fixed_activation to objective function
objectiveFcn = @(params) nnObjectiveWithDetailedMetrics(params, X_train, y_train, fixed_activation);

%% RUN BAYESIAN OPTIMIZATION
fprintf('========================================\n');
fprintf('STARTING BAYESIAN OPTIMIZATION\n');
fprintf('========================================\n');
fprintf('- Max evaluations: 30\n');
fprintf('- Each evaluation runs 5-fold CV\n');
fprintf('- Constrained architecture search\n');
fprintf('- Activation function: %s (fixed)\n', fixed_activation);
fprintf('- Random seed: 42\n\n');

rng(42, 'twister');

results_bayesopt = bayesopt(objectiveFcn, optimVars, ...
    'Verbose', 1, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', false);

%% EXTRACT BEST HYPERPARAMETERS
best_params = results_bayesopt.XAtMinObjective;

fprintf('\n========================================\n');
fprintf('BEST HYPERPARAMETERS FOUND\n');
fprintf('========================================\n');
fprintf('Number of Layers: %d\n', best_params.NumLayers);
fprintf('Layer 1 Size: %d\n', best_params.Layer1Size);
fprintf('Layer 2 Size: %d\n', best_params.Layer2Size);
if best_params.NumLayers == 3
    fprintf('Layer 3 Size: %d\n', best_params.Layer3Size);
end
fprintf('L2 Regularization: %.4e\n', best_params.L2Reg);
fprintf('Activation: %s (fixed)\n', fixed_activation);
fprintf('Best CV RMSE: %.4f\n', results_bayesopt.MinObjective);

% Build layer array
layers = [best_params.Layer1Size, best_params.Layer2Size];
if best_params.NumLayers == 3
    layers = [layers, best_params.Layer3Size];
end
fprintf('Architecture: %s\n', mat2str(layers));

% Calculate parameter count
n_params = 0;
prev_size = size(X, 2);
for j = 1:length(layers)
    n_params = n_params + (prev_size + 1) * layers(j);
    prev_size = layers(j);
end
n_params = n_params + (prev_size + 1) * 1;
fprintf('Total parameters: %d (%.1f samples per parameter)\n', ...
    n_params, n_samples / n_params);

%% DETAILED CV EVALUATION WITH BEST CONFIG
fprintf('\n========================================\n');
fprintf('DETAILED K-FOLD CV FOR BEST CONFIG\n');
fprintf('========================================\n');

cv = cvpartition(size(X_train, 1), 'KFold', 5);
cv_metrics = struct();

for fold = 1:cv.NumTestSets
    fprintf('\n--- FOLD %d/%d ---\n', fold, cv.NumTestSets);
    
    % Split data
    X_cv_train = X_train(training(cv, fold), :);
    y_cv_train = y_train(training(cv, fold));
    X_cv_val = X_train(test(cv, fold), :);
    y_cv_val = y_train(test(cv, fold));
    
    fprintf('Train samples: %d, Val samples: %d\n', ...
        length(y_cv_train), length(y_cv_val));
    
    % Create network
    net = feedforwardnet(layers, 'trainlm');
    
    % Set activation to logsig (fixed)
    for lyr = 1:length(net.layers)-1
        net.layers{lyr}.transferFcn = fixed_activation;
    end
    
    % Training parameters
    net.trainParam.epochs = 500;
    net.trainParam.max_fail = 15;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.performParam.regularization = best_params.L2Reg;
    net.divideParam.trainRatio = 0.9;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0.0;
    
    % Train
    fprintf('Training...\n');
    tic;
    net = train(net, X_cv_train', y_cv_train');
    train_time = toc;
    fprintf('Training completed in %.2f seconds\n', train_time);
    
    % Predict on TRAINING partition of this fold
    y_pred_train = net(X_cv_train')';
    train_rmse = sqrt(mean((y_cv_train - y_pred_train).^2));
    train_mae = mean(abs(y_cv_train - y_pred_train));
    train_r2 = 1 - sum((y_cv_train - y_pred_train).^2) / ...
                   sum((y_cv_train - mean(y_cv_train)).^2);
    
    % Predict on VALIDATION partition of this fold
    y_pred_val = net(X_cv_val')';
    val_rmse = sqrt(mean((y_cv_val - y_pred_val).^2));
    val_mae = mean(abs(y_cv_val - y_pred_val));
    val_r2 = 1 - sum((y_cv_val - y_pred_val).^2) / ...
                 sum((y_cv_val - mean(y_cv_val)).^2);
    
    % Store metrics
    cv_metrics(fold).train_rmse = train_rmse;
    cv_metrics(fold).train_mae = train_mae;
    cv_metrics(fold).train_r2 = train_r2;
    cv_metrics(fold).val_rmse = val_rmse;
    cv_metrics(fold).val_mae = val_mae;
    cv_metrics(fold).val_r2 = val_r2;
    cv_metrics(fold).train_time = train_time;
    
    fprintf('\nFold %d Results:\n', fold);
    fprintf('  TRAIN: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', ...
        train_rmse, train_mae, train_r2);
    fprintf('  VAL:   RMSE=%.3f, MAE=%.3f, R²=%.3f\n', ...
        val_rmse, val_mae, val_r2);
end

% Aggregate CV metrics
train_rmse_all = [cv_metrics.train_rmse];
train_mae_all = [cv_metrics.train_mae];
train_r2_all = [cv_metrics.train_r2];
val_rmse_all = [cv_metrics.val_rmse];
val_mae_all = [cv_metrics.val_mae];
val_r2_all = [cv_metrics.val_r2];

fprintf('\n========================================\n');
fprintf('CV SUMMARY (mean ± std)\n');
fprintf('========================================\n');
fprintf('TRAIN: RMSE=%.3f±%.3f, MAE=%.3f±%.3f, R²=%.3f±%.3f\n', ...
    mean(train_rmse_all), std(train_rmse_all), ...
    mean(train_mae_all), std(train_mae_all), ...
    mean(train_r2_all), std(train_r2_all));
fprintf('VAL:   RMSE=%.3f±%.3f, MAE=%.3f±%.3f, R²=%.3f±%.3f\n', ...
    mean(val_rmse_all), std(val_rmse_all), ...
    mean(val_mae_all), std(val_mae_all), ...
    mean(val_r2_all), std(val_r2_all));

%% TRAIN FINAL MODEL ON FULL TRAINING SET
fprintf('\n========================================\n');
fprintf('TRAINING FINAL MODEL ON FULL TRAINING SET\n');
fprintf('========================================\n');

net_final = feedforwardnet(layers, 'trainlm');

for lyr = 1:length(net_final.layers)-1
    net_final.layers{lyr}.transferFcn = fixed_activation;
end

net_final.trainParam.epochs = 1000;
net_final.trainParam.max_fail = 20;
net_final.trainParam.showWindow = false;
net_final.trainParam.showCommandLine = false;
net_final.performParam.regularization = best_params.L2Reg;
net_final.divideParam.trainRatio = 0.85;
net_final.divideParam.valRatio = 0.15;
net_final.divideParam.testRatio = 0.0;

fprintf('Training on %d samples...\n', length(y_train));
tic;
net_final = train(net_final, X_train', y_train');
final_train_time = toc;
fprintf('Training completed in %.2f seconds\n', final_train_time);

%% EVALUATE FINAL MODEL
fprintf('\n========================================\n');
fprintf('FINAL MODEL EVALUATION\n');
fprintf('========================================\n');

% Training set
y_pred_train_final = net_final(X_train')';
final_train_rmse = sqrt(mean((y_train - y_pred_train_final).^2));
final_train_mae = mean(abs(y_train - y_pred_train_final));
final_train_r2 = 1 - sum((y_train - y_pred_train_final).^2) / ...
                     sum((y_train - mean(y_train)).^2);

% Test set
y_pred_test = net_final(X_test')';
test_rmse = sqrt(mean((y_test - y_pred_test).^2));
test_mae = mean(abs(y_test - y_pred_test));
test_r2 = 1 - sum((y_test - y_pred_test).^2) / ...
              sum((y_test - mean(y_test)).^2);

fprintf('\n===== TRAIN METRICS =====\n');
fprintf('RMSE: %.3f\n', final_train_rmse);
fprintf('MAE:  %.3f\n', final_train_mae);
fprintf('R²:   %.3f\n', final_train_r2);

fprintf('\n===== TEST METRICS =====\n');
fprintf('RMSE: %.3f\n', test_rmse);
fprintf('MAE:  %.3f\n', test_mae);
fprintf('R²:   %.3f\n', test_r2);
fprintf('Normalized RMSE: %.2f%% of std dev\n', (test_rmse / std(y_test)) * 100);

%% ========================================
%% REQUIRED: RESIDUAL ANALYSIS
%% ========================================
fprintf('\n========== RESIDUAL ANALYSIS ==========\n');
residuals_train = y_train - y_pred_train_final;
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
scatter(y_pred_train_final, residuals_train, 50, 'filled', 'MarkerFaceAlpha', 0.4);
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
saveas(gcf, fullfile(figures_folder, 'mlp_residual_analysis.png'));
fprintf('Residual analysis plots saved.\n');

%% ========================================
%% REQUIRED: FEATURE IMPORTANCE (Permutation)
%% ========================================
fprintf('\n========== FEATURE IMPORTANCE ==========\n');

% Compute permutation importance on test set
baseline_rmse = test_rmse;
feature_importance = zeros(length(feature_names), 1);

for i = 1:length(feature_names)
    X_test_permuted = X_test;
    X_test_permuted(:, i) = X_test_permuted(randperm(size(X_test, 1)), i);
    
    y_pred_permuted = net_final(X_test_permuted')';
    permuted_rmse = sqrt(mean((y_test - y_pred_permuted).^2));
    
    feature_importance(i) = permuted_rmse - baseline_rmse;
end

% Normalize
feature_importance = max(feature_importance, 0); % Ensure non-negative
feature_importance = feature_importance / sum(feature_importance);

% Sort features by importance
[importance_sorted, sort_idx] = sort(feature_importance, 'descend');
features_sorted = feature_names(sort_idx);

figure('Position', [100, 100, 1000, 600]);
bar(importance_sorted, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:length(features_sorted), 'XTickLabel', features_sorted);
xtickangle(45);
ylabel('Importance Score');
title('Feature Importance (Permutation Method)');
grid on;
saveas(gcf, fullfile(figures_folder, 'mlp_feature_importance.png'));

fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(features_sorted))
    fprintf('  %d. %s: %.4f\n', i, features_sorted{i}, importance_sorted(i));
end

%% ========================================
%% REQUIRED: LEARNING CURVES
%% ========================================
fprintf('\n========== LEARNING CURVES ==========\n');

figure('Position', [100, 100, 1400, 500]);

% BayesOpt convergence
subplot(1,3,1);
plot(results_bayesopt.ObjectiveTrace, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Bayesian Optimization Iteration');
ylabel('CV RMSE');
title('Hyperparameter Optimization Progress');
grid on;

% CV metrics across folds
subplot(1,3,2);
x_folds = 1:5;
plot(x_folds, train_rmse_all, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(x_folds, val_rmse_all, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold Number');
ylabel('RMSE');
title('K-Fold Cross-Validation RMSE');
legend('Train', 'Validation', 'Location', 'best');
grid on;

% R² across folds
subplot(1,3,3);
plot(x_folds, train_r2_all, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(x_folds, val_r2_all, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold Number');
ylabel('R²');
title('K-Fold Cross-Validation R²');
legend('Train', 'Validation', 'Location', 'best');
grid on;

sgtitle('Learning Curves', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'mlp_learning_curves.png'));
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
    
    % Create PDP manually for neural network
    x_range = linspace(min(X_train(:, feat_idx)), max(X_train(:, feat_idx)), 50);
    pdp_values = zeros(size(x_range));
    
    for j = 1:length(x_range)
        X_pdp = X_train;
        X_pdp(:, feat_idx) = x_range(j);
        y_pdp = net_final(X_pdp')';
        pdp_values(j) = mean(y_pdp);
    end
    
    plot(x_range, pdp_values, 'b-', 'LineWidth', 2);
    xlabel(feat_name);
    ylabel('Effect on Deformation (mm)');
    title(sprintf('PDP: %s', feat_name));
    grid on;
end

sgtitle('Partial Dependence Plots (Top 4 Features)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'mlp_partial_dependence_plots.png'));
fprintf('Partial dependence plots saved.\n');

%% ========================================
%% ADDITIONAL VISUALIZATIONS
%% ========================================
fprintf('\n========== ADDITIONAL VISUALIZATIONS ==========\n');

figure('Position', [100, 100, 1400, 500]);

% Predicted vs True (Test Set)
subplot(1,3,1);
scatter(y_test, y_pred_test, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', 'LineWidth', 2);
xlabel('True Deformation (mm)');
ylabel('Predicted Deformation (mm)');
title(sprintf('Test Set Predictions (R²=%.3f)', test_r2));
grid on;
axis equal;

% Predicted vs True (Train Set)
subplot(1,3,2);
scatter(y_train, y_pred_train_final, 50, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', 'LineWidth', 2);
xlabel('True Deformation (mm)');
ylabel('Predicted Deformation (mm)');
title(sprintf('Train Set Predictions (R²=%.3f)', final_train_r2));
grid on;
axis equal;

% Network Architecture Visualization
subplot(1,3,3);
view(net_final);
title('Neural Network Architecture');

sgtitle('Model Predictions and Architecture', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_folder, 'mlp_predictions_and_architecture.png'));
fprintf('Predictions and architecture plots saved.\n');

%% ========================================
%% SAVE MODEL AND ALL RESULTS
%% ========================================
fprintf('\n========== SAVING RESULTS ==========\n');

model_filename = fullfile(models_folder, 'best_mlp_model.mat');
save(model_filename, 'net_final', 'best_params', 'results_bayesopt', 'layers', 'fixed_activation', ...
    'final_train_rmse', 'final_train_mae', 'final_train_r2', ...
    'test_rmse', 'test_mae', 'test_r2', ...
    'cv_metrics', 'train_rmse_all', 'val_rmse_all', ...
    'feature_importance', 'feature_names', ...
    'residuals_train', 'residuals_test');

fprintf('Model and all metrics saved at: %s\n', model_filename);

%% ========================================
%% SUMMARY REPORT
%% ========================================
fprintf('\n========================================\n');
fprintf('         TRAINING COMPLETE\n');
fprintf('========================================\n');
fprintf('Best Configuration:\n');
fprintf('  - Architecture: %s\n', mat2str(layers));
fprintf('  - Parameters: %d\n', n_params);
fprintf('  - Activation: %s (fixed)\n', fixed_activation);
fprintf('  - L2 Regularization: %.4e\n', best_params.L2Reg);
fprintf('\nPerformance Metrics:\n');
fprintf('  Train: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', final_train_rmse, final_train_mae, final_train_r2);
fprintf('  Test:  RMSE=%.3f, MAE=%.3f, R²=%.3f\n', test_rmse, test_mae, test_r2);
fprintf('  CV:    RMSE=%.3f ± %.3f\n', mean(val_rmse_all), std(val_rmse_all));
fprintf('\nTraining Time: %.2f seconds\n', final_train_time);
fprintf('\nModel saved to: %s\n', models_folder);
fprintf('Figures saved to: %s\n', figures_folder);
fprintf('========================================\n');

%% OBJECTIVE FUNCTION (CALLED BY BAYESOPT)
function rmse = nnObjectiveWithDetailedMetrics(params, X, y, fixed_activation)
    % Build layer array
    layers = [params.Layer1Size, params.Layer2Size];
    if params.NumLayers == 3
        layers = [layers, params.Layer3Size];
    end
    
    % Calculate parameters
    n_params = 0;
    prev_size = size(X, 2);
    for j = 1:length(layers)
        n_params = n_params + (prev_size + 1) * layers(j);
        prev_size = layers(j);
    end
    n_params = n_params + (prev_size + 1) * 1;
    
    fprintf('\n========================================\n');
    fprintf('Testing Configuration:\n');
    fprintf('  Architecture: %s\n', mat2str(layers));
    fprintf('  Parameters: %d\n', n_params);
    fprintf('  L2 Reg: %.4e\n', params.L2Reg);
    fprintf('  Activation: %s (fixed)\n', fixed_activation);
    fprintf('----------------------------------------\n');
    
    % 5-fold CV
    cv = cvpartition(size(X, 1), 'KFold', 5);
    cv_rmse = zeros(cv.NumTestSets, 1);
    cv_mae = zeros(cv.NumTestSets, 1);
    cv_r2 = zeros(cv.NumTestSets, 1);
    
    for fold = 1:cv.NumTestSets
        X_cv_train = X(training(cv, fold), :);
        y_cv_train = y(training(cv, fold));
        X_cv_val = X(test(cv, fold), :);
        y_cv_val = y(test(cv, fold));
        
        try
            % Create and configure network
            net = feedforwardnet(layers, 'trainlm');
            
            for lyr = 1:length(net.layers)-1
                net.layers{lyr}.transferFcn = fixed_activation;
            end
            
            net.trainParam.epochs = 500;
            net.trainParam.max_fail = 15;
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.performParam.regularization = params.L2Reg;
            net.divideParam.trainRatio = 0.9;
            net.divideParam.valRatio = 0.1;
            net.divideParam.testRatio = 0.0;
            
            % Train
            net = train(net, X_cv_train', y_cv_train');
            
            % Predict
            y_pred = net(X_cv_val')';
            
            % Compute metrics
            cv_rmse(fold) = sqrt(mean((y_cv_val - y_pred).^2));
            cv_mae(fold) = mean(abs(y_cv_val - y_pred));
            cv_r2(fold) = 1 - sum((y_cv_val - y_pred).^2) / ...
                              sum((y_cv_val - mean(y_cv_val)).^2);
            
            fprintf('  Fold %d: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', ...
                fold, cv_rmse(fold), cv_mae(fold), cv_r2(fold));
            
        catch ME
            fprintf('  Fold %d: FAILED - %s\n', fold, ME.message);
            cv_rmse(fold) = 1e6;
            cv_mae(fold) = 1e6;
            cv_r2(fold) = -1e6;
        end
    end
    
    % Return mean RMSE
    valid = cv_rmse < 1e6;
    if sum(valid) == 0
        rmse = 1e6;
    else
        rmse = mean(cv_rmse(valid));
        fprintf('----------------------------------------\n');
        fprintf('Summary: RMSE=%.3f±%.3f, MAE=%.3f±%.3f, R²=%.3f±%.3f\n', ...
            mean(cv_rmse(valid)), std(cv_rmse(valid)), ...
            mean(cv_mae(valid)), std(cv_mae(valid)), ...
            mean(cv_r2(valid)), std(cv_r2(valid)));
    end
    fprintf('========================================\n');
end