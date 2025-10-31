%% NN TRAINING WITH BAYESIAN OPTIMIZATION + DETAILED K-FOLD METRICS
% Uses BayesOpt to find best architecture, with full metrics per fold

%% Load and prepare data
data = load('cleaned_data.mat');
data = data.data;
X = data{:, 1:end-1};  % 4 parameters
y = data{:, end};       % Deformation
n_samples = size(X, 1);

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

%% DEFINE BAYESOPT SEARCH SPACE (CONSTRAINED)
% This prevents crazy architectures like [109, 68, 124]

optimVars = [
    % Number of hidden layers (2 or 3)
    optimizableVariable('NumLayers', [2, 3], 'Type', 'integer')
    
    % Layer sizes (MUCH more constrained than before!)
    optimizableVariable('Layer1Size', [8, 64], 'Type', 'integer')
    optimizableVariable('Layer2Size', [4, 32], 'Type', 'integer')
    optimizableVariable('Layer3Size', [4, 16], 'Type', 'integer')
    
    % Regularization
    optimizableVariable('L2Reg', [1e-5, 1e-2], 'Transform', 'log')
    
    % Activation function
    optimizableVariable('Activation', {'tansig', 'logsig', 'poslin'}, 'Type', 'categorical')
];

%% OBJECTIVE FUNCTION FOR BAYESOPT
% This function runs 5-fold CV and returns mean RMSE
% BayesOpt will call this function ~30 times with different hyperparameters

objectiveFcn = @(params) nnObjectiveWithDetailedMetrics(params, X_train, y_train);

%% RUN BAYESIAN OPTIMIZATION
fprintf('========================================\n');
fprintf('STARTING BAYESIAN OPTIMIZATION\n');
fprintf('========================================\n');
fprintf('- Max evaluations: 30\n');
fprintf('- Each evaluation runs 5-fold CV\n');
fprintf('- Constrained architecture search\n');
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
fprintf('Activation: %s\n', char(best_params.Activation));
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
    
    % Set activation
    for lyr = 1:length(net.layers)-1
        net.layers{lyr}.transferFcn = char(best_params.Activation);
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
    net_final.layers{lyr}.transferFcn = char(best_params.Activation);
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

fprintf('\nFINAL TRAIN SET:\n');
fprintf('  RMSE: %.3f\n', final_train_rmse);
fprintf('  MAE:  %.3f\n', final_train_mae);
fprintf('  R²:   %.3f\n', final_train_r2);

fprintf('\nFINAL TEST SET:\n');
fprintf('  RMSE: %.3f\n', test_rmse);
fprintf('  MAE:  %.3f\n', test_mae);
fprintf('  R²:   %.3f\n', test_r2);
fprintf('  Normalized RMSE: %.2f%% of std dev\n', (test_rmse / std(y_test)) * 100);

%% COMPARISON WITH YOUR ORIGINAL RESULT
fprintf('\n========================================\n');
fprintf('COMPARISON: Constrained vs Unconstrained BayesOpt\n');
fprintf('========================================\n');
fprintf('YOUR ORIGINAL (Unconstrained):\n');
fprintf('  Architecture: [109, 68, 124]\n');
fprintf('  Parameters: ~15,000\n');
fprintf('  CV RMSE: 12.205\n\n');

fprintf('NEW (Constrained Search Space):\n');
fprintf('  Architecture: %s\n', mat2str(layers));
fprintf('  Parameters: %d\n', n_params);
fprintf('  CV RMSE: %.3f\n', mean(val_rmse_all));
fprintf('  Test RMSE: %.3f\n', test_rmse);

if test_rmse < 12.205
    improvement = (12.205 - test_rmse) / 12.205 * 100;
    fprintf('\n✓ IMPROVED by %.1f%%!\n', improvement);
else
    fprintf('\n○ Similar performance but fewer parameters\n');
end

%% VISUALIZATION
figure('Position', [100, 100, 1400, 900]);

% Fold-by-fold metrics
subplot(2, 3, 1);
x_folds = 1:5;
plot(x_folds, train_rmse_all, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(x_folds, val_rmse_all, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold', 'FontSize', 11);
ylabel('RMSE', 'FontSize', 11);
title('K-Fold RMSE', 'FontSize', 11);
legend('Train', 'Val', 'Location', 'best');
grid on;

subplot(2, 3, 2);
plot(x_folds, train_mae_all, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(x_folds, val_mae_all, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold', 'FontSize', 11);
ylabel('MAE', 'FontSize', 11);
title('K-Fold MAE', 'FontSize', 11);
legend('Train', 'Val', 'Location', 'best');
grid on;

subplot(2, 3, 3);
plot(x_folds, train_r2_all, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(x_folds, val_r2_all, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold', 'FontSize', 11);
ylabel('R²', 'FontSize', 11);
title('K-Fold R²', 'FontSize', 11);
legend('Train', 'Val', 'Location', 'best');
grid on;

% Test set predictions
subplot(2, 3, 4);
scatter(y_test, y_pred_test, 30, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], ...
    'r--', 'LineWidth', 2);
xlabel('True Deformation', 'FontSize', 11);
ylabel('Predicted Deformation', 'FontSize', 11);
title(sprintf('Test Set (R²=%.3f)', test_r2), 'FontSize', 11);
grid on;

% Residuals
subplot(2, 3, 5);
residuals = y_test - y_pred_test;
scatter(y_pred_test, residuals, 30, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
yline(0, 'r--', 'LineWidth', 2);
xlabel('Predicted', 'FontSize', 11);
ylabel('Residuals', 'FontSize', 11);
title('Residual Plot', 'FontSize', 11);
grid on;

% BayesOpt progress
subplot(2, 3, 6);
plot(results_bayesopt.ObjectiveTrace, 'b-o', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 11);
ylabel('Objective (CV RMSE)', 'FontSize', 11);
title('Bayesian Optimization Progress', 'FontSize', 11);
grid on;

%% SAVE EVERYTHING
output_folder = '../fea-surrogate-model/Models';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

model_filename = fullfile(output_folder, 'best_nn_model.mat');
save(model_filename);

fprintf('\n========================================\n');
fprintf('Model saved: %s\n', model_filename);
fprintf('========================================\n');

%% OBJECTIVE FUNCTION (CALLED BY BAYESOPT)
function rmse = nnObjectiveWithDetailedMetrics(params, X, y)
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
    fprintf('  Activation: %s\n', char(params.Activation));
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
                net.layers{lyr}.transferFcn = char(params.Activation);
            end
            
            net.trainParam.epochs = 500;
            net.trainParam.max_fail = 15;
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.performParam.regularization = params.L2Reg;
            net.divideParam.trainRatio = 0.9;
            net.divideParam.valRatio = 0.1;
            net.divideParam.testRatio = 0.0;
            
            % Train (THIS IS THE TRAINING LOOP - hidden inside train())
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