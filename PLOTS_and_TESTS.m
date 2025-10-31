
clear; clc; close all;

%% Configuration
models_folder = '/home/adx/fea-surrogate-model/Models';
output_folder = '/home/adx/fea-surrogate-model/Figures';
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% Load original data for feature names and test set
data_file = load('cleaned_data.mat');
data = data_file.data;
feature_names = data.Properties.VariableNames(1:end-1);
target_name = data.Properties.VariableNames{end};

% Recreate the same train-test split (rng 42, 85-15 split)
X = data{:, 1:end-1};
y = data{:, end};
n_samples = size(X, 1);
rng(42);
cv_holdout = cvpartition(n_samples, 'HoldOut', 0.15);
X_train = X(training(cv_holdout), :);
y_train = y(training(cv_holdout));
X_test = X(test(cv_holdout), :);
y_test = y(test(cv_holdout));

fprintf('Test set recreated: %d samples\n', size(X_test, 1));

%% Load all model files
model_files = dir(fullfile(models_folder, '*.mat'));
fprintf('Found %d model files\n', length(model_files));

models_info = struct();
valid_models = 0;

for i = 1:length(model_files)
    filepath = fullfile(models_folder, model_files(i).name);
    loaded = load(filepath);
    
    % Determine model type and extract model object
    if isfield(loaded, 'best_gpr_model')
        model_type = 'GPR';
        model_obj = loaded.best_gpr_model;
    elseif isfield(loaded, 'best_svr_model')
        model_type = 'SVR';
        model_obj = loaded.best_svr_model;
    elseif isfield(loaded, 'best_rf_model')
        model_type = 'RF';
        model_obj = loaded.best_rf_model;
    elseif isfield(loaded, 'best_gbt_model')
        model_type = 'GBT';
        model_obj = loaded.best_gbt_model;
    else
        fprintf('Skipping %s - no recognizable model found\n', model_files(i).name);
        continue;
    end
    
    valid_models = valid_models + 1;
    models_info(valid_models).name = model_files(i).name;
    models_info(valid_models).type = model_type;
    models_info(valid_models).model = model_obj;
    models_info(valid_models).data = loaded;
    
    fprintf('Loaded: %s (Type: %s)\n', model_files(i).name, model_type);
end

models_info = models_info(1:valid_models);
fprintf('\nSuccessfully loaded %d models\n\n', valid_models);

%% 1. PARITY PLOTS (Predicted vs Actual) - Test Set
fprintf('Generating parity plots...\n');
n_models = length(models_info);
n_cols = min(3, n_models);
n_rows = ceil(n_models / n_cols);

figure('Position', [50, 50, 400*n_cols, 400*n_rows]);
for i = 1:n_models
    subplot(n_rows, n_cols, i);
    
    % Predict on test set
    if strcmp(models_info(i).type, 'GPR')
        [y_pred, ~, y_int] = predict(models_info(i).model, X_test);
        models_info(i).y_pred_test = y_pred;
        models_info(i).y_int_test = y_int;
    else
        y_pred = predict(models_info(i).model, X_test);
        models_info(i).y_pred_test = y_pred;
    end
    
    % Calculate metrics
    rmse = sqrt(mean((y_test - y_pred).^2));
    mae = mean(abs(y_test - y_pred));
    r2 = 1 - sum((y_test - y_pred).^2)/sum((y_test - mean(y_test)).^2);
    
    % Plot
    scatter(y_test, y_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', 'LineWidth', 2);
    
    % Annotations
    text(0.05, 0.95, sprintf('RMSE: %.3f\nMAE: %.3f\nR²: %.3f', rmse, mae, r2), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    xlabel('Actual', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Predicted', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('%s - %s', models_info(i).type, strrep(models_info(i).name, '_', '\_')), ...
        'FontSize', 11, 'Interpreter', 'tex');
    grid on;
    axis equal tight;
end
saveas(gcf, fullfile(output_folder, 'parity_plots_all_models.png'));
fprintf('Parity plots saved.\n');

%% 2. RESIDUAL PLOTS (Residuals vs Predicted + Histogram)
fprintf('Generating residual plots...\n');
for i = 1:n_models
    figure('Position', [100, 100, 1200, 400]);
    
    y_pred = models_info(i).y_pred_test;
    residuals = y_test - y_pred;
    
    % Residual scatter plot
    subplot(1, 2, 1);
    scatter(y_pred, residuals, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(0, 'r--', 'LineWidth', 2);
    xlabel('Predicted', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Residuals (Actual - Predicted)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s - Residual Plot', models_info(i).type), 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    
    % Residual histogram
    subplot(1, 2, 2);
    histogram(residuals, 30, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'black');
    xlabel('Residuals', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
    title('Residual Distribution', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    xline(0, 'r--', 'LineWidth', 2);
    
    saveas(gcf, fullfile(output_folder, sprintf('residuals_%s.png', models_info(i).type)));
end
fprintf('Residual plots saved.\n');

%% 3. FEATURE IMPORTANCE (RF and GBT only)
fprintf('Generating feature importance plots...\n');
for i = 1:n_models
    if strcmp(models_info(i).type, 'RF') || strcmp(models_info(i).type, 'GBT')
        figure('Position', [100, 100, 800, 600]);
        
        importance = models_info(i).model.OOBPermutedPredictorDeltaError;
        [sorted_imp, idx] = sort(importance, 'descend');
        sorted_names = feature_names(idx);
        
        barh(sorted_imp);
        set(gca, 'YTick', 1:length(sorted_names), 'YTickLabel', sorted_names, 'FontSize', 10);
        xlabel('Importance (OOB Permuted Predictor Delta Error)', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('%s - Feature Importance', models_info(i).type), 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        saveas(gcf, fullfile(output_folder, sprintf('feature_importance_%s.png', models_info(i).type)));
        
        % Store for later use
        models_info(i).feature_importance = importance;
        models_info(i).top_features_idx = idx(1:min(2, length(idx)));
    end
end
fprintf('Feature importance plots saved.\n');

%% 4. PARTIAL DEPENDENCE PLOTS (for each model, top 2 features)
fprintf('Generating partial dependence plots...\n');

% Determine top 2 features from tree models
top_features_idx = [];
for i = 1:n_models
    if strcmp(models_info(i).type, 'RF') || strcmp(models_info(i).type, 'GBT')
        importance = models_info(i).model.OOBPermutedPredictorDeltaError;
        [~, idx] = sort(importance, 'descend');
        top_features_idx = idx(1:2);
        break;
    end
end

if isempty(top_features_idx)
    % If no tree models, use first 2 features
    top_features_idx = [1, 2];
    fprintf('No tree models found. Using first 2 features for PDP.\n');
end

fprintf('Top 2 features: %s, %s\n', feature_names{top_features_idx(1)}, feature_names{top_features_idx(2)});

for i = 1:n_models
    fprintf('Computing PDP for %s...\n', models_info(i).type);
    
    figure('Position', [100, 100, 1400, 500]);
    
    for f = 1:2
        feat_idx = top_features_idx(f);
        feat_name = feature_names{feat_idx};
        
        % Create grid for feature
        feat_vals = linspace(min(X(:, feat_idx)), max(X(:, feat_idx)), 50);
        pdp_vals = zeros(size(feat_vals));
        
        % Compute partial dependence
        for j = 1:length(feat_vals)
            X_temp = X_test;
            X_temp(:, feat_idx) = feat_vals(j);
            
            if strcmp(models_info(i).type, 'GPR')
                y_temp = predict(models_info(i).model, X_temp);
            else
                y_temp = predict(models_info(i).model, X_temp);
            end
            pdp_vals(j) = mean(y_temp);
        end
        
        % Plot 1D PDP
        subplot(1, 2, f);
        plot(feat_vals, pdp_vals, 'LineWidth', 2.5, 'Color', [0.2, 0.4, 0.8]);
        xlabel(feat_name, 'FontSize', 12, 'FontWeight', 'bold');
        ylabel(sprintf('Partial Dependence on %s', target_name), 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('%s - PDP: %s', models_info(i).type, feat_name), 'FontSize', 13, 'FontWeight', 'bold');
        grid on;
    end
    
    saveas(gcf, fullfile(output_folder, sprintf('pdp_1d_%s.png', models_info(i).type)));
    
    % 2D PDP Surface
    figure('Position', [100, 100, 800, 600]);
    feat1_vals = linspace(min(X(:, top_features_idx(1))), max(X(:, top_features_idx(1))), 30);
    feat2_vals = linspace(min(X(:, top_features_idx(2))), max(X(:, top_features_idx(2))), 30);
    [F1, F2] = meshgrid(feat1_vals, feat2_vals);
    Z = zeros(size(F1));
    
    for r = 1:size(F1, 1)
        for c = 1:size(F1, 2)
            X_temp = X_test;
            X_temp(:, top_features_idx(1)) = F1(r, c);
            X_temp(:, top_features_idx(2)) = F2(r, c);
            
            if strcmp(models_info(i).type, 'GPR')
                y_temp = predict(models_info(i).model, X_temp);
            else
                y_temp = predict(models_info(i).model, X_temp);
            end
            Z(r, c) = mean(y_temp);
        end
    end
    
    surf(F1, F2, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    xlabel(feature_names{top_features_idx(1)}, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(feature_names{top_features_idx(2)}, 'FontSize', 12, 'FontWeight', 'bold');
    zlabel(target_name, 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s - 2D PDP Surface', models_info(i).type), 'FontSize', 14, 'FontWeight', 'bold');
    colorbar;
    view(45, 30);
    
    saveas(gcf, fullfile(output_folder, sprintf('pdp_2d_%s.png', models_info(i).type)));
end
fprintf('Partial dependence plots saved.\n');

%% 5. GPR UNCERTAINTY PLOT (if GPR model exists)
fprintf('Generating GPR uncertainty plots...\n');
for i = 1:n_models
    if strcmp(models_info(i).type, 'GPR')
        figure('Position', [100, 100, 1000, 600]);
        
        [y_pred, y_sd, y_int] = predict(models_info(i).model, X_test);
        
        % Sort by predicted values for better visualization
        [y_pred_sorted, sort_idx] = sort(y_pred);
        y_test_sorted = y_test(sort_idx);
        y_sd_sorted = y_sd(sort_idx);
        
        x_axis = 1:length(y_pred_sorted);
        
        % Plot predictions with uncertainty
        fill([x_axis, fliplr(x_axis)], ...
             [y_pred_sorted' + 2*y_sd_sorted', fliplr(y_pred_sorted' - 2*y_sd_sorted')], ...
             [0.8, 0.9, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        hold on;
        plot(x_axis, y_pred_sorted, 'b-', 'LineWidth', 2, 'DisplayName', 'Predicted Mean');
        scatter(x_axis, y_test_sorted, 30, 'r', 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual');
        
        xlabel('Test Sample (sorted by prediction)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel(target_name, 'FontSize', 12, 'FontWeight', 'bold');
        title('GPR - Predicted Mean ± 2SD (95% Confidence)', 'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 11);
        grid on;
        
        saveas(gcf, fullfile(output_folder, 'gpr_uncertainty.png'));
        fprintf('GPR uncertainty plot saved.\n');
    end
end

%% 6. 2D HEATMAP SLICES (All combinations for SVR - adaptable for other models)
fprintf('Generating 2D heatmap slices...\n');

% Select model for heatmap (default: SVR, but can be changed)
target_model_idx = find(strcmp({models_info.type}, 'SVR'), 1);
if isempty(target_model_idx)
    target_model_idx = 1; % Use first model if no SVR
    fprintf('No SVR found. Using %s for heatmaps.\n', models_info(target_model_idx).type);
end

selected_model = models_info(target_model_idx).model;
selected_type = models_info(target_model_idx).type;

% Compute median values for fixing features
X_median = median(X_train, 1);

% Generate all combinations of 2 features
n_features = length(feature_names);
feature_pairs = nchoosek(1:n_features, 2);
n_pairs = size(feature_pairs, 1);

fprintf('Generating %d heatmap combinations...\n', n_pairs);

grid_resolution = 50;

for p = 1:n_pairs
    feat1_idx = feature_pairs(p, 1);
    feat2_idx = feature_pairs(p, 2);
    feat1_name = feature_names{feat1_idx};
    feat2_name = feature_names{feat2_idx};
    
    fprintf('  Heatmap %d/%d: %s vs %s\n', p, n_pairs, feat1_name, feat2_name);
    
    % Create grid
    feat1_vals = linspace(min(X(:, feat1_idx)), max(X(:, feat1_idx)), grid_resolution);
    feat2_vals = linspace(min(X(:, feat2_idx)), max(X(:, feat2_idx)), grid_resolution);
    [F1, F2] = meshgrid(feat1_vals, feat2_vals);
    Z = zeros(size(F1));
    
    % Predict on grid
    for r = 1:size(F1, 1)
        for c = 1:size(F1, 2)
            X_temp = repmat(X_median, 1, 1);
            X_temp(feat1_idx) = F1(r, c);
            X_temp(feat2_idx) = F2(r, c);
            
            if strcmp(selected_type, 'GPR')
                Z(r, c) = predict(selected_model, X_temp);
            else
                Z(r, c) = predict(selected_model, X_temp);
            end
        end
    end
    
    % Plot heatmap
    figure('Position', [100, 100, 800, 600]);
    imagesc(feat1_vals, feat2_vals, Z);
    set(gca, 'YDir', 'normal');
    colorbar;
    xlabel(feat1_name, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(feat2_name, 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s - Predicted %s\n(%s vs %s, others at median)', ...
        selected_type, target_name, feat1_name, feat2_name), ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    % Save with descriptive filename
    safe_feat1 = strrep(feat1_name, ' ', '_');
    safe_feat2 = strrep(feat2_name, ' ', '_');
    saveas(gcf, fullfile(output_folder, sprintf('heatmap_%s_%s_vs_%s.png', ...
        selected_type, safe_feat1, safe_feat2)));
    close(gcf);
end

fprintf('All heatmap slices saved.\n');

%% Summary
fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Total models processed: %d\n', n_models);
fprintf('Figures saved to: %s\n', output_folder);
fprintf('========================================\n');