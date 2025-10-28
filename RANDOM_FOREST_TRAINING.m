%% RANDOM_FOREST_TRAINING.m

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

%% 5-fold CV setup
cv_kfold = cvpartition(size(X_train,1), 'KFold', 5);

%% Hyperparameter grid
numTreesList = [50, 100, 200];
leafSizeList = [1, 5, 10];

best_rmse = inf;
best_params = struct();

%% Grid search with CV metrics
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

fprintf('\nBest RF config: %d trees, MinLeafSize=%d\n', ...
    best_params.NumTrees, best_params.MinLeafSize);
fprintf('Mean CV RMSE: %.3f\n', best_rmse);

%% Retrain best model on full training set
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

%% Save model and metrics in folder
output_folder = '../fea-surrogate-model/Models';
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

model_filename = fullfile(output_folder, ...
    sprintf('best_random_forest_%dtree_leaf%d.mat', best_params.NumTrees, best_params.MinLeafSize));

save(model_filename, 'best_rf_model', 'best_params', ...
    'train_rmse', 'train_mae', 'train_r2', ...
    'test_rmse', 'test_mae', 'test_r2');

fprintf('\nModel and metrics saved at: %s\n', model_filename);
