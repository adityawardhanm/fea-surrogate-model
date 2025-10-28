%% EDA.m - Exploratory Data Analysis Script for Dataset

clear; clc; close all;

data = readtable('Soft_Actuator_FEA_Dataset.csv'); % Load dataset

fprintf('Dataset Sample Preview:\n');
head(data);

% Extract features and target variable from the dataset
x = data{:, 1:end-1}; 
y = data{:, end}; 

feature_names = {'Pressure', 'Height', 'Length', 'Thickness'};

fprintf('Dataset loaded successfully!\n\n');

% Dataset statistics
n_samples = size(x, 1);
n_features = size(x, 2);
fprintf('   Total samples: %d\n', n_samples);
fprintf('   Features: %d\n', n_features);

% Display parameter ranges
fprintf('   Parameter Ranges:\n');
for i = 1:n_features
    fprintf('       %s: [%.2f, %.2f]\n', feature_names{i}, min(x(:,i)), max(x(:,i)));
end
fprintf('   Deformation range: [%.2f, %.2f] mm\n', min(y), max(y));
fprintf('   Deformation mean: %.2f mm (std: %.2f mm)\n\n', mean(y), std(y));
fprintf('\n');

% Figure 1: Histogram of Deformation (REQUIRED PLOT #1)
% figure('Position', [100, 100, 800, 600]);
% histogram(y, 30, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'k', 'LineWidth', 1);
% xlabel('Deformation (mm)', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
% title('Distribution of Actuator Deformation', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% set(gca, 'FontSize', 11);

% % Add statistics text box
% text_str = sprintf('Mean: %.2f mm\nStd: %.2f mm\nMin: %.2f mm\nMax: %.2f mm', ...
%     mean(y), std(y), min(y), max(y));
% annotation('textbox', [0.65, 0.7, 0.2, 0.15], 'String', text_str, ...
%     'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', 'k');

% % Figure 2: Input feature distributions
% figure('Position', [150, 150, 1200, 800]);
% for i = 1:4
%     subplot(2, 2, i);
%     histogram(x(:,i), 20, 'FaceColor', [0.3 0.7 0.4], 'EdgeColor', 'k', 'LineWidth', 1);
%     xlabel(feature_names{i}, 'FontSize', 11, 'FontWeight', 'bold');
%     ylabel('Frequency', 'FontSize', 11, 'FontWeight', 'bold');
%     title(['Distribution of ' feature_names{i}], 'FontSize', 12, 'FontWeight', 'bold');
%     grid on;
%     set(gca, 'FontSize', 10);
% end
% sgtitle('Input Parameter Distributions', 'FontSize', 14, 'FontWeight', 'bold');


anyMissing = any(ismissing(data), 'all');

if anyMissing
    fprintf('Warning: Missing values detected in the dataset.\n');
else
    fprintf('No missing values detected in the dataset.\n');
end

save('cleaned_data.mat', 'data');
fprintf('Cleaned dataset saved to cleaned_data.mat\n');