clc;
clear;
close all;

%% Load Dataset
fprintf("Loading dataset...\n");
data = readtable('dataset.xlsx');
% data = data(11:end, :);
data.Date = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd');
data = sortrows(data, 'Date');
tail(data, 10); % Display the last 10 rows of the dataset
head(data, 5); % Display the first 5 rows of the dataset

%% Visualize the dataset
fprintf("\nVisualizing dataset...\n");
% Plot the entire dataset
figure;
plot(data.Date, data.Opening, 'b', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Opening Price');
title('ASELS Opening Prices Time Series');
grid on;

% Select the last 30 entries
last30 = data(end-29:end, :);

% Plot the last 30 days
figure;
plot(last30.Date, last30.Opening, 'r', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Opening Price');
title('ASELS Last 30 Days Opening Prices');
grid on;

%% Split Data into Train and Test Sets
fprintf("\nSplitting dataset into train and test sets...\n");
trainOpening = data.Opening(1:end-10); % Training set for opening prices
testOpening = data.Opening(end-9:end); % Test set for opening prices (last 10 days)

trainDate = data.Date(1:end-10); % Training set for dates
testDate = data.Date(end-9:end); % Test set for dates (last 10 days)


%% GBM Model
% model here
fprintf("\nTraining GBM Model using GBM...\n");

windowSize = 29;
X_train = [];
Y_train = [];

for i = (windowSize+1):length(trainOpening)
    X_train = [X_train; trainOpening(i-windowSize:i-1)'];
    Y_train = [Y_train; trainOpening(i)];
end

% Train a GBM model (Gradient Boosted Regression Trees)
gbmModel = fitensemble(X_train, Y_train, 'LSBoost', 135, 'Tree');

%% Prediction - Batch
%prediction vector
fprintf("\nPredicting future 10 values using GBM Model...\n");

% Tüm eğitim serisini bilinen olarak al
knownSeries = trainOpening;
predictedPrices = zeros(10,1);

for i = 1:10
    % Her iterasyonda son 5 değeri al
    lastFeatures = knownSeries(end-windowSize+1:end);
    % Tahmini yap
    predictedValue = predict(gbmModel, lastFeatures');
    predictedPrices(i) = predictedValue;
    % Tahmin edilen değeri seriye ekle ki bir sonraki adımda kullanabilelim
    knownSeries = [knownSeries; predictedValue];
end





%% Performance Evaluation
modelName = "GBM (data) 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = predictedPrices;
testDates = testDate;
lastNValues = 20;

% Calculate performance metrics
mae = mean(abs(testData - predictedData)); % Mean Absolute Error
mse = mean((testData - predictedData).^2); % Mean Squared Error
rmse = sqrt(mse); % Root Mean Squared Error
mape = mean(abs((testData - predictedData) ./ testData)) * 100; % Mean Absolute Percentage Error

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s Model Predictions vs Actual Values", modelName));
legend('show');
grid on;


%% Performance Evaluation
modelName = "GBM (data) 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = predictedPrices(1:5, :);
testDates = testDate(1:5, :);
lastNValues = 20;

% Calculate performance metrics
mae = mean(abs(testData - predictedData)); % Mean Absolute Error
mse = mean((testData - predictedData).^2); % Mean Squared Error
rmse = sqrt(mse); % Root Mean Squared Error
mape = mean(abs((testData - predictedData) ./ testData)) * 100; % Mean Absolute Percentage Error

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end-5, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s Model Predictions vs Actual Values", modelName));
legend('show');
grid on;




