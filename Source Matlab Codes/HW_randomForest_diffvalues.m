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

%% ADF Test
fprintf("\nADF test is being performed...\n");
% Check stationarity of the time series using the ADF test
[h, pValue] = adftest(trainOpening);

if h == 1
    fprintf('\nThe time series is stationary (p = %.4f).\n', pValue);
else
    fprintf('The time series is not stationary (p = %.4f). Differencing should be applied!\n', pValue);
end

% If the series is not stationary, apply differencing
if h == 0
    diffTrainOpening = diff(trainOpening); % First differencing
    diffTrainDate = trainDate(2:end); % Dates corresponding to the differenced data

    figure;
    plot(diffTrainDate, diffTrainOpening, 'b');
    title('First Differenced Time Series');
    xlabel('Date');
    ylabel('Differenced Opening Price');
    grid on;
end

%% ACF and PACF Analysis
fprintf("\nACF and PACF analysis is being performed...\n");

% ACF and PACF for the original data
performACFandPACF(trainOpening, 20);

% ACF and PACF for the differenced data
performACFandPACF(diffTrainOpening, 20);

%% Re-ADF Test for diffOpening
fprintf("\nADF test is being performed...\n");
% Check stationarity of the differenced time series using the ADF test
[h, pValue] = adftest(diffTrainOpening);

if h == 1
    fprintf('\nThe newly differenced time series is stationary (p = %.4f).\n', pValue);
else
    fprintf('The newly differenced time series is not stationary (p = %.4f). Further differencing may be required!\n', pValue);
end

%% Random Forest Model
fprintf("\nTraining Random Forest model...\n");

% Features (Independent variables): Prices from the past 1.5 years
% Target (Dependent variable): The price of the next day
% Here, a specific window size is chosen for each prediction (e.g., 30 days).

windowSize = 60; % Use the past 30 days to make predictions
x = [];
y = [];

% Organize the training data with the specified window size
for i = 1:(length(diffTrainOpening) - windowSize)
    x = [x; diffTrainOpening(i:i+windowSize-1)']; % Features in the window size
    y = [y; diffTrainOpening(i+windowSize)]; % Target value: the price of the next day
end

% Create the Random Forest model
numTrees = 50; % Number of trees in the Random Forest
rfModel = TreeBagger(numTrees, x, y, 'Method', 'regression', 'OOBPrediction', 'off');

fprintf("Random Forest model training complete.\n");


%% Prediction - Separate for Each Day
fprintf("\nPredicting future values iteratively using Random Forest...\n");

% Initialize an array to store predictions
predictedPrices = zeros(1, length(testOpening));

% Initial features: The last 'windowSize' values from the training set
currentFeatures = trainOpening(end-windowSize+1:end)';

% Predict for each day
for i = 1:length(testOpening)
    % Predict the next day's value
    nextPrediction = predict(rfModel, currentFeatures);
    
    % Save the prediction
    predictedPrices(i) = nextPrediction;
    
    % Add the predicted value to the current features
    % and remove the oldest feature (shifting operation)
    currentFeatures = [currentFeatures(2:end), nextPrediction];
end

% Convert forecasts back to the original scale
lastOriginalValue = trainOpening(end); % Last original value from the training set
predictedRealPrices = cumsum([lastOriginalValue; predictedPrices']); % Cumulative sum to restore original scale
predictedRealPrices = predictedRealPrices(2:end); % Remove the initial value (start point)

%% Performance Evaluation
modelName = "Random Forest (diffData) 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = predictedRealPrices;
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
fprintf("\nComparison Table for %s: \n", modelName);
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
modelName = "Random Forest (diffData) 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = predictedRealPrices(1:5, :);
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
fprintf("\nComparison Table for %s: \n", modelName);
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

%% Functions

function performACFandPACF(series, numLags)
% performACFandPACF - Performs ACF and PACF analysis for a time series
%
% Parameters:
%   series: Time series vector
%   numLags: Number of lags to analyze (default: 20)
%
% Example:
%   performACFandPACF(trainOriginal, 20);

% Check if numLags is provided
if nargin < 2
    numLags = 20; % Default value if numLags is not specified
end

% ACF and PACF plots
figure;
subplot(2,1,1);
autocorr(series, 'NumLags', numLags);
title("Autocorrelation (ACF)");

subplot(2,1,2);
parcorr(series, 'NumLags', numLags);
title("Partial Autocorrelation (PACF)");

fprintf("Analysis complete. Check the figures for results.\n");
end





