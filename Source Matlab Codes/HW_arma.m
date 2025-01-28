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
last20 = data(end-19:end, :);

% Plot the last 30 days
figure;
plot(last20.Date, last20.Opening, 'r', 'LineWidth', 1.5);
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


%% Optimal ARMA Model - Grid Search with AIC Calculation
fprintf("\nFinding optimal ARMA(p, q) model with detailed output...\n");

maxP = 5; % Maximum degrees for AR
maxQ = 5; % Maximum degrees for MA
bestAIC = Inf; % Initial large value for comparison
optimalP = 0;
optimalQ = 0;

fprintf("%5s %5s %10s\n", "p", "q", "AIC"); % Column headers

% Grid Search
for p = 0:maxP
    for q = 0:maxQ
        try
            model = arima(p, 0, q); % ARMA(p, q) model
            [~, ~, logL] = estimate(model, diffTrainOpening, 'Display', 'off'); % Train the model and get log-likelihood
            numParams = p + q + 1; % Count of AR, MA parameters, and constant
            aic = -2 * logL + 2 * numParams; % AIC calculation
            
            % Print AIC results
            fprintf("%5d %5d %10.4f\n", p, q, aic);
            
            % Update the best model
            if aic < bestAIC
                bestAIC = aic;
                optimalP = p;
                optimalQ = q;
            end
        catch ME
            % Indicate failed combinations
            fprintf("%5d %5d %10s - Error: %s\n", p, q, "FAILED", ME.message);
        end
    end
end

fprintf("\nOptimal ARMA model: p = %d, q = %d, AIC = %.4f\n", optimalP, optimalQ, bestAIC);

%% ARMA Model - Forecasting and Performance Evaluation
fprintf("\nFitting ARMA model with differenced data...\n");

% Fit the ARMA model with the optimal parameters
p = optimalP; % AR parameter
q = optimalQ; % MA parameter

model = arima(p, 0, q); % ARMA(p, q) model
fittedModel = estimate(model, diffTrainOpening);

% Forecast using the ARMA model (based on differenced data)
numSteps = 10;
forecastDiff = forecast(fittedModel, numSteps, 'Y0', diffTrainOpening);

% Convert forecasts back to the original scale
lastOriginalValue = trainOpening(end); % Last original value from the training set
forecastValues = cumsum([lastOriginalValue; forecastDiff]); % Cumulative sum to restore original scale
forecastValues = forecastValues(2:end); % Remove the initial value (start point)

%% Performance Evaluation
modelName = "ARMA 10 Days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = forecastValues;
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
modelName = "ARMA 5 Days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = forecastValues(1:5, :);
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






