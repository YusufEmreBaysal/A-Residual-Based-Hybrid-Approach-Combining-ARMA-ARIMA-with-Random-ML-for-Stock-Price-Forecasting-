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


%% Optimal ARIMA Model - Grid Search with AIC Calculation
fprintf("\nFinding optimal ARIMA(p, d, q) model with detailed output...\n");

maxP = 5; % Maximum degrees for AR
maxQ = 5; % Maximum degrees for MA
d = 1; % Degree of differencing
bestAIC = Inf; % Initial large value for comparison
optimalP = 0;
optimalQ = 0;

fprintf("%5s %5s %5s %10s\n", "p", "d", "q", "AIC"); % Column headers

% Grid Search
for p = 0:maxP
    for q = 0:maxQ
        try
            % Define ARIMA(p, d, q) model
            model = arima(p, d, q);
            
            % Estimate the model using original data
            [~, ~, logL] = estimate(model, trainOpening, 'Display', 'off');
            
            % Calculate AIC
            numParams = p + q + 1; % AR, MA parameters, and constant
            aic = -2 * logL + 2 * numParams;
            
            % Print AIC results
            fprintf("%5d %5d %5d %10.4f\n", p, d, q, aic);
            
            % Update the best model
            if aic < bestAIC
                bestAIC = aic;
                optimalP = p;
                optimalQ = q;
            end
        catch ME
            % Indicate failed combinations
            fprintf("%5d %5d %5d %10s - Error: %s\n", p, d, q, "FAILED", ME.message);
        end
    end
end

fprintf("\nOptimal ARIMA model: p = %d, d = %d, q = %d, AIC = %.4f\n", optimalP, d, optimalQ, bestAIC);


%% ARIMA Model - Forecasting and Performance Evaluation
fprintf("\nFitting ARIMA model with differenced data...\n");

% Fit the ARIMA model with the optimal parameters
p = optimalP; % AR parameter
d = 1; % Differencing parameter (set to 1 for ARIMA)
q = optimalQ; % MA parameter

model = arima(p, d, q); % ARIMA(p, d, q) model
fittedModel = estimate(model, trainOpening);

% Forecast using the ARIMA model
numSteps = 10; % Number of steps to forecast
forecastValues = forecast(fittedModel, numSteps, 'Y0', trainOpening);

%% Performance Evaluation
modelName = "ARIMA 10 Days";
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
modelName = "ARIMA 5 Days";
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







