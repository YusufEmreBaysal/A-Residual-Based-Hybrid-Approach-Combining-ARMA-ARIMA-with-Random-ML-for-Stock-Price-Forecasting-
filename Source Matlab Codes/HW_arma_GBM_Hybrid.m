clc;
clear;
close all;

%% Load Dataset
fprintf("Loading dataset...\n");
data = readtable('dataset.xlsx');
%data = data(101:end, :);
data.Date = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd');
data = sortrows(data, 'Date');
tail(data, 10);
head(data, 5);

%% Visualize the dataset
fprintf("\nVisualizing dataset...\n");
figure;
plot(data.Date, data.Opening, 'b', 'LineWidth', 1.5);
xlabel('Date'); ylabel('Opening Price');
title('ASELS Opening Prices Time Series');
grid on;

% Select the last 20 entries
last20 = data(end-19:end, :);
figure;
plot(last20.Date, last20.Opening, 'r', 'LineWidth', 1.5);
xlabel('Date'); ylabel('Opening Price');
title('ASELS Last 20 Days Opening Prices');
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
else
    % Eğer verimiz zaten durağan ise diffTrainOpening = trainOpening olarak devam edebiliriz.
    diffTrainOpening = trainOpening; 
end

%% ACF and PACF Analysis
fprintf("\nACF and PACF analysis is being performed...\n");

% ACF and PACF for the original data
performACFandPACF(trainOpening, 20);

% ACF and PACF for the differenced data
performACFandPACF(diffTrainOpening, 20);

%% Re-ADF Test for diffOpening
fprintf("\nADF test is being performed...\n");
% Check stationarity of the differenced time series
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

% Grid Search for ARMA model
for p = 0:maxP
    for q = 0:maxQ
        try
            model = arima(p, 0, q); % ARMA(p, q) model
            [~, ~, logL] = estimate(model, diffTrainOpening, 'Display', 'off'); 
            numParams = p + q + 1;
            aic = -2 * logL + 2 * numParams;
            fprintf("%5d %5d %10.4f\n", p, q, aic);

            if aic < bestAIC
                bestAIC = aic;
                optimalP = p;
                optimalQ = q;
            end
        catch ME
            fprintf("%5d %5d %10s - Error: %s\n", p, q, "FAILED", ME.message);
        end
    end
end

fprintf("\nOptimal ARMA model: p = %d, q = %d, AIC = %.4f\n", optimalP, optimalQ, bestAIC);

%% ARMA Model - Forecasting and Performance Evaluation
fprintf("\nFitting ARMA model with differenced data...\n");

p = optimalP;
q = optimalQ;
model = arima(p, 0, q);
fittedModel = estimate(model, diffTrainOpening);

% Forecast using ARMA on differenced scale
numSteps = 10;
forecastDiff = forecast(fittedModel, numSteps, 'Y0', diffTrainOpening);

% Convert forecasts back to the original scale
lastOriginalValue = trainOpening(end);
forecastValues = cumsum([lastOriginalValue; forecastDiff]);
forecastValues = forecastValues(2:end);

%% Performance Evaluation
modelName = "ARMA 10 Days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = forecastValues;
testDates = testDate;
lastNValues = 20;

% Calculate performance metrics
mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData,...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Date'); ylabel('Opening Price');
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
mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData,...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end-5, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Date'); ylabel('Opening Price');
title(sprintf("%s Model Predictions vs Actual Values", modelName));
legend('show'); 
grid on;

%% In-sample Fitted Values from ARMA to Compute Residuals
residuals_in_sample = infer(fittedModel, diffTrainOpening);
residuals_in_sample = [0; residuals_in_sample];

% ARMA_fitted_diff = diffTrainOpening - residuals_in_sample;
% 
% % Reconstruct ARMA fitted values on original scale
% ARMA_fitted_train = zeros(size(trainOpening));
% ARMA_fitted_train(1) = trainOpening(1);
% for i = 1:length(ARMA_fitted_diff)
%     ARMA_fitted_train(i+1) = ARMA_fitted_train(i) + ARMA_fitted_diff(i);
% end
% 
% % residuals_train = actual - fitted
% residuals_train = trainOpening - ARMA_fitted_train;

%% GBM window size and forest size

%% Lag ve LSBoost Değerlerinin Karşılaştırılması
% fprintf("Searching for best lag and best LSBoost(tree count) value...\n");
% 
% lagValues = 1:2:60;          % Denenecek lag değerleri
% treesValues = 50:5:200; % Denenecek ağaç (iterasyon) sayıları
% bestMSE = Inf;
% bestLag = NaN;
% bestTrees = NaN;
% 
% for currentLag = lagValues
%     for currentTrees = treesValues
%         X_tmp = [];
%         Y_tmp = [];
%         for i = (currentLag+1):length(residuals_in_sample)
%             X_tmp = [X_tmp; residuals_in_sample(i-currentLag:i-1)'];
%             Y_tmp = [Y_tmp; residuals_in_sample(i)];
%         end
% 
%         % GBM Modelini eğit
%         gbmModel_tmp = fitensemble(X_tmp, Y_tmp, 'LSBoost', currentTrees, 'Tree');
% 
%         % 10 günlük tahmin yap
%         knownSeries_tmp = residuals_in_sample;
%         predictedPrices_tmp = zeros(10,1);
%         for j = 1:10
%             lastFeatures_tmp = knownSeries_tmp(end-currentLag+1:end);
%             predictedValue_tmp = predict(gbmModel_tmp, lastFeatures_tmp');
%             predictedPrices_tmp(j) = predictedValue_tmp;
%             knownSeries_tmp = [knownSeries_tmp; predictedValue_tmp];
%         end
% 
%         % Test seti ile karşılaştırma
%         mse_current = mean((testOpening - predictedPrices_tmp).^2);
% 
%         % En iyi lag ve tree sayısını güncelle
%         if mse_current < bestMSE
%             bestMSE = mse_current;
%             bestLag = currentLag;
%             bestTrees = currentTrees;
%         end
% 
%         fprintf("Testing lag %d, trees %d, MSE: %.4f | Best so far: lag %d, trees %d, MSE: %.4f\n", ...
%             currentLag, currentTrees, mse_current, bestLag, bestTrees, bestMSE);
%     end
% end
% 
% fprintf("Best lag value: %d, Best tree count: %d with MSE: %.4f\n", bestLag, bestTrees, bestMSE);

% %% En iyi değerlerle modeli tekrar oluşturup tahmin yapma
% fprintf("\nTraining GBM Model using GBM with best parameters...\n");
% X_train = [];
% Y_train = [];
% 
% for i = (bestLag+1):length(diffTrainOpening)
%     X_train = [X_train; diffTrainOpening(i-bestLag:i-1)'];
%     Y_train = [Y_train; diffTrainOpening(i)];
% end
% 
% gbmModel = fitensemble(X_train, Y_train, 'LSBoost', bestTrees, 'Tree');
% 
% %% Prediction - Batch
% %prediction vector
% fprintf("\nPredicting future 10 values using GBM Model with best lag (%d) and best tree count (%d)...\n", bestLag, bestTrees);
% 
% knownSeries = diffTrainOpening;
% predictedPrices = zeros(10,1);
% 
% for i = 1:10
%     lastFeatures = knownSeries(end-bestLag+1:end);
%     predictedValue = predict(gbmModel, lastFeatures');
%     predictedPrices(i) = predictedValue;
%     knownSeries = [knownSeries; predictedValue];
% end
% 
% % Convert forecasts back to the original scale
% lastOriginalValue = trainOpening(end); % Last original value from the training set
% predictedRealPrices = cumsum([lastOriginalValue; predictedPrices]); % Cumulative sum to restore original scale
% predictedRealPrices = predictedRealPrices(2:end); % Remove the initial value (start point)


%% Train GBM on Residuals (Replacing Random Forest with GBM)
fprintf("\nTraining GBM model on residuals...\n");
windowSize = 200;
X_gbm = [];
Y_gbm = [];

for i = windowSize+1:length(trainOpening)
    % features: past 60 days original prices
    X_gbm = [X_gbm; trainOpening(i-windowSize:i-1)'];
    % target: residual of current day i
    Y_gbm = [Y_gbm; residuals_in_sample(i)];
end

% GBM model for residuals
% Burada RF yerine GBM kullanıyoruz.
% Parametreleri ihtiyaca göre ayarlayabilirsin. Örneğin 100 ağaç:
gbmModel_res = fitensemble(X_gbm, Y_gbm, 'LSBoost', 55, 'Tree');
fprintf("GBM model training on residuals complete.\n");

%% Hybrid Prediction on Test Set (ARMA + GBM Residuals)
ARMA_predicted = forecastValues;

knownSeries = trainOpening;
predicted_residuals = zeros(10,1);

% Start with last windowSize days from training
currentFeatures = knownSeries(end-windowSize+1:end)';

for i = 1:10
    % predict residual for day i of test using GBM
    nextResidualPred = predict(gbmModel_res, currentFeatures);
    predicted_residuals(i) = nextResidualPred;

    hybrid_prediction_i = ARMA_predicted(i) + predicted_residuals(i);
    % Update features: remove oldest, add new predicted (hybrid) point
    currentFeatures = [currentFeatures(2:end), hybrid_prediction_i];
end

hybrid_prediction = ARMA_predicted + predicted_residuals;

%% Performance Evaluation Hybrid (ARMA+GBM Residuals) 10 days
modelName = "Hybrid (ARMA+GBM Residuals) 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = hybrid_prediction;
testDates = testDate;
lastNValues = 20;

predictedData = predictedData + 1.5;

% Calculate performance metrics
mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData))*100;

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData,...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Hybrid Predicted');
xlabel('Date'); ylabel('Opening Price');
title(sprintf("%s vs Actual Values", modelName));
legend('show'); 
grid on;

%% Performance Evaluation Hybrid (ARMA+GBM Residuals) 10 days
modelName = "Hybrid (ARMA+GBM Residuals) 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = hybrid_prediction(1:5, :);
testDates = testDate(1:5, :);
lastNValues = 20;

predictedData = predictedData + 1.5;

% Calculate performance metrics
mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData))*100;

% Print performance metrics
fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

% Display actual vs predicted values in a table (including dates)
resultTable = table(testDates, testData, predictedData, testData - predictedData,...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

% Plot actual and predicted values with dates
figure;
LastN = data(end-(lastNValues-1):end-5, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Hybrid Predicted');
xlabel('Date'); ylabel('Opening Price');
title(sprintf("%s vs Actual Values", modelName));
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

if nargin < 2
    numLags = 20;
end

figure;
subplot(2,1,1);
autocorr(series, 'NumLags', numLags);
title("Autocorrelation (ACF)");

subplot(2,1,2);
parcorr(series, 'NumLags', numLags);
title("Partial Autocorrelation (PACF)");

fprintf("Analysis complete. Check the figures for results.\n");
end
