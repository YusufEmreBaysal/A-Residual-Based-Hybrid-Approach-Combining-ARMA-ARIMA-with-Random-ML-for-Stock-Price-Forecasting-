clc;
clear;
close all;

%% Load Dataset
fprintf("Loading dataset...\n");
data = readtable('dataset.xlsx');
%data = data(21:end, :);
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


%% GBM Model
% model here
fprintf("\nTraining GBM Model using GBM...\n");

windowSize = 15;
X_train = [];
Y_train = [];

for i = (windowSize+1):length(diffTrainOpening)
    X_train = [X_train; diffTrainOpening(i-windowSize:i-1)'];
    Y_train = [Y_train; diffTrainOpening(i)];
end

% Train a GBM model (Gradient Boosted Regression Trees)
gbmModel = fitensemble(X_train, Y_train, 'LSBoost', 100, 'Tree');

%% Prediction - Batch
%prediction vector
fprintf("\nPredicting future 10 values using GBM Model...\n");

% Tüm eğitim serisini bilinen olarak al
knownSeries = diffTrainOpening;
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

% % Convert forecasts back to the original scale
lastOriginalValue = trainOpening(end); % Last original value from the training set
predictedRealPrices = cumsum([lastOriginalValue; predictedPrices]); % Cumulative sum to restore original scale
predictedRealPrices = predictedRealPrices(2:end); % Remove the initial value (start point)

% %% Lag ve LSBoost Değerlerinin Karşılaştırılması
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
%         for i = (currentLag+1):length(diffTrainOpening)
%             X_tmp = [X_tmp; diffTrainOpening(i-currentLag:i-1)'];
%             Y_tmp = [Y_tmp; diffTrainOpening(i)];
%         end
% 
%         % GBM Modelini eğit
%         gbmModel_tmp = fitensemble(X_tmp, Y_tmp, 'LSBoost', currentTrees, 'Tree');
% 
%         % 10 günlük tahmin yap
%         knownSeries_tmp = diffTrainOpening;
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
% 
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
% 


%% Performance Evaluation
modelName = "GBM (diffData) 10 days";
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
modelName = "GBM (diffData) 5 days";
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


