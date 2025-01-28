clc;
clear;
close all;

%% Load Dataset
fprintf("Loading dataset...\n");
data = readtable('dataset.xlsx');
data.Date = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd');
data = sortrows(data, 'Date');
tail(data, 10); % Display the last 10 rows of the dataset
head(data, 5); % Display the first 5 rows of the dataset

%% Visualize the dataset
fprintf("\nVisualizing dataset...\n");
figure;
plot(data.Date, data.Opening, 'b', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Opening Price');
title('ASELS Opening Prices Time Series');
grid on;

% Select the last 20 entries
last20 = data(end-19:end, :);
figure;
plot(last20.Date, last20.Opening, 'r', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Opening Price');
title('ASELS Last 20 Days Opening Prices');
grid on;

%% Split Data into Train and Test Sets
fprintf("\nSplitting dataset into train and test sets...\n");
trainOpening = data.Opening(1:end-10); % Training set for opening prices
testOpening = data.Opening(end-9:end); % Test set for opening prices (last 10 days)

trainDate = data.Date(1:end-10); % Training set for dates
testDate = data.Date(end-9:end); % Test set for dates (last 10 days)


%% Find optimal ARIMA(p,d,q) based on AIC
fprintf("\nFinding optimal ARIMA(p, d, q) model with detailed output...\n");
maxP = 5; 
maxQ = 5; 
d = 1; 
bestAIC = Inf; 
optimalP = 0;
optimalQ = 0;

fprintf("%5s %5s %5s %10s\n", "p", "d", "q", "AIC");
for p = 0:maxP
    for q = 0:maxQ
        try
            model = arima(p, d, q);
            [~, ~, logL] = estimate(model, trainOpening, 'Display', 'off');
            numParams = p + q + 1; % rough count of parameters
            aic = -2 * logL + 2 * numParams;
            fprintf("%5d %5d %5d %10.4f\n", p, d, q, aic);
            if aic < bestAIC
                bestAIC = aic;
                optimalP = p;
                optimalQ = q;
            end
        catch
            fprintf("%5d %5d %5d %10s\n", p, d, q, "FAILED");
        end
    end
end

fprintf("\nOptimal ARIMA model: p = %d, d = %d, q = %d, AIC = %.4f\n", optimalP, d, optimalQ, bestAIC);

%% Fit the chosen ARIMA model
fprintf("\nFitting chosen ARIMA model...\n");
p = optimalP;
q = optimalQ;
model = arima(p, d, q);
fittedModel = estimate(model, trainOpening);

% Forecast 10 steps ahead with ARIMA
numSteps = 10;
forecastValues_ARIMA = forecast(fittedModel, numSteps, 'Y0', trainOpening);

%% ARIMA Performance
modelName = "ARIMA 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = forecastValues_ARIMA;
testDates = testDate;
lastNValues = 20;

mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

figure;
LastN = data(end-(lastNValues-1):end, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'ARIMA Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s Model Predictions vs Actual Values", modelName));
legend('show');
grid on;

%% ARIMA Performance
modelName = "ARIMA 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = forecastValues_ARIMA(1:5, :);
testDates = testDate(1:5, :);
lastNValues = 20;

mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

figure;
LastN = data(end-(lastNValues-1):end-5, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'ARIMA Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s Model Predictions vs Actual Values", modelName));
legend('show');
grid on;

%% Compute ARIMA In-sample Predictions (Training) to get Residuals for RF
% İn-sample residual değerlerini elde etmek için infer kullan

ARIMA_predicted = forecastValues_ARIMA;
residuals_in_sample = infer(fittedModel, trainOpening);

%% Train Random Forest on Residuals
fprintf("\nTraining Random Forest model on residuals...\n");
windowSize = 60; % Use the past 60 days as features

% RF features (X) and target (Y) for residual prediction
% residuals_train(i) corresponds to trainOpening(i+1)
% So we should build features up to length(residuals_train) position:
X_rf = [];
Y_rf = [];
% We can only start creating windows from where we have enough past data
% Also note: residuals_train starts from the second day of training, so indexing carefully:
for i = windowSize : (length(residuals_in_sample)-0)
    % Features: the past windowSize values of trainOpening (aligning with residual i)
    % residuals_train(i) corresponds to trainOpening(i+1), so for features we use trainOpening(i-windowSize+1 : i)
    % Actually, residuals_train(i) is about trainOpening(i+1), let's keep it simple:
    X_rf = [X_rf; trainOpening(i-windowSize+1:i)']; 
    Y_rf = [Y_rf; residuals_in_sample(i)];
end

numTrees = 100;
rfModel_res = TreeBagger(numTrees, X_rf, Y_rf, 'Method', 'regression', 'OOBPrediction', 'off');
fprintf("Random Forest model training on residuals complete.\n");

%% Hybrid Prediction on Test Set
% 1) We already have ARIMA predictions for the next 10 days: ARIMA_predicted
% 2) Predict residuals for the next 10 days using RF
% To predict residuals for test days, we must use the last windowSize days from the end of training + any predicted days.

% Build initial feature vector for the first test prediction step
knownSeries = trainOpening; 
% We'll iteratively predict residual day by day
predicted_residuals = zeros(10,1);

% For residual prediction, we use the same approach as RF code:
currentFeatures = knownSeries(end-windowSize+1:end)'; % last windowSize days from training

for i = 1:10
    % Predict residual for day i of test
    nextResidualPred = predict(rfModel_res, currentFeatures);
    predicted_residuals(i) = nextResidualPred;

    hybrid_prediction_i = ARIMA_predicted(i) + predicted_residuals(i);

    currentFeatures = [currentFeatures(2:end), hybrid_prediction_i];
end

% Sonunda, hibrit tahmin:
hybrid_prediction = ARIMA_predicted + predicted_residuals;

%% Performans Değerlendirmesi Hibrit Model
modelName = "Hybrid (ARIMA+RF Residuals) 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = hybrid_prediction;
testDates = testDate;
lastNValues = 20;

mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

figure;
LastN = data(end-(lastNValues-1):end, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Hybrid Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s vs Actual Values", modelName));
legend('show');
grid on;

%% Performans Değerlendirmesi Hibrit Model
modelName = "Hybrid (ARIMA+RF Residuals) 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedData = hybrid_prediction(1:5, :);
testDates = testDate(1:5, :);
lastNValues = 20;

mae = mean(abs(testData - predictedData));
mse = mean((testData - predictedData).^2);
rmse = sqrt(mse);
mape = mean(abs((testData - predictedData) ./ testData)) * 100;

fprintf("\nPerformance Metrics for %s:\n", modelName);
fprintf("MAE: %.4f\n", mae);
fprintf("MSE: %.4f\n", mse);
fprintf("RMSE: %.4f\n", rmse);
fprintf("MAPE: %.2f%%\n", mape);

resultTable = table(testDates, testData, predictedData, testData - predictedData, ...
    'VariableNames', {'Date', 'Actual', 'Predicted', 'Difference'});
fprintf("\nComparison Table for %s:\n", modelName);
disp(resultTable);

figure;
LastN = data(end-(lastNValues-1):end-5, :);
plot(LastN.Date, LastN.Opening, 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(testDates, predictedData, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Hybrid Predicted');
xlabel('Date');
ylabel('Opening Price');
title(sprintf("%s vs Actual Values", modelName));
legend('show');
grid on;
