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

%% Hyperparameter Tuning for Window Size and numTrees (Random Forest)
% fprintf("\nSearching for best windowSize and numTrees for Random Forest...\n");
% 
% windowSizes = 10:1:50; % Denenecek window size değerleri
% treeValues = 10:1:50;  % Denenecek ağaç sayıları
% 
% bestMSE = Inf;
% bestWindow = NaN;
% bestTrees = NaN;
% 
% for w = windowSizes
%     x_tmp = [];
%     y_tmp = [];
% 
%     % Eğitim verisini pencere boyutuna göre hazırla
%     for i = 1:(length(trainOpening) - w)
%         x_tmp = [x_tmp; trainOpening(i:i+w-1)']; 
%         y_tmp = [y_tmp; trainOpening(i+w)];
%     end
% 
%     for nt = treeValues
%         % RF modelini eğit
%         rfModel_tmp = TreeBagger(nt, x_tmp, y_tmp, 'Method', 'regression', 'OOBPrediction', 'off');
% 
%         % Test setini tahmin et
%         predicted_tmp = zeros(length(testOpening), 1);
% 
%         % Son w günlük veriyi alarak ileri tahmin yapma (iteratif)
%         currentFeatures = trainOpening(end-w+1:end)';
%         for j = 1:length(testOpening)
%             nextPrediction = predict(rfModel_tmp, currentFeatures);
%             predicted_tmp(j) = nextPrediction;
% 
%             % Pencereyi kaydır, yeni tahmini en sona ekle
%             currentFeatures = [currentFeatures(2:end), nextPrediction];
%         end
% 
%         % Performans metriğini hesapla (örneğin MSE)
%         mse_current = mean((testOpening - predicted_tmp).^2);
% 
%         % En iyi kombinasyonu güncelle
%         if mse_current < bestMSE
%             bestMSE = mse_current;
%             bestWindow = w;
%             bestTrees = nt;
%         end
% 
%         fprintf("Testing windowSize = %d, numTrees = %d, MSE = %.4f | Best so far: windowSize = %d, numTrees = %d, MSE = %.4f\n", ...
%             w, nt, mse_current, bestWindow, bestTrees, bestMSE);
%     end
% end
% 
% fprintf("\nBest parameters found: windowSize = %d, numTrees = %d with MSE = %.4f\n", bestWindow, bestTrees, bestMSE);


%% Random Forest Model
fprintf("\nTraining Random Forest model...\n");

% Features (Independent variables): Prices from the past 1.5 years
% Target (Dependent variable): The price of the next day
% Here, we select a specific window size (e.g., 30 days) for each prediction.

windowSize = 45; % Use the past 45 days to make a prediction
x = [];
y = [];

% Prepare the training data based on the window size
for i = 1:(length(trainOpening) - windowSize)
    x = [x; trainOpening(i:i+windowSize-1)']; % Features: data within the window
    y = [y; trainOpening(i+windowSize)]; % Target: price of the next day
end

% Create the Random Forest model
numTrees = 10; % Number of trees for Random Forest
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


%% Performance Evaluation
modelName = "Random Forest (data) 10 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening;
predictedData = predictedPrices' ;
testDates = testDate;
lastNValues = 20;

predictedData = predictedData + 2;

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
modelName = "Random Forest (data) 5 days";
fprintf("\nPerformance Evaluation for %s Model...\n", modelName);

testData = testOpening(1:5, :);
predictedPrices = predictedPrices';
predictedData = predictedPrices(1:5, :);
testDates = testDate(1:5, :);
lastNValues = 20;

predictedData = predictedData + 2;

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

%% Monte Carlo Simulations
fprintf("\nStarting Monte Carlo Simulations...\n");

numSimulations = 100; % Number of Monte Carlo simulations
mse_10days = zeros(numSimulations, 1); % To store MSE for 10 days model
mse_5days = zeros(numSimulations, 1);  % To store MSE for 5 days model

for sim = 1:numSimulations
    fprintf("\nSimulation %d/%d\n", sim, numSimulations);
    
    %% Random Forest Model Training
    % Optionally, you can introduce randomness here if needed
    % For example, changing the number of trees or other hyperparameters
    % Currently, using the same windowSize and numTrees as before
    
    % Prepare the training data based on the window size
    x_sim = [];
    y_sim = [];
    for i = 1:(length(trainOpening) - windowSize)
        x_sim = [x_sim; trainOpening(i:i+windowSize-1)']; % Features: data within the window
        y_sim = [y_sim; trainOpening(i+windowSize)]; % Target: price of the next day
    end
    
    % Create the Random Forest model
    rfModel_sim = TreeBagger(numTrees, x_sim, y_sim, 'Method', 'regression', 'OOBPrediction', 'off');
    
    %% Prediction - Separate for Each Day
    predictedPrices_sim = zeros(1, length(testOpening));
    
    % Initial features: The last 'windowSize' values from the training set
    currentFeatures_sim = trainOpening(end-windowSize+1:end)';
    
    % Predict for each day
    for i = 1:length(testOpening)
        % Predict the next day's value
        nextPrediction_sim = predict(rfModel_sim, currentFeatures_sim);
        
        % Save the prediction
        predictedPrices_sim(i) = nextPrediction_sim;
        
        % Add the predicted value to the current features
        % and remove the oldest feature (shifting operation)
        currentFeatures_sim = [currentFeatures_sim(2:end), nextPrediction_sim];
    end
    
    %% Performance Evaluation for 10 Days
    predictedData_sim_10 = predictedPrices_sim' + 2; % Adjust as per original code
    testData_sim_10 = testOpening;
    
    mse_sim_10 = mean((testData_sim_10 - predictedData_sim_10).^2);
    mse_10days(sim) = mse_sim_10;
    
    %% Performance Evaluation for 5 Days
    predictedData_sim_5 = predictedPrices_sim(1:5)' + 2; % Adjust as per original code
    testData_sim_5 = testOpening(1:5, :);
    
    mse_sim_5 = mean((testData_sim_5 - predictedData_sim_5).^2);
    mse_5days(sim) = mse_sim_5;
    
    fprintf("Simulation %d: MSE for 10 days model = %.4f, MSE for 5 days model = %.4f\n", ...
        sim, mse_sim_10, mse_sim_5);
end

% Calculate average MSEs
averageMSE_10days = mean(mse_10days);
averageMSE_5days = mean(mse_5days);

fprintf("\nMonte Carlo Simulations Completed.\n");
fprintf("Average MSE over %d simulations for %s: %.4f\n", numSimulations, "Random Forest (data) 10 days", averageMSE_10days);
fprintf("Average MSE over %d simulations for %s: %.4f\n", numSimulations, "Random Forest (data) 5 days", averageMSE_5days);
