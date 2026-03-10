function RMSE = testFunction_for_students_MTb_ERROR_ANA(teamName)
tic; 

load monkeydata0.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')
meanSqError = 0;
n_predictions = 0;  

% --- NEW: DATA COLLECTION FOR ERROR ANALYSIS ---
all_pred = {};
all_real = {};
test_labels = [];
trial_counter = 1;

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];
        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        end
        
        % --- NEW: RECORD DATA FOR ANALYSIS ---
        all_pred{trial_counter} = decodedHandPos;
        all_real{trial_counter} = testData(tr,direc).handPos(1:2, times);
        test_labels(trial_counter) = direc;
        trial_counter = trial_counter + 1;
        
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

legend('Decoded Position', 'Actual Position')
RMSE = sqrt(meanSqError/n_predictions);

% --- NEW: TRIGGER ANALYSIS PLOTS ---
plot_directional_rmse(all_pred, all_real, test_labels);
plot_temporal_error(all_pred, all_real);

rmpath(genpath(teamName))
elapsedTime = toc; 
fprintf('\nDecoding complete in %.2f seconds.\n', elapsedTime);
end

% --- SUB-FUNCTIONS (PASTE AT BOTTOM OF FILE) ---

function plot_directional_rmse(all_pred, all_real, test_labels)
    dir_rmse = zeros(1, 8);
    for d = 1:8
        dir_indices = find(test_labels == d);
        total_sq_error = 0;
        total_points = 0;
        for i = 1:length(dir_indices)
            idx = dir_indices(i);
            pred = all_pred{idx};
            real = all_real{idx};
            sq_err = sum((pred - real).^2, 1);
            total_sq_error = total_sq_error + sum(sq_err);
            total_points = total_points + size(real, 2);
        end
        dir_rmse(d) = sqrt(total_sq_error / total_points);
    end
    figure; bar(1:8, dir_rmse);
    xlabel('Direction Index'); ylabel('RMSE (mm)');
    title('RMSE Breakdown by Direction'); grid on;
end

function plot_temporal_error(all_pred, all_real)
    max_t = 1000; 
    time_bins = 20:20:max_t;
    temporal_error = zeros(size(time_bins));
    counts = zeros(size(time_bins));
    for i = 1:length(all_pred)
        pred = all_pred{i};
        real = all_real{i};
        for b = 1:length(time_bins)
            idx = time_bins(b) / 20;
            if idx <= size(pred, 2)
                temporal_error(b) = temporal_error(b) + sum((pred(:,idx) - real(:,idx)).^2);
                counts(b) = counts(b) + 1;
            end
        end
    end
    temporal_rmse = sqrt(temporal_error ./ (counts * 2)); 
    figure; plot(time_bins, temporal_rmse, 'LineWidth', 2, 'Color', 'r');
    xlabel('Time from Start (ms)'); ylabel('RMSE (mm)');
    title('Temporal Error Analysis (Drift Check)'); grid on;
end