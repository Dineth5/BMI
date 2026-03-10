% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

%For the module BMI, we are tasked with decoding a monkey's brain signal and hand movement, by estimating its position based on the neural network
%testFunction_for_students_MTb was given by the lecturer, I have edited it slighly to include a tic tac (to calculate the time it has taken) along with RSME
%The aim is to have the fastest and most accurate code (10:90) weightage between the two factors, but accuracy is bounded from 0-100 RSME, while time in this case is unbounded

%positionEstimatorTraining and positionEstimator were made by us, how do I improve it, i have attached the output when we run the test function "RMSE = testFunction_for_students_MTb('m_codes_svm11')"
%As you can see the graph occasionally jumps between paths, we would like
%to minimize the instances these occur

function RMSE = testFunction_for_students_MTb(teamName)
tic; % TIMER START

load monkeydata0.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

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
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end




legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions) 

rmpath(genpath(teamName))


elapsedTime = toc; % TIMER END
fprintf('\nDecoding complete in %.2f seconds.\n', elapsedTime);
end
