function [x, y] = positionEstimator(test_data, modelParameters)
    spikes = test_data.spikes;
    t = size(spikes, 2);
    
    % Stage 1: Direction Classification using Temporal Beams
    % We use the fixed 1-300ms window to lock in the direction early
    feat = [mean(spikes(:,1:100),2)', mean(spikes(:,101:200),2)', mean(spikes(:,201:300),2)'];
    feat_n = (feat - modelParameters.mu_s) ./ modelParameters.sig_s;
    
    scores = zeros(1,8);
    for k = 1:8
        scores(k) = feat_n * modelParameters.svm_models{k}.w + modelParameters.svm_models{k}.b;
    end
    [~, dir] = max(scores);
    
    % Stage 2: Velocity-Based Estimation
    % Hold at start position during the initial 320ms window
    if t <= 320
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end
    
    % Feature extraction for the current time step
    fvec = [];
    for lag_val = modelParameters.lags
        ws = max(1, t - lag_val);
        fvec = [fvec, mean(spikes(:,ws:t), 2)'];
    end
    
    % Project to PCA space and predict Velocity (Delta Position)
    Wk = modelParameters.pca_models{dir}.W;
    mu_k = modelParameters.pca_models{dir}.mu;
    pf = [(fvec - mu_k)*Wk, 1];
    delta_pos = pf * modelParameters.ridge_models{dir};
    
    % Position update: Previous Decoded Position + Velocity
    % If decodedHandPos is empty, use startHandPos
    if isempty(test_data.decodedHandPos)
        prev_pos = test_data.startHandPos(1:2);
    else
        prev_pos = test_data.decodedHandPos(:, end);
    end
    
    raw_x = prev_pos(1) + delta_pos(1);
    raw_y = prev_pos(2) + delta_pos(2);
    
    % Clipping: Global workspace boundary
    max_reach = 97000;
    x = max(min(raw_x, max_reach), -max_reach);
    y = max(min(raw_y, max_reach), -max_reach);
end