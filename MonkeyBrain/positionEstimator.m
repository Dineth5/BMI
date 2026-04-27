function [x, y] = positionEstimator(test_data, modelParameters)
  % Arguments:

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by the
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times the function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  % Return Value:
  
  % - [x, y]:   current position of the hand

    % the kalman filter values will persist across the calls to the
    % function
    persistent kal_s kal_P kal_dir

    %% Extract parameters
    spikes        = test_data.spikes;
    lda_model     = modelParameters.lda_model;
    mu_s          = modelParameters.mu_s;
    sig_s         = modelParameters.sig_s;
    bin_edges     = modelParameters.bin_edges;
    n_bins        = modelParameters.n_bins;
    pca_models    = modelParameters.pca_models;
    seg_ridge     = modelParameters.seg_ridge;
    seg_vel_ridge = modelParameters.seg_vel_ridge;
    seg_bounds    = modelParameters.seg_bounds;
    kalman_models = modelParameters.kalman_models;
    A             = modelParameters.A;
    lags          = modelParameters.lags;
    bin_size      = modelParameters.bin_size;
    win_min       = modelParameters.win_min;
    win_max       = modelParameters.win_max;
    win_ramp      = modelParameters.win_ramp;
    mu_fr         = modelParameters.mu_fr;
    W_fr          = modelParameters.W_fr;
    n_segs        = modelParameters.n_segs;
    target_pos    = modelParameters.target_pos;
    t            = size(spikes,2);
    neuronNumber = size(spikes,1);
    num_lags     = length(lags);

    % Tunable variables
    temp = 0.01;   % softmax temperature
    reclass_t_start = 380;     % when to start to attempt reclassification
    reclass_t_end = 440;     % stop trying after this
    reclass_min_move = 3;       % minimum decoded movement before reclassifying

    %% Stage 1: Two-stage classification
    if isempty(test_data.decodedHandPos) % if we are at the first movement
        % take only the first 3 bins
        feat_early = zeros(1, neuronNumber*3);
        for b = 1:3
            feat_early((b-1)*neuronNumber+1:b*neuronNumber)=mean(spikes(:, bin_edges(b):bin_edges(b+1)-1), 2)';
        end
    
        % Project using first 3 bins of mu_s/sig_s
        mu_s_early  = modelParameters.mu_s(1:neuronNumber*3);
        sig_s_early = modelParameters.sig_s(1:neuronNumber*3);
        feat_en     = (feat_early - mu_s_early) ./ sig_s_early;
    
        % Compute the lda scores
        scores_early = zeros(1,8);
        for k = 1:8
            cm = lda_model.class_means(k, 1:neuronNumber*3);
            Sw_early = lda_model.Sw_inv(1:neuronNumber*3, 1:neuronNumber*3);
            % difference between test features and that class mean features
            d = feat_en - cm;
            % LDA discriminant function
            scores_early(k) = -0.5*(d*Sw_early)*d' + log(lda_model.priors(k));
        end
        [~, kal_dir]   = max(scores_early); % prediction direction is the one with maximum score
        kal_s          = [test_data.startHandPos(1);
                          test_data.startHandPos(2);
                          0; 0];
        kal_P          = kalman_models{kal_dir}.P;
    end
    
    %% Refined classification window
    % Fires any timestep between reclass_t_start and reclass_t_end
    % Stops once a direction change has been made (or confirmed)

    % if the amount of data points is more than a set number
    already_reclassed = size(test_data.decodedHandPos,2) > ...
                        (reclass_t_end - 320)/bin_size;
    % if not confirmed direction and t is between reclassification boundaries
    if ~already_reclassed && t >= reclass_t_start && t <= reclass_t_end
    
        % Check hand has moved enough to be informative
        if ~isempty(test_data.decodedHandPos) % discards first movement
            move_dist = norm(test_data.decodedHandPos(:,end) - test_data.startHandPos);
        else
            move_dist = 0;
        end
    
        if move_dist >= reclass_min_move % movement has been to great, suspicious
            feat = zeros(1, neuronNumber*n_bins);
            for b = 1:n_bins
                t_start = bin_edges(b);
                t_end   = min(bin_edges(b+1)-1, t);
                if t_start > t % only compute the bins appropiate for this time
                    break
                end
                % save the mean firing rates
                feat((b-1)*neuronNumber+1:b*neuronNumber) = ...
                    mean(spikes(:,t_start:t_end),2)';
            end
            % normalize the mean firing rates
            feat_n = (feat - mu_s) ./ sig_s;
            % Compute the lda scores
            scores = zeros(1,8);
            for k = 1:8
                d = feat_n - lda_model.class_means(k,:);
                scores(k) = -0.5*(d*lda_model.Sw_inv)*d' + log(lda_model.priors(k));
            end
            % see if scores have shifted
            scores_shifted = scores - max(scores);
            % update the weights
            weights        = exp(scores_shifted / temp);
            [~, new_dir]   = max(weights);
            if new_dir ~= kal_dir % if the new direction is different, update it
                kal_dir = new_dir;
                kal_P   = kalman_models{kal_dir}.P;
            end
        end
    end

    %% Stage 2: Lag features via cumsum
    cum_s = cumsum(spikes,2);
    fvec  = zeros(1, neuronNumber*num_lags);

    for li = 1:num_lags
        % Define the time window
        ws = max(1, t-lags(li));
        wl = t-ws+1;
        % Get the cumulative spikes for that specific window,
        % with a different code if its the first window or if
        % there were previous windows
        if ws==1
            wsum=cum_s(:,t);
        else
            wsum=cum_s(:,t)-cum_s(:,ws-1);
        end
        % the feature is the mean firing rate
        fvec((li-1)*neuronNumber+1:li*neuronNumber) = (wsum/wl)';
    end

    %% Stage 3: Ridge predictions
    % Use dir=1 as neutral fallback before classification
    active_dir = max(1, kal_dir);
    % Extract the parameters for this direction
    Wk   = pca_models{active_dir}.W;
    mu_k = pca_models{active_dir}.mu;
    % PCA projection of the features centered (with the bias term)
    pf   = [(fvec-mu_k)*Wk, 1];
    % Find which segment this timestep belongs to
    seg_idx = find(t < seg_bounds, 1) - 1;
    if isempty(seg_idx)
        seg_idx = n_segs; % default: last segment
    end

    % Extract the position weight for that segment
    W_pos = seg_ridge{active_dir}{seg_idx};
    % Predict the displacement
    dp_pos = pf * W_pos;
    pos_pred_x = dp_pos(1) + test_data.startHandPos(1);
    pos_pred_y = dp_pos(2) + test_data.startHandPos(2);

    % Predict the velocity
    W_vel = seg_vel_ridge{active_dir}{seg_idx};
    dp_vel = pf * W_vel;

    % Weighted sum, depending on when in the movement we are
    if t <= 360
        K_blend = 0.95;
    elseif t <= 420
        K_blend = 0.90;
    else
        t_norm_k = min(1.0, (t-420)/win_ramp);
        K_blend  = 0.85 - 0.25*t_norm_k;
    end

    if ~isempty(test_data.decodedHandPos) % if this is not the first position
        prev_pos   = test_data.decodedHandPos(:,end);
        vel_pred_x = prev_pos(1) + dp_vel(1);
        vel_pred_y = prev_pos(2) + dp_vel(2);
        blended_x  = K_blend*pos_pred_x + (1-K_blend)*vel_pred_x;
        blended_y  = K_blend*pos_pred_y + (1-K_blend)*vel_pred_y;
    else % if this is the first position, we only use the position
        blended_x  = pos_pred_x;
        blended_y  = pos_pred_y;
    end

    %% Stage 4: Kalman observation — use blended position
    % Normalised time position in the trial: 0 at t=320, 1 at t=320+win_ramp
    % caps at 1 so window stops growing after win_ramp ms
    t_norm_win = min(1.0, (t-320)/win_ramp);
    % Window size linearly interpolated between win_min and
    % win_max
    win_t = round(win_min + (win_max-win_min)*t_norm_win);
    % Start of the observation window
    t_win = max(1, t-win_t+1);
    % Mean firing rate of each neuron in the window
    fr = mean(spikes(:,t_win:t),2) * 1000/win_t;
    % the neural observation is the projection through the PCA
    z_neural = W_fr'*(fr-mu_fr);
    % Fuse the predicted position with the neural observation
    z = [blended_x; blended_y; z_neural];   % blended not raw

    %% Stage 5: Kalman predict + update
    % Use averaged model before classification, direction model after
    if kal_dir == 0 % if we have not been able to predict the direction, we use a combination of all of them
        H = zeros(size(kalman_models{1}.H));
        Q = zeros(size(kalman_models{1}.Q));
        R = zeros(size(kalman_models{1}.R));
        for k = 1:8
            H = H + kalman_models{k}.H/8;
            Q = Q + kalman_models{k}.Q/8;
            R = R + kalman_models{k}.R/8;
        end
    else % extract the kalman model values
        H = kalman_models{kal_dir}.H;
        Q = kalman_models{kal_dir}.Q;
        R = kalman_models{kal_dir}.R;
    end
    % Normalised time position in the trial: 0 at t=320, 1 at t=400
    t_norm   = min(1.0, (t-320)/400);
    % The decay represents the hand decelerating
    decay_t  = 0.85 - 0.15*t_norm;
    % construct the physics simulator A with the new decay
    A_t      = A;
    A_t(3,3) = decay_t;
    A_t(4,4) = decay_t;
    % state prediction
    s_pred = A_t * kal_s;
    % current P prediction
    P_pred = A_t * kal_P * A_t' + Q;
    % Kalman gain
    K     = P_pred * H' / (H*P_pred*H' + R);
    % state as combination of prediction and observation
    s_upd = s_pred + K*(z - H*s_pred);

    % Velocity cap
    if t <= 360 % the maximum reasonable velocity changes through the movement
        max_vel = 0.15;
    elseif t <= 420
        max_vel = 0.25;
    else
        t_norm_vel = min(1.0, (t-420)/300);
        max_vel    = 0.35*(1-0.4*t_norm_vel);
    end
    % we extract the velocity from the state result of the kalman filter
    vel_mag = norm(s_upd(3:4));
    % if the maximum reasonable velocity is too large, we reduce it
    if vel_mag > max_vel
        s_upd(3:4) = s_upd(3:4)*(max_vel/vel_mag);
    end
    % update the P in a persistent variable
    kal_P = (eye(4) - K*H)*P_pred;
    % update the previous state for the next iteration
    kal_s = s_upd;

    %% Stage 6: Target pull (only after classification)
    if kal_dir > 0 % if we managed to predict the direction
        % we get the target of the movement for that direction
        target_soft = target_pos(:,kal_dir);

        % depending on what time we are at
        t_norm_target = min(1.0, 0.9*max(0.0, (t-400)/200));
        % Pull the position to the target
        x = (1-t_norm_target)*s_upd(1) + t_norm_target*target_soft(1);
        y = (1-t_norm_target)*s_upd(2) + t_norm_target*target_soft(2);
    else % if we didnt manage to predict the direction, we stick to the kalman prediction
        x = s_upd(1);
        y = s_upd(2);
    end
end