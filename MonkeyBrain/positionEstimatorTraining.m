function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of the
  %     model and which can be used by the "positionEstimator" function

    [n_trials, n_dirs] = size(training_data); % get the dimensions of the data
    neuronNumber = size(training_data(1,1).spikes,1); % get the number of neurons

    %% Parameters
    bin_size   = 20;                        % bin size for ridge and pca
    lambda     = 0.1;                       % lambda for ridge
    lags       = [50,100,150,200,250,300];  % for ridge and pca
    nc         = 300;                       % number of principal components kept by pca
    nc_kalman  = 25;                        % number of principal components for kalman
    smooth_win = 3;                         % smoothing window for velocity
    decay      = 0.85;                      % in kalman filter, velocity is multiplied by 0.85 each 20ms step, some deacceleration
    q_scale    = 0.05;                      % scale for Q in kalman filter
    r_scale    = 10.0;                      % scale for R neural in kalman filter
    r_pos      = 0.2;                       % scale for ridge noise in kalman filter
    win_min    = 100;                       % window minimum or maximum for the kalman pca
    win_max    = 300;
    win_ramp   = 300;                       % the kalman pca window reaches maximum size by t=620ms
    shrinkage = 0.3;                        % Regularisation strength for LDA

   %% Step 1: LDA for direction classification with every neuron decomposed in 4 bins
    % Goal: learn a model that maps pre-movement neural activity, to obtain reaching direction (1-8)
    % This runs ONCE at the start of each test trial to identify which direction
    % the monkey is about to reach, selecting the correct ridge/Kalman models

    n_bins    = 4; % we only take the first 420 ms, divide in 4 bins and train the model
    bin_edges = [1, 101, 201, 301, 421]; % Boundaries of the 4 bins in milliseconds:

    % We preallocate the features and labels for speed
    lda_features = zeros(n_trials*n_dirs, neuronNumber*n_bins);
    lda_labels = zeros(n_trials*n_dirs, 1);
    % Extract the features for the lda
    row = 1;   % running row index into lda_features
    for k = 1:n_dirs       % loop over 8 reaching directions
        for n = 1:n_trials % loop over 50 training trials per direction
            sp   = training_data(n,k).spikes;
            feat = zeros(1, neuronNumber*n_bins);
            for b = 1:n_bins
                % Each feat is the mean of that neuron for that time period
                feat((b-1)*neuronNumber+1:b*neuronNumber) = ...
                    mean(sp(:, bin_edges(b):bin_edges(b+1)-1), 2)';
            end
            % We store the features and its labels
            lda_features(row,:) = feat;   % store this trial's features
            lda_labels(row)     = k;      % store its direction label
            row = row + 1;
        end
    end

    % Normalize the features for the lda
    mu_s  = mean(lda_features, 1);
    sig_s = std(lda_features, 0, 1); 
    sig_s(sig_s==0) = 1; % avoid deviation=0
    X_norm = (lda_features - mu_s) ./ sig_s;
    
    % Dimensions
    D = size(X_norm, 2);   % number of features = 392
    N = size(X_norm, 1);   % number of training samples
    %Initialize the variables for speed
    class_means = zeros(n_dirs, D);
    priors = zeros(n_dirs, 1);
    Sw = zeros(D, D); % Within-class scatter matrix
    
    for k = 1:n_dirs
        idx = lda_labels == k; % 1 for this direction and 0 for the others
        % Compute the mean feature values for that direction
        class_means(k,:) = mean(X_norm(idx,:), 1);
        % Number of trials for that direction
        % in this case is 1/8, but would allow to face asymmetric data for
        % training
        priors(k) = sum(idx) / N;
        % Compute the within class covariance matrix
        Xc = X_norm(idx,:) - class_means(k,:);
        Sw = Sw + Xc'*Xc;
    end

    Sw = Sw / (N - n_dirs); % Normalize the covariance matrix (for asymmetric data)
    % The covariance matrix is close to singular, so we apply shrinkage
    % regularization
    Sw = (1-shrinkage)*Sw + shrinkage*(trace(Sw)/D)*eye(D);
    Sw_inv = inv(Sw); % Invert the regularised scatter matrix
    % We store the lda model parameters
    lda_model.class_means = class_means;
    lda_model.priors = priors;
    lda_model.Sw_inv = Sw_inv;

    %% Step 2: PCA
    num_lags   = length(lags); % Number of lag windows = 6 (lags = [50,100,150,200,250,300]ms)
    % Initialize the variable for storing the pca
    pca_models = cell(n_dirs,1);
    sp_first  = training_data(1,1).spikes;
    T_first   = size(sp_first,2);
    for k = 1:n_dirs   % loop over 8 reaching directions
        sp_first  = training_data(1,k).spikes; % take the first spike train for measurements
        T_first   = size(sp_first,2); % time length of a spike train
        n_time    = length(320:bin_size:T_first); % discrete number of time bins from 320 to end of spike train
        n_samples = n_trials * n_time; % total number of samples as number of times bins times the number of trials 
        lag_k = zeros(n_samples, neuronNumber*num_lags); % feature matrix
        
        idx = 1;   % running row index into lag_k
        for n = 1:n_trials
            sp    = training_data(n,k).spikes; % extract the spike trains
            T     = size(sp,2); % time length of the spike train
            cum_s = cumsum(sp,2); % cumulative sum, as the spikes are 1 or 0

            for current_time = 320:bin_size:T % Loop over time bins
                fvec = zeros(1, neuronNumber*num_lags); % feature vector prealocated for speed
                for li = 1:num_lags % loop over lags
                    % Define the time window
                    ws = max(1, current_time - lags(li));
                    wl = current_time - ws + 1; % window length
                    % Get the cumulative spikes for that specific window,
                    % with a different code if its the first window or if
                    % there were previous windows
                    if ws==1
                        wsum = cum_s(:,current_time);
                    else
                        wsum = cum_s(:,current_time) - cum_s(:,ws-1);
                    end
                    % the feature is the mean firing rate
                    fvec((li-1)*neuronNumber+1 : li*neuronNumber) = (wsum/wl)';
                end
                lag_k(idx,:) = fvec; % store the feature
                idx = idx + 1;
            end
        end
        % mean of the features, to center the data
        mu_k = mean(lag_k, 1);
        % svd to extract the principal components
        [~,~,Wk] = svd(lag_k - mu_k, 'econ');
        % we keep only the first nc principal components
        Wk = Wk(:, 1:nc);
        % We store the pca information
        pca_models{k}.W  = Wk;
        pca_models{k}.mu = mu_k;
    end

    %% Step 3: Segmented position ridge regression
    % Define the segments
    seg_bounds = [320, 340, 360, 380, 410, 450, 500, 560, 1000];
    n_segs = length(seg_bounds) - 1;
    % Initialize the variables for storing the ridge model
    seg_ridge = cell(n_dirs, 1);

    seg_vel_ridge = cell(n_dirs, 1);
    
    for k = 1:n_dirs   % loop over 8 reaching directions
        % Retrieve this direction's PCA parameters computed in Step 2
        mu_k = pca_models{k}.mu;
        Wk   = pca_models{k}.W;
        
        seg_W = cell(n_segs, 1);
        seg_Wv = cell(n_segs, 1);

        for seg = 1:n_segs
            seg_W{seg} = []; % Initialise all segments as empty
            seg_Wv{seg} = [];
        end

        for seg = 1:n_segs % loop over the different time segments
            % Save the two boundaries of the current time segment
            t_lo = seg_bounds(seg);
            t_hi = seg_bounds(seg+1);
            Xlist = [];   % store the neural information
            Ylist = [];   % store the hand position

            Yv = [];   % velocity targets

            for n = 1:n_trials 
                % We extract the spikes and hand position
                sp    = training_data(n,k).spikes;
                hp    = training_data(n,k).handPos;
                T     = size(sp,2); % size of the spike train
                sp0   = hp(1:2, 1); % initial hand opsition
                cum_s = cumsum(sp, 2); % cumulative sum, as the spikes are 1 or 0

                hp_smooth = zeros(2, T);
                for tt = 1:T
                    % At each timestep, average hand position over a window
                    % simple moving average
                    t_s = max(1, tt - smooth_win);
                    t_e = min(T, tt + smooth_win);
                    hp_smooth(:,tt) = mean(hp(1:2, t_s:t_e), 2);
                end

                for current_time = t_lo:bin_size:min(T, t_hi) % loop over time bins inside the segment
                    if current_time >= t_lo && current_time < t_hi % check that we are inside the segment
                        fvec = zeros(1, neuronNumber*num_lags); % feature vector preallocated for speed
                        for li = 1:num_lags % loop over lags
                            % Define the time window
                            ws = max(1, current_time - lags(li));
                            wl = current_time - ws + 1;
                            % Get the cumulative spikes for that specific window,
                            % with a different code if its the first window or if
                            % there were previous windows
                            if ws==1
                                wsum = cum_s(:,current_time);
                            else
                                wsum = cum_s(:,current_time) - cum_s(:,ws-1); 
                            end
                            % the feature is the mean firing rate
                            fvec((li-1)*neuronNumber+1:li*neuronNumber) = (wsum/wl)';
                        end
                        % Store the projected neural features using PCA, with a
                        % bias term = 1
                        Xlist = [Xlist; (fvec-mu_k)*Wk, 1];
                        % store the target position as displacement
                        Ylist = [Ylist; (hp(1:2,current_time) - sp0)'];

                        prev_t = max(1, current_time - bin_size);
                        Yv = [Yv; (hp_smooth(:,current_time) - hp_smooth(:,prev_t))'];
                        % Velocity target = smoothed position change over one bin
                    end
                end
            end

            if size(Xlist, 1) > 10 % Only fit if we have more than 10 samples
                % Solve the ridge regression for that segment
                nf = size(Xlist, 2); % Number of features
                seg_W{seg} = (Xlist'*Xlist + lambda*eye(nf)) \ (Xlist'*Ylist);
                seg_Wv{seg} = (Xlist'*Xlist + lambda*eye(nf)) \ (Xlist'*Yv);
            else
                seg_W{seg} = zeros(nc+1, 2); % Not enough samples, we use zero weights as safe fallback
                seg_Wv{seg} = zeros(nc+1, 2);
            end
        end

        seg_ridge{k} = seg_W;  %save the ridge weights
        seg_vel_ridge{k} = seg_Wv;
    end


    %% Step 4: Small PCA on firing rates for Kalman observation
    % Goal: create a low-dimensional neural observation vector for the Kalman filter
    all_fr = []; % accumulate firing rate vectors from all trials, directions, timesteps
    
    for k = 1:n_dirs
        for n = 1:n_trials
            % We extract the spikes
            sp = training_data(n,k).spikes;
            T  = size(sp,2);  % size of the spike train

            for t = 320:bin_size:T % loop over time bins
                % Normalised time position in the trial: 0 at t=320, 1 at t=320+win_ramp
                % caps at 1 so window stops growing after win_ramp ms
                t_norm_win = min(1.0, (t-320) / win_ramp);
                
                % Window size linearly interpolated between win_min and
                % win_max
                win_t = round(win_min + (win_max-win_min)*t_norm_win);
                % Start of the observation window
                t_start = max(1, t - win_t + 1);
                
                fr = mean(sp(:, t_start:t), 2) * 1000/win_t; % Mean firing rate of each neuron in the window
                all_fr = [all_fr, fr]; % Append as a new column
                
            end
        end
    end
    % mean of the features, to center the data
    mu_fr = mean(all_fr, 2);
    % svd to extract the principal components
    [~,~,Vfr] = svd((all_fr - mu_fr)', 'econ');
    % we keep only the first nc_kalman principal components
    W_fr = Vfr(:, 1:nc_kalman);

    %% Step 5: Mean final position per direction
    target_pos = zeros(2, n_dirs); % preallocate the target position for each direction

    for k = 1:n_dirs % for all directions
        finals = zeros(2, n_trials); % final hand position from each training trial
        for n = 1:n_trials % for all trials
            hp = training_data(n,k).handPos; % hand position
            finals(:,n) = hp(1:2, end); % last point in the hand position
        end
        target_pos(:,k) = mean(finals, 2); % average endpoint for this direciton
    end

    %% Step 6: Kalman filter training for each direction
    % Use the kalman filter to get some memory, and avoid the independent
    % predictions of the ridge regression

    % bin_size is equivalent to the time step
    % State transition matrix (for the physic model, where position updates with velocity and velocity decays as it decelerates)
    A = [1  0  bin_size     0;
        0  1   0    bin_size;
        0  0  decay  0;
        0  0   0   decay];
    
    % Observation matrix for position (only x and y)
    H_pos_fixed = [1 0 0 0; 0 1 0 0];

    kalman_models = cell(n_dirs, 1); % one kalman model for each direction

    for k = 1:n_dirs
        % PCA parameters for this direction
        mu_k = pca_models{k}.mu;
        Wk   = pca_models{k}.W;

        % We collect state, previous_state and observation
        all_S      = [];   % states
        all_S_prev = [];   % previous states
        all_Z_n    = [];   % neural observations
        all_Ridge  = [];   % ridge prediction

        for n = 1:n_trials
            % We extract the spikes and hand position
            sp    = training_data(n,k).spikes;
            hp    = training_data(n,k).handPos;
            T     = size(sp,2); % size of the spike train
            sp0   = hp(1:2,1);     % initial hand opsition
            cum_s = cumsum(sp,2);  % cumulative sum, as the spikes are 1 or 0

            hp_smooth = zeros(2,T);
            for tt = 1:T
                % At each timestep, average hand position over a window
                % simple moving average
                t_s = max(1,tt-smooth_win);
                t_e = min(T,tt+smooth_win);
                hp_smooth(:,tt) = mean(hp(1:2,t_s:t_e),2);
            end

            times = 320:bin_size:T;

            % Loop over consecutive timestep pairs
            for ti = 2:length(times)
                % Start at ti=2 because we need both t_now and t_prev
                t_now  = times(ti);
                t_prev = times(ti-1);
                t_pp   = max(1, t_prev - bin_size); % for previous velocity, before t_prev

                % Velocity with the smooth position divided by the timestep
                vel_now  = (hp_smooth(:,t_now)  - hp_smooth(:,t_prev)) / bin_size;
                vel_prev = (hp_smooth(:,t_prev) - hp_smooth(:,t_pp)) / bin_size;
                
                % state equal to position and velocity
                s_now  = [hp(1:2,t_now);  vel_now];
                s_prev = [hp(1:2,t_prev); vel_prev];

                % Normalised time position in the trial: 0 at t=320, 1 at t=320+win_ramp
                % caps at 1 so window stops growing after win_ramp ms
                t_norm_win = min(1.0, (t_now-320)/win_ramp);
                % Window size linearly interpolated between win_min and
                % win_max
                win_t      = round(win_min + (win_max-win_min)*t_norm_win);
                % Start of the observation window
                t_win      = max(1, t_now-win_t+1);
                % Mean firing rate of each neuron in the window
                fr         = mean(sp(:,t_win:t_now),2)*1000/win_t;
                % the neural observation is the projection through the PCA
                z_n        = W_fr'*(fr-mu_fr);
                
                % Compute what the ridge would have predicted at this timestep
                fvec = zeros(1, neuronNumber*num_lags); % feature vector preallocated for speed
                for li = 1:num_lags 
                    % Define the time window
                    ws = max(1, t_now-lags(li));
                    wl = t_now-ws+1;
                    % Get the cumulative spikes for that specific window,
                    % with a different code if its the first window or if
                    % there were previous windows
                    if ws==1
                        wsum=cum_s(:,t_now);
                    else
                        wsum=cum_s(:,t_now)-cum_s(:,ws-1);
                    end
                    % the feature is the mean firing rate
                    fvec((li-1)*neuronNumber+1:li*neuronNumber) = (wsum/wl)';
                end
                pf = [(fvec-mu_k)*Wk, 1]; % Project lag features through PCA
                
                % Find which segment t_now falls in
                seg_idx = find(t_now < seg_bounds, 1) - 1;
                if isempty(seg_idx)
                    seg_idx = n_segs;
                end
                % pick the correct ridge model
                W_pos = seg_ridge{k}{seg_idx};
                dp = pf * W_pos; % displacement prediction
                ridge_pred = dp' + sp0; % position prediction

                % Save the data for kalman filter
                all_S      = [all_S,      s_now];
                all_S_prev = [all_S_prev, s_prev];
                all_Z_n    = [all_Z_n,    z_n];
                all_Ridge  = [all_Ridge,  ridge_pred];
            end
        end

        % Least square solution for z_n = H_neural*s
        H_neural = all_Z_n * all_S' / (all_S*all_S' + 1e-4*eye(4));
        
        % H is the combination of  the position and the ridge prediction
        H = [H_pos_fixed; H_neural];


        % Estimate noise covariances from residuals(diff between true next
        % state and what the model predicted)
        res_A   = all_S - A*all_S_prev;
        % Neural observation residuals (diff between neural PCA and what
        % H_neural predicts)
        res_n   = all_Z_n - H_neural*all_S;

        % Ridge prediction residual (diff between ridge pos prediction and
        % true position)
        res_pos = all_Ridge - H_pos_fixed*all_S;

        % Covariance matrices
        Q = (res_A*res_A') / size(res_A,2);
        % Neural obs noise covariance
        R_neural  = (res_n*res_n') / size(res_n,2);
        % Ridge pos noise covariance
        R_pos_est = (res_pos*res_pos') / size(res_pos,2);

        % We scale the covariance matrices preventing it from being singular
        Q = Q * q_scale + 1e-4*eye(4); % trust physics model more
        R_neural = R_neural * r_scale + 1e-4*eye(nc_kalman); % trust neural PCA less
        R_pos_mat = R_pos_est * r_pos + 1e-4*eye(2); % trust ridge more
        
        % Assemble the R matrix
        R = [R_pos_mat,          zeros(2,nc_kalman);
            zeros(nc_kalman,2), R_neural];
        
        % Store the kalman model
        kalman_models{k}.H = H;
        kalman_models{k}.Q = Q;
        kalman_models{k}.R = R;

        kalman_models{k}.P = diag([500 500 50 50]); % Initial state uncertainty covariance
    end

    %% Step 7: Save all parameters
    modelParameters.lda_model     = lda_model;
    modelParameters.bin_edges     = bin_edges;
    modelParameters.n_bins        = n_bins;
    modelParameters.mu_s          = mu_s;
    modelParameters.sig_s         = sig_s;
    modelParameters.pca_models    = pca_models;
    modelParameters.seg_ridge     = seg_ridge;
    modelParameters.seg_vel_ridge = seg_vel_ridge;
    modelParameters.seg_bounds    = seg_bounds;
    modelParameters.kalman_models = kalman_models;
    modelParameters.A             = A;
    modelParameters.lags          = lags;
    modelParameters.bin_size      = bin_size;
    modelParameters.win_min       = win_min;
    modelParameters.win_max       = win_max;
    modelParameters.win_ramp      = win_ramp;
    modelParameters.mu_fr         = mu_fr;
    modelParameters.W_fr          = W_fr;
    modelParameters.nc_kalman     = nc_kalman;
    modelParameters.n_segs        = n_segs;
    modelParameters.target_pos    = target_pos;
end