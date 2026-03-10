function [modelParameters] = positionEstimatorTraining(training_data)
    [n_trials, n_dirs] = size(training_data); % Get the dimensions of the training data
    neuronNumber = size(training_data(1,1).spikes,1); % Get the number of neurons
    bin_size = 20; 
    lambda = 0.1;
    lags = [50,100,150,200,250,300];
    nc = 550;
    % Parameters for SVM
    C = 0.08;
    lr = 0.0001; % learning rate
    epochs = 250; % times the learning will be done
    

    % Step 1: Raw firing rates for SVM
    firingRates = zeros(n_trials,neuronNumber,n_dirs);
    for k = 1:n_dirs % for all directions and trials, store the firint rates
            for i = 1:n_trials
                firingRates(i,:,k) = mean(training_data(i,k).spikes,2);
            end
    end
    
    % Step 2: SVM one vs all (zscore, lr=0.0001, shuffle each epoch)
    X_svm = reshape(permute(firingRates,[1,3,2]), n_trials*n_dirs, neuronNumber);
    mu_s  = mean(X_svm,1);
    sig_s = std(X_svm,0,1); sig_s(sig_s==0)=1;
    X_svm = (X_svm - mu_s) ./ sig_s; % normalize the firing rates
    N_total = n_trials*n_dirs;
    svm_models = cell(n_dirs,1);
    for k = 1:n_dirs % we compute the model for all directions in the one vs all
        y  = -ones(N_total,1);
        y((k-1)*n_trials+1:k*n_trials) = 1; % Only one direction will have label=1
        % Initialize the w and b
        w  = zeros(neuronNumber,1); 
        b = 0;
        % Count the number of each class
        Np = sum(y==1); 
        Nn = sum(y==-1);
        % Weight C by the number of each class to reduce imbalance
        Cp = C*(N_total/(2*Np)); 
        Cn = C*(N_total/(2*Nn));
 
        for ep = 1:epochs
            lr_e = lr/(1+ep*0.01); % The learning rate in each epoch will be reduced, to change the model less as it learns
            perm = randperm(N_total); % we permutate the training data to avoid bias
            for idx = 1:N_total
                i  = perm(idx);
                % Select the current normalized firing rate and its label
                xi = X_svm(i,:)';
                yi = y(i);
                %mg = yi*(w'*xi+b); % Compute the margin
                % Select the C value depending on the y label
                if yi==1
                    Ci=Cp; 
                else
                    Ci=Cn; 
                end
                % If margin is more than one, we only reduce the weights
                % (to avoid going to infinite weigths)
                % If margin is less than one, we update the w and b to get
                % away from the datapoint
                %if mg >=1
                if yi*(w'*xi+b) >=1
                    %w = w-lr_e*w;
                    w = (1-lr_e)*w;
                    
                else
                    %w = w-lr_e*(w-Ci*yi*xi);
                    w = (1-lr_e)*w+(lr_e*Ci*yi)*xi;
                    b = b+lr_e*Ci*yi;
                end
            end
        end
        % We save the model parameters
        svm_models{k}.w = w;
        svm_models{k}.b = b;
    end

    % Step 3: Per-direction PCA on lag features (164 components)
    pca_models   = cell(n_dirs,1);
    ridge_models = cell(n_dirs,1);

    sp = training_data(1,1).spikes;
    T  = size(sp,2);
    n_time = length(320:bin_size:T);
    n_samples = n_trials * n_time;
    n_lags = length(lags);
    n_features = neuronNumber * n_lags;
    lag_k = zeros(n_samples, n_features);
    
    num_lags = length(lags);

    for k = 1:n_dirs
        idx = 1;

        for n = 1:n_trials
            sp = training_data(n,k).spikes;
            T  = size(sp,2);

            % cumulative spike counts over time
            cumulative_spikes = cumsum(sp,2);

            fvec = zeros(1, neuronNumber*num_lags);

            for current_time = 320:bin_size:T

                for lag_index = 1:num_lags

                    lag_value = lags(lag_index);

                    window_start_time = max(1, current_time - lag_value);
                    window_length = current_time - window_start_time + 1;

                    if window_start_time == 1
                        window_spike_sum = cumulative_spikes(:,current_time);
                    else
                        window_spike_sum = cumulative_spikes(:,current_time) ...
                            - cumulative_spikes(:,window_start_time-1);
                    end

                    mean_firing_rate = window_spike_sum / window_length;

                    feature_idx = (lag_index-1)*neuronNumber + 1 : lag_index*neuronNumber;

                    fvec(feature_idx) = mean_firing_rate';

                end

                lag_k(idx,:) = fvec;
                idx = idx + 1;

            end
        end
        mu_k = mean(lag_k,1);
        [~,~,Wk] = svd(lag_k-mu_k,'econ');
        Wk = Wk(:,1:nc);
        pca_models{k}.W  = Wk;
        pca_models{k}.mu = mu_k;

        % Ridge in PCA space
        Xk    = zeros(n_samples, nc+1);
        Yk    = zeros(n_samples, 2);
        row = 1;
        
        %%
        for n = 1:n_trials

            sp  = training_data(n,k).spikes;
            hp  = training_data(n,k).handPos;
            T   = size(sp,2);
            sp0 = hp(1:2,1);

            cumulative_spikes = cumsum(sp,2);

            fvec = zeros(1, neuronNumber*num_lags);

            for current_time = 320:bin_size:T
                for lag_index = 1:num_lags

                    lag_value = lags(lag_index);

                    window_start_time = max(1, current_time - lag_value);
                    window_length = current_time - window_start_time + 1;

                    if window_start_time == 1
                        window_spike_sum = cumulative_spikes(:,current_time);
                    else
                        window_spike_sum = cumulative_spikes(:,current_time) ...
                            - cumulative_spikes(:,window_start_time-1);
                    end

                    mean_firing_rate = window_spike_sum / window_length;

                    feature_idx = (lag_index-1)*neuronNumber + 1 : lag_index*neuronNumber;

                    fvec(feature_idx) = mean_firing_rate';
                end

                pf = [(fvec-mu_k)*Wk, 1];

                dv = hp(1:2,current_time) - sp0;

                Xk(row,:) = pf;
                Yk(row,:) = dv';

                row = row + 1;
            end
        end

        nf  = size(Xk,2);
        XtX = Xk'*Xk; XtY = Xk'*Yk;
        ridge_models{k} = (XtX+lambda*eye(nf))\XtY;
    end
    % We save all the model parameters
    modelParameters.svm_models   = svm_models;
    modelParameters.mu_s         = mu_s;
    modelParameters.sig_s        = sig_s;
    modelParameters.pca_models   = pca_models;
    modelParameters.ridge_models = ridge_models;
    modelParameters.lags         = lags;
end
