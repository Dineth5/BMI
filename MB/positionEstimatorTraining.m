function [modelParameters] = positionEstimatorTraining(training_data)
    [n_trials, n_dirs] = size(training_data);
    neuronNumber = size(training_data(1,1).spikes,1);
    bin_size = 20; 
    lambda = 0.1;
    lags = [50,100,150,200,250,300];
    nc = 550;
    
    % --- BREAKTHROUGH 1: TEMPORAL SVM BEAMS ---
    % Dividing the "Plan" phase (1-300ms) into three 100ms features per neuron
    svm_features = zeros(n_trials * n_dirs, neuronNumber * 3);
    svm_labels = zeros(n_trials * n_dirs, 1);
    
    row = 1;
    for k = 1:n_dirs
        for n = 1:n_trials
            sp = training_data(n,k).spikes;
            feat = [mean(sp(:,1:100),2)', mean(sp(:,101:200),2)', mean(sp(:,201:300),2)'];
            svm_features(row, :) = feat;
            svm_labels(row) = k;
            row = row + 1;
        end
    end
    
    % Normalize SVM Features
    mu_s = mean(svm_features, 1);
    sig_s = std(svm_features, 0, 1); sig_s(sig_s == 0) = 1;
    X_svm = (svm_features - mu_s) ./ sig_s;

    % Train One-vs-All SVM (Soft Margin)
    C = 0.08; lr = 0.0001; epochs = 250;
    svm_models = cell(n_dirs, 1);
    for k = 1:n_dirs
        y = -ones(size(svm_labels));
        y(svm_labels == k) = 1;
        w = zeros(size(X_svm, 2), 1); b = 0;
        Np = sum(y==1); Nn = sum(y==-1);
        Cp = C*(length(y)/(2*Np)); Cn = C*(length(y)/(2*Nn));
        
        for ep = 1:epochs
            p = randperm(length(y));
            lr_e = lr/(1+ep*0.01);
            for i = p
                Ci = (y(i)==1)*Cp + (y(i)==-1)*Cn;
                if y(i)*(X_svm(i,:)*w + b) < 1
                    w = (1-lr_e)*w + (lr_e*Ci*y(i))*X_svm(i,:)';
                    b = b + lr_e*Ci*y(i);
                else
                    w = (1-lr_e)*w;
                end
            end
        end
        svm_models{k}.w = w; svm_models{k}.b = b;
    end

    % --- BREAKTHROUGH 2: VELOCITY-BASED RIDGE (PCA Space) ---
    pca_models = cell(n_dirs,1);
    ridge_models = cell(n_dirs,1);
    
    for k = 1:n_dirs
        % Calculate sample size for this direction
        s_count = 0;
        for n = 1:n_trials
            s_count = s_count + length(320:bin_size:size(training_data(n,k).spikes,2));
        end
        
        lag_k = zeros(s_count, neuronNumber * length(lags));
        Yk = zeros(s_count, 2); % To store Velocity (Delta Pos)
        
        idx = 1;
        for n = 1:n_trials
            sp = training_data(n,k).spikes;
            hp = training_data(n,k).handPos(1:2,:);
            cumulative_spikes = cumsum(sp,2);
            
            for t_curr = 320:bin_size:size(sp,2)
                fvec = [];
                for lag_val = lags
                    ws = max(1, t_curr - lag_val);
                    mean_fr = (cumulative_spikes(:,t_curr) - cumulative_spikes(:,ws)) / (t_curr - ws + 1);
                    fvec = [fvec, mean_fr'];
                end
                lag_k(idx,:) = fvec;
                
                % Target: Velocity (Current position minus position 20ms ago)
                Yk(idx,:) = (hp(:,t_curr) - hp(:,t_curr - bin_size))';
                idx = idx + 1;
            end
        end
        
        mu_k = mean(lag_k,1);
        [~,~,Wk] = svd(lag_k-mu_k,'econ');
        Wk = Wk(:,1:nc);
        pca_models{k}.W = Wk;
        pca_models{k}.mu = mu_k;
        
        % Solve Ridge for Velocity
        Xk = [(lag_k - mu_k)*Wk, ones(size(lag_k,1), 1)];
        ridge_models{k} = (Xk'*Xk + lambda*eye(nc+1)) \ (Xk'*Yk);
    end
    
    modelParameters.svm_models = svm_models;
    modelParameters.mu_s = mu_s;
    modelParameters.sig_s = sig_s;
    modelParameters.pca_models = pca_models;
    modelParameters.ridge_models = ridge_models;
    modelParameters.lags = lags;
end