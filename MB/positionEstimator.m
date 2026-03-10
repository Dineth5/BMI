function [x, y] = positionEstimator(test_data, modelParameters)
    spikes       = test_data.spikes;
    svm_models   = modelParameters.svm_models;
    mu_s         = modelParameters.mu_s;
    sig_s        = modelParameters.sig_s;
    pca_models   = modelParameters.pca_models;
    ridge_models = modelParameters.ridge_models;
    lags         = modelParameters.lags;
    t            = size(spikes,2);

    % Stage 1: SVM v2 classify
    raw_r  = (sum(spikes,2)/t)';
    raw_rn = (raw_r - mu_s) ./ sig_s;
    scores = zeros(1,8);
    for k = 1:8
        scores(k) = raw_rn*svm_models{k}.w + svm_models{k}.b;
    end
    [~,dir] = max(scores);

    % Stage 2: Direction-specific PCA then Ridge
    Wk   = pca_models{dir}.W;
    mu_k = pca_models{dir}.mu;
    fvec = [];
    for li = 1:length(lags)
        ws = max(1,t-lags(li));
        fvec = [fvec, mean(spikes(:,ws:t),2)'];
    end
    pf = [(fvec-mu_k)*Wk, 1];
    W  = ridge_models{dir};
    dp = pf*W;
    x  = dp(1)+test_data.startHandPos(1);
    y  = dp(2)+test_data.startHandPos(2);


    % Clipping

    % Define the physical workspace boundary (thresholding for artifact reduction)
    max_reach = 97; % Adjust this value based on your specific dataset limits

    % Calculate raw coordinates
    raw_x = dp(1) + test_data.startHandPos(1);
    raw_y = dp(2) + test_data.startHandPos(2);

    % Clip values to be within [-max_reach, max_reach]
    x = max(min(raw_x, max_reach), -max_reach);
    y = max(min(raw_y, max_reach), -max_reach);
end
