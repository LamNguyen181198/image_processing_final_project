function features = extract_features(imgPath)
    % EXTRACT_FEATURES - Extract numerical features from an image for ML classification
    %
    % Usage:
    %   features = extract_features('path/to/image.png')
    %
    % Returns:
    %   features - struct with all computed feature values
    
    try
        img = imread(imgPath);
        if size(img, 3) == 3, img = rgb2gray(img); end
        img = im2double(img);
        
        % Extract noise residuals
        noiseResidual = img - imfilter(img, fspecial('gaussian', [5 5], 1.0), 'replicate');
        
        %% FEATURE 1-3: Variance-Mean Relationship
        patchSize = 10;
        [imgH, imgW] = size(img);
        localMeans = [];
        localVars = [];
        
        for i = 1:patchSize:(imgH - patchSize + 1)
            for j = 1:patchSize:(imgW - patchSize + 1)
                patch = img(i:i+patchSize-1, j:j+patchSize-1);
                localMeans(end+1) = mean(patch(:));
                localVars(end+1) = var(patch(:));
            end
        end
        
        valid = (localMeans > 0.05) & isfinite(localMeans) & isfinite(localVars);
        localMeans = localMeans(valid);
        localVars = localVars(valid);
        
        % Default values
        features.r2_linear = 0;
        features.r2_quadratic = 0;
        features.variance_coefficient = 0;
        features.linear_slope = 0;
        features.linear_intercept = 0;
        features.quadratic_a = 0;
        
        if numel(localMeans) >= 10
            p_lin = polyfit(localMeans, localVars, 1);
            r2_lin = 1 - sum((localVars - polyval(p_lin, localMeans)).^2) / (sum((localVars - mean(localVars)).^2) + eps);
            p_quad = polyfit(localMeans, localVars, 2);
            r2_quad = 1 - sum((localVars - polyval(p_quad, localMeans)).^2) / (sum((localVars - mean(localVars)).^2) + eps);
            varCoeff = var(localVars) / (mean(localVars)^2 + eps);
            
            features.r2_linear = r2_lin;
            features.r2_quadratic = r2_quad;
            features.variance_coefficient = varCoeff;
            features.linear_slope = p_lin(1);
            features.linear_intercept = p_lin(2);
            features.quadratic_a = p_quad(1);
        end
        
        %% FEATURE 4-6: Histogram Shape
        [histCounts, ~] = histcounts(noiseResidual(:), 100);
        histCounts = histCounts / sum(histCounts);
        [maxPeak, peakIdx] = max(histCounts);
        
        % Central peak indicator
        hasCentralPeak = ismember(peakIdx, (numel(histCounts)*0.4):(numel(histCounts)*0.6));
        features.has_central_peak = double(hasCentralPeak);
        
        % Flatness
        features.histogram_flatness = std(histCounts) / (mean(histCounts) + eps);
        
        % Bimodal extremes
        extremeBins = 5;
        bimodalExtreme = (sum(histCounts(1:extremeBins)) + sum(histCounts(end-extremeBins+1:end))) / ...
                         (sum(histCounts(extremeBins+1:end-extremeBins)) + eps);
        features.bimodal_extreme_ratio = bimodalExtreme;
        
        %% FEATURE 7-9: Statistical Moments
        features.kurtosis = kurtosis(noiseResidual(:));
        features.skewness = skewness(noiseResidual(:));
        features.noise_variance = var(noiseResidual(:));
        
        %% FEATURE 10-11: Global Variance-Mean Ratios
        globalMean = mean(img(:));
        globalVar = var(img(:));
        features.var_mean_ratio = globalVar / (globalMean + eps);
        features.var_mean_squared_ratio = globalVar / (globalMean^2 + eps);
        
        %% FEATURE 12-14: Impulse/Salt-Pepper Indicators
        features.salt_pepper_score = sum(img(:) < 0.05) / numel(img) + sum(img(:) > 0.95) / numel(img);
        medFiltered = medfilt2(img, [3 3], 'symmetric');
        features.impulse_ratio = sum(abs(img(:) - medFiltered(:)) > 0.3) / numel(img);
        features.median_diff_variance = var(img(:) - medFiltered(:));
        
        %% FEATURE 15-17: Frequency Domain Features
        % DCT blockiness (for JPEG)
        blockSize = 8;
        dctBlock = dct2(img(1:min(blockSize, size(img,1)), 1:min(blockSize, size(img,2))));
        features.dct_dc_energy = abs(dctBlock(1,1));
        features.dct_ac_energy = sum(abs(dctBlock(:))) - abs(dctBlock(1,1));
        
        % Edge strength variation (JPEG creates blocking)
        [Gx, Gy] = gradient(img);
        edgeMag = sqrt(Gx.^2 + Gy.^2);
        features.edge_variance = var(edgeMag(:));
        
        %% FEATURE 18-20: Additional discriminative features
        % Peak signal value (for Poisson detection)
        features.peak_intensity = max(img(:));
        features.min_intensity = min(img(:));
        
        % Entropy
        features.entropy = entropy(img);
        
    catch ME
        fprintf('ERROR in extract_features: %s\n', ME.message);
        % Return NaN features on error
        features = struct('r2_linear', NaN, 'r2_quadratic', NaN, ...
                         'variance_coefficient', NaN, 'linear_slope', NaN, ...
                         'linear_intercept', NaN, 'quadratic_a', NaN, ...
                         'has_central_peak', NaN, 'histogram_flatness', NaN, ...
                         'bimodal_extreme_ratio', NaN, 'kurtosis', NaN, ...
                         'skewness', NaN, 'noise_variance', NaN, ...
                         'var_mean_ratio', NaN, 'var_mean_squared_ratio', NaN, ...
                         'salt_pepper_score', NaN, 'impulse_ratio', NaN, ...
                         'median_diff_variance', NaN, 'dct_dc_energy', NaN, ...
                         'dct_ac_energy', NaN, 'edge_variance', NaN, ...
                         'peak_intensity', NaN, 'min_intensity', NaN, 'entropy', NaN);
    end
end
