function out = detect_noise_type(imgPath)
    try
        img = imread(imgPath);
        if size(img, 3) == 3, img = rgb2gray(img); end
        img = im2double(img);
        
        % Extract noise residuals
        noiseResidual = img - imfilter(img, fspecial('gaussian', [5 5], 1.0), 'replicate');
        
        % STEP 1: Intensity-dependence (var vs mean)
        patchSize = 10;
        [imgH, imgW] = size(img);
        localMeans = []; localVars = [];
        for i = 1:patchSize:(imgH - patchSize + 1)
            for j = 1:patchSize:(imgW - patchSize + 1)
                patch = img(i:i+patchSize-1, j:j+patchSize-1);
                localMeans(end+1) = mean(patch(:));
                localVars(end+1) = var(patch(:));
            end
        end
        valid = (localMeans > 0.05) & isfinite(localMeans) & isfinite(localVars);
        localMeans = localMeans(valid); localVars = localVars(valid);
        
        isAdditive = false; isPoisson = false; isSpeckle = false;
        if numel(localMeans) >= 10
            p_lin = polyfit(localMeans, localVars, 1);
            r2_lin = 1 - sum((localVars - polyval(p_lin, localMeans)).^2) / (sum((localVars - mean(localVars)).^2) + eps);
            p_quad = polyfit(localMeans, localVars, 2);
            r2_quad = 1 - sum((localVars - polyval(p_quad, localMeans)).^2) / (sum((localVars - mean(localVars)).^2) + eps);
            varCoeff = var(localVars) / (mean(localVars)^2 + eps);
            
            isAdditive = (r2_lin < 0.3) && (varCoeff < 0.5);
            isPoisson = (r2_lin > 0.4) && (p_lin(1) > 0.5) && (p_lin(1) < 1.8) && (abs(p_lin(2)) < 0.01);
            isSpeckle = (r2_quad > 0.5) && (r2_quad > r2_lin + 0.15) && (p_quad(1) > 0.1);
        end
        
        % STEP 2: Histogram shape
        [histCounts, ~] = histcounts(noiseResidual(:), 100);
        histCounts = histCounts / sum(histCounts);
        [maxPeak, peakIdx] = max(histCounts);
        hasCentralPeak = ismember(peakIdx, (numel(histCounts)*0.4):(numel(histCounts)*0.6));
        isFlat = (std(histCounts) / (mean(histCounts) + eps)) < 0.5;
        extremeBins = 5;
        bimodalExtreme = (sum(histCounts(1:extremeBins)) + sum(histCounts(end-extremeBins+1:end))) / ...
                         (sum(histCounts(extremeBins+1:end-extremeBins)) + eps);
        isBimodal = bimodalExtreme > 0.3;
        
        % STEP 3: Statistical moments
        kurt = kurtosis(noiseResidual(:));
        skew = skewness(noiseResidual(:));
        isGaussianKurt = abs(kurt - 3) < 1.0;
        isUniformKurt = (kurt < -0.5) && (abs(kurt - (-1.2)) < 1.0);
        isSymmetric = abs(skew) < 0.4;
        noiseVar = var(noiseResidual(:));
        
        % STEP 4: Variance-to-mean ratio
        globalMean = mean(img(:));
        globalVar = var(img(:));
        varMeanRatio = globalVar / (globalMean + eps);
        isPoissonRatio = (varMeanRatio > 0.01) && (varMeanRatio < 0.15);
        isSpeckleRatio = false;
        if globalMean > 0.05
            isSpeckleRatio = (globalVar / (globalMean^2) > 0.01) && (globalVar / (globalMean^2) < 0.5);
        end
        
        % Specific tests
        saltPepperScore = sum(img(:) < 0.05) / numel(img) + sum(img(:) > 0.95) / numel(img);
        impulseRatio = sum(abs(img - medfilt2(img, [3 3])) > 0.3, 'all') / numel(img);
        
        isSaltPepper = (saltPepperScore > 0.01) && (impulseRatio > 0.01) && isBimodal;
        isGaussian = isAdditive && isSymmetric && isGaussianKurt && hasCentralPeak && (noiseVar > 0.0001);
        isUniform = isAdditive && isSymmetric && isUniformKurt && isFlat;
        isPoisson = (isPoisson || isPoissonRatio) && ~isAdditive && ~isSaltPepper;
        isSpeckle = (isSpeckle || isSpeckleRatio) && ~isAdditive && ~isSaltPepper && ~isPoisson;
        
        % Final decision
        if isSaltPepper, out = 'salt_pepper';
        elseif isUniform, out = 'uniform';
        elseif isSpeckle, out = 'speckle';
        elseif isPoisson, out = 'poisson';
        elseif isGaussian, out = 'gaussian';
        else, out = 'none';
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        out = 'error';
    end
end
