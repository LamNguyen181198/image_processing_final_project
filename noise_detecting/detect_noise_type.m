function out = detect_noise_type(imgPath)
    % Initialize output to prevent "not assigned" error
    out = 'none';
    
    try
        % -----------------------------------------------------------
        % READ IMAGE
        % -----------------------------------------------------------
        img = imread(imgPath);

        % Convert to grayscale double
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        img = im2double(img);

        % ===========================================================
        % THEORY-BASED NOISE DETECTION
        % ===========================================================
        
        % Extract noise by subtracting smoothed version
        smoothed = imgaussfilt(img, 1.5);
        noise = img - smoothed;
        
        % ===========================================================
        % 1. SALT & PEPPER DETECTION (Impulse Noise)
        % ===========================================================
        % Characteristics: Random black (0) and white (1) pixels
        
        % Count extreme pixels
        blackPixels = sum(img(:) < 0.05);
        whitePixels = sum(img(:) > 0.95);
        extremeRatio = (blackPixels + whitePixels) / numel(img);
        
        % Median filter response (median kills impulses)
        medFiltered = medfilt2(img, [3 3]);
        medDiff = abs(img - medFiltered);
        impulsePixels = sum(medDiff(:) > 0.3);
        impulseRatio = impulsePixels / numel(img);
        
        % S&P detection (requires BOTH extreme pixels AND impulse response)
        isSaltPepper = (extremeRatio > 0.01) && (impulseRatio > 0.01);
        
        % ===========================================================
        % 2. GAUSSIAN NOISE DETECTION (Additive White Gaussian Noise)
        % ===========================================================
        % Characteristics: Normal distribution, bell-shaped histogram
        
        % Histogram of noise residual
        [counts, edges] = histcounts(noise(:), 50);
        counts = counts / sum(counts);
        
        % Find peak
        [peakVal, peakIdx] = max(counts);
        
        % Check if peak is centered
        isCentered = (peakIdx > 20) && (peakIdx < 30);
        
        % Check bell shape (peak >> edges)
        edgeVal = mean([counts(1:3), counts(end-2:end)]);
        peakRatio = peakVal / (edgeVal + 0.001);
        isBellShaped = peakRatio > 3.0;
        
        % Statistical moments
        noiseKurt = kurtosis(noise(:));
        noiseSkew = abs(skewness(noise(:)));
        
        % Gaussian has kurtosis ≈ 3, skewness ≈ 0
        isGaussianMoments = (abs(noiseKurt - 3) < 1.0) && (noiseSkew < 0.5);
        
        % Combined Gaussian test
        isGaussian = isCentered && isBellShaped && isGaussianMoments && ~isSaltPepper;
        
        % ===========================================================
        % 3. SPECKLE NOISE DETECTION (Multiplicative Noise)
        % ===========================================================
        % Characteristics: 
        % - Multiplicative: I_noisy = I_clean * (1 + n)
        % - Variance proportional to signal intensity
        % - NOT present in dark regions
        
        % Divide image into patches and analyze local statistics
        patchSize = 16;
        [h, w] = size(img);
        
        localMeans = [];
        localStds = [];
        
        for i = 1:patchSize:h-patchSize+1
            for j = 1:patchSize:w-patchSize+1
                patch = img(i:i+patchSize-1, j:j+patchSize-1);
                patchMean = mean(patch(:));
                patchStd = std(patch(:));
                
                % Only consider bright patches (speckle doesn't affect dark areas)
                if patchMean > 0.2
                    localMeans = [localMeans; patchMean];
                    localStds = [localStds; patchStd];
                end
            end
        end
        
        isSpeckle = false;
        if length(localMeans) >= 5
            % For multiplicative noise: std ∝ mean (strong linear relationship)
            correlation = corrcoef(localMeans, localStds);
            stdMeanCorr = abs(correlation(1,2));
            
            % Linear fit
            p = polyfit(localMeans, localStds, 1);
            slope = p(1);
            
            % Coefficient of Variation should be roughly constant
            CV = localStds ./ (localMeans + eps);
            meanCV = mean(CV);
            stdCV = std(CV);
            cvConsistency = stdCV / (meanCV + eps);
            
            % Speckle criteria:
            % 1. Strong correlation (> 0.7) between std and mean
            % 2. Positive slope (std increases with intensity)
            % 3. Consistent CV across patches
            hasStrongCorr = stdMeanCorr > 0.7;
            hasPositiveSlope = slope > 0.1;
            hasConsistentCV = cvConsistency < 0.5 && meanCV > 0.15 && meanCV < 0.40;
            
            isSpeckle = hasStrongCorr && hasPositiveSlope && hasConsistentCV && ~isSaltPepper;
        end
        
        % ===========================================================
        % 4. UNIFORM NOISE DETECTION (Uniform Distribution)
        % ===========================================================
        % Characteristics: Flat rectangular distribution
        
        % Histogram flatness
        histStd = std(counts);
        isFlat = histStd < 0.012;
        
        % Kurtosis (uniform has kurtosis ≈ 1.8, much lower than Gaussian's 3)
        isLowKurt = noiseKurt < 2.2;
        
        % Wide plateau in histogram
        threshold = max(counts) * 0.4;
        plateauBins = sum(counts > threshold);
        hasWidePlateau = plateauBins > 12;
        
        % Combined uniform test
        isUniform = isFlat && isLowKurt && hasWidePlateau && ~isSaltPepper && ~isSpeckle;
        
        % ===========================================================
        % FINAL DECISION
        % ===========================================================
        % Priority: Salt&Pepper > Speckle > Gaussian > Uniform
        if isSaltPepper
            out = 'salt_pepper';
        elseif isSpeckle
            out = 'speckle';
        elseif isGaussian
            out = 'gaussian';
        elseif isUniform
            out = 'uniform';
        else
            out = 'none';
        end

    catch ME
        disp("ERROR in detect_noise_type:");
        disp(getReport(ME));
        out = 'none';
    end
end
