function out = detect_noise_type(imgPath)
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

        [imgH, imgW] = size(img);

        % ===========================================================
        % 1. GAUSSIAN NOISE DETECTION
        % ===========================================================
        % High-frequency extraction
        h = fspecial('gaussian', [5 5], 1.0);
        low = imfilter(img, h, 'replicate');
        high = img - low;

        hfVar   = var(high(:));
        hfKurt  = kurtosis(high(:));
        hfSkew  = skewness(high(:));

        isHighVar      = hfVar > 0.0005;
        isGaussKurt    = abs(hfKurt - 3) < 1.2;
        isGaussSkew    = abs(hfSkew) < 0.7;

        isGaussian = isHighVar && (isGaussKurt || isGaussSkew);

        % ===========================================================
        % 2. JPEG ARTIFACT DETECTION
        % ===========================================================

        rows = 8:8:size(img,1)-1;
        cols = 8:8:size(img,2)-1;

        bbd_h = mean(abs(img(rows,:) - img(rows+1,:)), 'all');
        bbd_v = mean(abs(img(:,cols) - img(:,cols+1)), 'all');
        blockBoundary = (bbd_h + bbd_v) / 2;

        % internal edges
        internal = mean(abs(diff(img,1,1)), 'all') + ...
                   mean(abs(diff(img,1,2)), 'all');
        internal = internal/2 + 1e-8;

        blockiness_ratio = blockBoundary / internal;

        % ---------- DCT quantization detection ----------
        blockSize = 8;
        fun = @(b) dct2(b.data);

        dctImg = blockproc(img, [blockSize blockSize], fun);

        % Extract AC high-frequency region
        HF_AC = abs(dctImg(5:end,5:end));

        % Look for spikes in the histogram
        edges = linspace(0, max(HF_AC(:)), 80);
        hst = histcounts(HF_AC(:), edges);

        % Peakiness metric
        peakiness = max(hst) / mean(hst);

        % Mean AC magnitude (avoid spurious peakiness when coefficients ~ 0)
        mean_HF_AC = mean(HF_AC(:));

        % Empirical thresholds (tuned)
        % Require a larger mean AC magnitude to avoid
        % misclassifying low-energy Poisson images as JPEG.
        isJPEG_DCT = (peakiness > 3.2) && (mean_HF_AC > 5e-3);  % stricter guard
        isJPEG_Block = (blockiness_ratio > 1.25) && (blockBoundary > 0.010);

        isJPEG = isJPEG_DCT || isJPEG_Block;

        jpeg_peakiness = peakiness; jpeg_blockiness_ratio = blockiness_ratio; 
        jpeg_blockBoundary = blockBoundary; jpeg_mean_HF_AC = mean_HF_AC;

        % ===========================================================
        % 3. SALT & PEPPER DETECTION
        % ===========================================================

        medImg = medfilt2(img, [3 3]);

        % Moderate
        impulseMask = abs(img - medImg) > 0.25;
        impulseRatio = sum(impulseMask(:)) / numel(img);

        % 2) HARD EXTREME PIXELS (very specific to S&P)
        extremeRatio = (sum(img(:) <= 0.02) + sum(img(:) >= 0.98)) / numel(img);

        % 3) IMPULSE PEAK SCORE
        % If S&P exists, median filtering reduces noise strongly → big residual drop
        medResidual = mean(abs(img(:) - medImg(:)));

        % Residual of the high-frequency
        hf = img - imgaussfilt(img, 1);  
        medResidualHF = mean(abs(hf(:))); 

        % Ratio is strong for impulse noise
        impulsePeak = medResidualHF / (medResidual + 1e-8);

        % ---------- Decision combining all metrics ----------
        cond1 = impulseRatio > 0.008;      % ≥ 0.8% impulses
        cond2 = extremeRatio > 0.003;      % ≥ 0.3% extreme pixels
        cond3 = impulsePeak > 2.0;         % big HF vs median difference

        % Require at least TWO conditions to classify as S&P (To make sure this doesn't get misclasify)
        isSaltPepper = (cond1 + cond2 + cond3) >= 2;

        % Prevent JPEG or Gaussian misfiring when impulses exist (Doubly sure)
        if isSaltPepper
            isGaussian = false;
            isJPEG = false;
        end
        
        % ===========================================================
        % 4. POISSON (SHOT) NOISE DETECTION
        % ===========================================================
        patchSize = 16;
        funMean = @(b) mean(b.data(:));
        funVar  = @(b) var(b.data(:));

        % Compute block-wise mean and variance (each block filled with scalar)
        meanBlocks = blockproc(img, [patchSize patchSize], funMean);
        varBlocks  = blockproc(img, [patchSize patchSize], funVar);

        % Sample one value per block (top-left of each block)
        means = meanBlocks(1:patchSize:end, 1:patchSize:end);
        vars  = varBlocks(1:patchSize:end, 1:patchSize:end);

        means = means(:);
        vars  = vars(:);

        % Filter out invalid or very dark patches (where variance is tiny)
        valid = isfinite(means) & isfinite(vars);
        valid = valid & (means > 0.005);  % ignore very dark patches
        means = means(valid);
        vars  = vars(valid);

        isPoisson = false;
        if numel(means) >= 8
            p = polyfit(means, vars, 1);
            slope = p(1);
            intercept = p(2);

            yfit = polyval(p, means);
            ssres = sum((vars - yfit).^2);
            sstot = sum((vars - mean(vars)).^2) + eps;
            r2 = 1 - ssres/sstot;

            vmr = mean(vars ./ (means + 1e-8));

            % Robust checks (scale-dependent):
            slopePos = (slope > 1e-6);
            slopeReasonable = (slope < 1e2); % prevent pathological slopes

            meanVar = mean(vars) + 1e-12;
            interceptRel = abs(intercept) / meanVar; % intercept relative to mean variance
            interceptSmall = (interceptRel < 0.5); % intercept less than half of mean variance

            residStd = std(vars - yfit);
            residRel = residStd / (meanVar + 1e-12);

            r2Good = (r2 > 0.18);
            residReasonable = (residRel < 2.0);

            isPoisson = slopePos && slopeReasonable && interceptSmall && r2Good && residReasonable;

            est_peak = 1 / (slope + 1e-12);
        else
            est_peak = NaN;
        end

        % If Poisson is confidently detected, avoid marking as Gaussian/JPEG
        if isPoisson
            isGaussian = false;
            isJPEG = false;
        end

        % Strong override: if high-frequency kurtosis is very large this
        % typically indicates heavy-tailed shot noise — prefer Poisson
        % even if a weak JPEG cue exists.
        if (~isSaltPepper) && (hfKurt > 20) && (slope < -1e-6)
            isPoisson = true; isGaussian = false; isJPEG = false; end

        % ----------------------
        % Fallback for weak Poisson signals
        % If Poisson heuristic failed but the high-frequency kurtosis is
        % extremely large (strong heavy tails) and no other noise was
        % detected, assume Poisson. This is a conservative fallback.
        if ~isPoisson && ~isSaltPepper && ~isJPEG && ~isGaussian && (hfKurt > 20) && (slope < -1e-6)
            isPoisson = true; end
        % end diagnostics

        % ===========================================================
        % FINAL DECISION
        % ===========================================================
        if isSaltPepper
            out = 'salt_pepper';
        elseif isPoisson
            out = 'poisson';
        elseif isGaussian
            out = 'gaussian';
        elseif isJPEG
            out = 'jpeg_artifact';
        else
            out = 'none';
        end

    catch ME
        disp("ERROR:");
        disp(getReport(ME));
    end
end
