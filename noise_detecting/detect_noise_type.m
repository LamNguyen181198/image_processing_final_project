function out = detect_noise_type(imgPath)
%DETECT_NOISE_TYPE Detect Gaussian or JPEG artifacts. Compatible with Python caller.

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
        % JPEG introduces block boundaries at multiples of 8

        % ---------- (A) BLOCKINESS (we keep your original idea) ----------
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

        % ---------- (B) DCT quantization detection ----------
        blockSize = 8;
        fun = @(b) dct2(b.data);

        dctImg = blockproc(img, [blockSize blockSize], fun);

        % Extract AC high-frequency region
        HF_AC = abs(dctImg(5:end,5:end));

        % Quantization leaves *strong peaks* at multiples of quant step
        % We look at histogram "spikiness"
        edges = linspace(0, max(HF_AC(:)), 80);
        hst = histcounts(HF_AC(:), edges);

        % Peakiness metric
        peakiness = max(hst) / mean(hst);

        % Empirical thresholds (tuned)
        isJPEG_DCT = peakiness > 3.2;  % sensitive even at QF=70–90
        isJPEG_Block = (blockiness_ratio > 1.25) && (blockBoundary > 0.010);

        isJPEG = isJPEG_DCT || isJPEG_Block;

        % ===========================================================
        % 3. SALT & PEPPER DETECTION (balanced, stable)
        % ===========================================================

        medImg = medfilt2(img, [3 3]);

        % 1) IMPULSE DEVIATION (moderate sensitivity)
        impulseMask = abs(img - medImg) > 0.25;
        impulseRatio = sum(impulseMask(:)) / numel(img);

        % 2) HARD EXTREME PIXELS (very specific to S&P)
        extremeRatio = (sum(img(:) <= 0.02) + sum(img(:) >= 0.98)) / numel(img);

        % 3) IMPULSE PEAK SCORE
        % If S&P exists, median filtering reduces noise strongly → big residual drop
        medResidual = mean(abs(img(:) - medImg(:)));

        % Residual of the high-frequency (Gaussian is smoother)
        hf = img - imgaussfilt(img, 1);   % high-frequency component
        medResidualHF = mean(abs(hf(:))); % now safe to index

        % Ratio is strong for impulse noise
        impulsePeak = medResidualHF / (medResidual + 1e-8);

        % ---------- Decision combining all metrics ----------
        cond1 = impulseRatio > 0.008;      % ≥ 0.8% impulses
        cond2 = extremeRatio > 0.003;      % ≥ 0.3% extreme pixels
        cond3 = impulsePeak > 2.0;         % big HF vs median difference

        % Require at least TWO conditions to classify as S&P
        isSaltPepper = (cond1 + cond2 + cond3) >= 2;

        % Prevent JPEG or Gaussian misfiring when impulses exist
        if isSaltPepper
            isGaussian = false;
            isJPEG = false;
        end

        % ===========================================================
        % FINAL DECISION
        % ===========================================================
        if isSaltPepper
            out = 'salt_pepper';
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
