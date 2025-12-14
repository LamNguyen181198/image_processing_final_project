function denoised = denoise_filters(imgPath, noiseType)
    % DENOISE_FILTERS Apply optimal denoising filter based on noise type
    %
    % Inputs:
    %   imgPath - Path to noisy image
    %   noiseType - Detected noise type: 'gaussian', 'salt_pepper',
    %               'speckle', 'uniform', or 'clean'
    %
    % Output:
    %   denoised - Denoised image (grayscale, double precision [0,1])
    %
    % Optimal filters for each noise type:
    %   - gaussian: Non-Local Means filter (adaptive smoothing)
    %   - salt_pepper: Adaptive Median filter (3x3 to 7x7)
    %   - speckle: Non-Local Means in log domain + detail restoration
    %   - uniform: Bilateral filter + multi-stage sharpening
    %   - clean: No filtering (return original)
    
    try
        % Read and convert image
        img = imread(imgPath);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = im2double(img);
        
        % Apply appropriate filter based on noise type
        switch lower(noiseType)
            case 'gaussian'
                % Gaussian noise: Gaussian filter
                denoised = apply_gaussian_filter(img);
                
            case 'salt_pepper'
                % Salt & Pepper: Median filter
                denoised = apply_median_filter(img);
                
            case 'speckle'
                % Speckle noise: Non-Local Means in log domain + detail restoration
                denoised = apply_speckle_filter(img);
                
            case 'uniform'
                % Uniform noise: Bilateral filter (preserves edges better than Wiener)
                denoised = apply_bilateral_for_uniform(img);
                
            case 'clean'
                % No noise: Return original
                denoised = img;
                fprintf('Image is clean - no denoising applied\n');
                
            otherwise
                % Unknown noise type: Apply conservative Gaussian filter
                fprintf('Warning: Unknown noise type "%s" - applying Gaussian filter\n', noiseType);
                denoised = apply_gaussian_filter(img);
        end
        
        % Ensure output is in valid range [0, 1]
        denoised = max(0, min(1, denoised));
        
    catch ME
        fprintf('ERROR in denoise_filters: %s\n', ME.message);
        denoised = [];
    end
end


function denoised = apply_gaussian_filter(img)
    % Improved Gaussian filter with Non-Local Means for better quality
    fprintf('Applying Gaussian noise filter...\n');
    
    % Estimate noise level
    noiseStd = estimate_noise_std(img);
    fprintf('  Estimated noise std: %.4f\n', noiseStd);
    
    % Adaptive filtering based on measured noise level
    if exist('imnlmfilt', 'file')
        % Non-Local Means: Reduced smoothing to preserve details better
        % Formula: DegreeOfSmoothing proportional to noise variance
        % Reduced multiplier from 15 to 8 for lighter smoothing
        DegreeOfSmoothing = min(noiseStd * noiseStd * 8, 0.06);  % Reduced from 0.12 to 0.06
        DegreeOfSmoothing = max(DegreeOfSmoothing, 0.005);  % Reduced minimum from 0.01
        fprintf('  Using Non-Local Means (smoothing=%.4f)\n', DegreeOfSmoothing);
        denoised = imnlmfilt(img, 'DegreeOfSmoothing', DegreeOfSmoothing);
        
        % Add detail restoration to preserve edges and textures
        % Unsharp masking with strong emphasis on edges
        blurred = imgaussfilt(denoised, 0.8);
        detailLayer = denoised - blurred;
        sharpenAmount = 0.55;  % 55% detail restoration for pronounced outlines
        denoised = denoised + sharpenAmount * detailLayer;
        denoised = max(0, min(1, denoised));
        fprintf('  Applied detail restoration (%.0f%%)\n', sharpenAmount * 100);
        
    elseif exist('imbilatfilt', 'file')
        % Bilateral filter: scale with noise level
        fprintf('  Using bilateral filter\n');
        intensitySigma = min(noiseStd * 1.2, 0.08);  % Reduced from 1.5 and 0.12
        spatialSigma = min(1.2 + noiseStd * 6, 2.5);  % Reduced spatial range
        fprintf('  Bilateral params: intensity=%.4f, spatial=%.2f\n', intensitySigma, spatialSigma);
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
        
        % Detail restoration for bilateral too
        blurred = imgaussfilt(denoised, 0.8);
        detailLayer = denoised - blurred;
        denoised = denoised + 0.55 * detailLayer;
        denoised = max(0, min(1, denoised));
        fprintf('  Applied detail restoration\n');
    else
        % Gentle Gaussian - single pass to preserve details
        fprintf('  Using gentle Gaussian filtering\n');
        sigma = max(0.7, min(noiseStd * 12, 1.5));  % Reduced from 18 and 2.0
        kernelSize = 2 * ceil(2 * sigma) + 1;
        
        h = fspecial('gaussian', [kernelSize kernelSize], sigma);
        denoised = imfilter(img, h, 'replicate');
        fprintf('  Applied filtering (sigma=%.2f, kernel=%dx%d)\n', sigma, kernelSize, kernelSize);
    end
end


function denoised = apply_median_filter(img)
    % Adaptive median filter for impulse noise
    fprintf('Applying adaptive Median filter...\n');
    
    % Estimate impulse noise density
    impulse_density = estimate_impulse_density(img);
    
    % Adaptive window size based on noise density
    if impulse_density > 0.15
        windowSize = 7;  % Heavy impulse noise
    elseif impulse_density > 0.05
        windowSize = 5;  % Moderate impulse noise
    else
        windowSize = 3;  % Light impulse noise
    end
    
    fprintf('  Estimated impulse density: %.3f, using window=%dx%d\n', ...
            impulse_density, windowSize, windowSize);
    denoised = medfilt2(img, [windowSize windowSize], 'symmetric');
end

function denoised = apply_speckle_filter(img)
    % --- Speckle removal using log-domain Non-Local Means ---
    % Multiplicative speckle noise is converted to additive via log transform
    % Non-Local Means (primary) or bilateral filter (fallback) applied in log domain
    % Aggressive detail restoration compensates for smoothing

    img = im2double(img);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    img = max(img, eps);

    fprintf("Applying SPECKLE filter...\n");

    % Estimate speckle strength
    sigma = estimate_noise_std(img);
    fprintf("  Estimated sigma = %.4f\n", sigma);

    % ---- Convert multiplicative noise → additive ----
    logI = log(img);

    % ------------------------------------------------------
    % Log-domain filtering for multiplicative speckle noise
    % ------------------------------------------------------
    
    % Try Non-Local Means in log domain first (best approach)
    if exist('imnlmfilt', 'file')
        fprintf("  Using Non-Local Means in log domain...\n");
        logI = log(img);
        
        % Maximum effective smoothing for speckle removal
        DegreeOfSmoothing = min(sigma * 15, 0.20);
        DegreeOfSmoothing = max(DegreeOfSmoothing, 0.10);
        
        fprintf("  NLM smoothing = %.4f\n", DegreeOfSmoothing);
        logDenoised = imnlmfilt(logI, 'DegreeOfSmoothing', DegreeOfSmoothing);
        denoised = exp(logDenoised);
        denoised = max(0, min(1, denoised));
    
    % Fallback: Multi-pass bilateral in log domain
    elseif exist('imbilatfilt', 'file')
        fprintf("  Using bilateral filtering in log domain...\n");
        logI = log(img);
        
        % Very strong smoothing
        intensitySigma = sigma * 3.5;
        spatialSigma = 3.5;
        logDenoised = imbilatfilt(logI, intensitySigma, spatialSigma);
        
        % Second pass for residual noise
        intensitySigma2 = sigma * 2.0;
        logDenoised = imbilatfilt(logDenoised, intensitySigma2, spatialSigma);
        
        denoised = exp(logDenoised);
        denoised = max(0, min(1, denoised));
        fprintf("  Applied 2-pass bilateral filtering\n");
    
    % Last resort: SRAD
    else
        fprintf("  Using SRAD (35 iterations)...\n");
        denoised = srad_filter(img, 35, 0.20, 0.5);
    end
    
    % Very aggressive detail restoration to compensate for strong smoothing
    blurred = imgaussfilt(denoised, 1.0);
    denoised = denoised + 0.85 * (denoised - blurred);
    
    % Additional micro-detail boost
    microBlur = imgaussfilt(denoised, 0.6);
    denoised = denoised + 0.25 * (denoised - microBlur);
    denoised = max(0, min(1, denoised));
    fprintf("  Applied detail restoration\n");
    return;
end


function out = srad_filter(I, numIter, deltaT, q0)
    I = im2double(I);
    out = I;

    for t = 1:numIter
        north = [out(1,:); out(1:end-1,:)] - out;
        south = [out(2:end,:); out(end,:)] - out;
        east  = [out(:,2:end), out(:,end)] - out;
        west  = [out(:,1), out(:,1:end-1)] - out;

        grad2 = (north.^2 + south.^2 + east.^2 + west.^2);
        lap   = north + south + east + west;

        q = sqrt( abs(0.5*(grad2./(out.^2)) - ((lap./out).^2) ./ (1 + out.^2)) );
        c = 1 ./ (1 + ((q.^2 - q0)./(q0 + eps)));
        c = max(0, min(1, c));

        div = c.*north + c.*south + c.*east + c.*west;
        out = out + deltaT * div;
        out = max(0, min(1, out));
    end
end


function denoised = apply_bilateral_for_uniform(img)
    % Enhanced bilateral filter for uniform noise with edge-preserving sharpening
    fprintf('Applying enhanced bilateral filter for uniform noise...\n');
    
    % Estimate noise level
    noise_std = estimate_noise_std(img);
    fprintf('  Noise level: %.4f\n', noise_std);
    
    % Very light bilateral filtering - minimal smoothing
    % Prioritize edge and detail preservation over noise removal
    spatialSigma = min(0.5 + (noise_std * 8), 1.2);
    spatialSigma = max(spatialSigma, 0.5);
    
    % Very low intensity sigma - strong edge preservation
    intensitySigma = min(0.01 + (noise_std * 0.4), 0.05);
    intensitySigma = max(intensitySigma, 0.01);
    
    % Aggressive sharpening for maximum detail pop
    sharpenAmount = max(0.8, min(1.0 - (noise_std * 1.5), 1.0));
    sharpenRadius = 1.0;  % Tighter radius for crisp details
    
    fprintf('  Bilateral params: spatial=%.2f, intensity=%.3f\n', ...
            spatialSigma, intensitySigma);
    fprintf('  Sharpening: amount=%.2f, radius=%.2f\n', sharpenAmount, sharpenRadius);
    
    % Apply very light bilateral filter
    if exist('imbilatfilt', 'file')
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
    else
        % Fallback: Very small Wiener filter
        fprintf('  Warning: imbilatfilt not available, using Wiener\n');
        windowSize = [3 3];
        denoised = wiener2(img, windowSize);
    end
    
    % Stage 1: Strong primary sharpening for detail pop
    if exist('imsharpen', 'file')
        denoised = imsharpen(denoised, 'Radius', sharpenRadius, 'Amount', sharpenAmount);
        fprintf('  Applied primary sharpening (amount=%.2f)\n', sharpenAmount);
    else
        % Manual unsharp mask with tight radius
        blurred = imgaussfilt(denoised, sharpenRadius * 0.7);
        highPass = denoised - blurred;
        denoised = denoised + sharpenAmount * highPass;
        denoised = max(0, min(1, denoised));
        fprintf('  Applied manual sharpening (amount=%.2f)\n', sharpenAmount);
    end
    
    % Stage 2: Micro-detail enhancement for texture pop
    % Use smaller Gaussian to capture fine details
    microBlurred = imgaussfilt(denoised, 0.5);
    microDetail = denoised - microBlurred;
    microEnhance = 0.4;  % 40% micro-detail boost
    denoised = denoised + microEnhance * microDetail;
    denoised = max(0, min(1, denoised));
    fprintf('  Applied micro-detail enhancement (%.1f%%)\n', microEnhance * 100);
    
    % Stage 3: Edge contrast boost for outline definition
    if noise_std < 0.06
        % Detect edges using Sobel
        [Gx, Gy] = gradient(denoised);
        edgeMag = sqrt(Gx.^2 + Gy.^2);
        
        % Create strong edge mask
        edgeMask = edgeMag / (max(edgeMag(:)) + eps);
        edgeMask = edgeMask .^ 0.3;  % More aggressive masking
        
        % Edge contrast enhancement using local contrast
        edgeBoost = 0.25;  % 25% edge boost
        localContrast = imgaussfilt(denoised, 0.6) - imgaussfilt(denoised, 2.0);
        denoised = denoised + edgeBoost * edgeMask .* localContrast;
        denoised = max(0, min(1, denoised));
        fprintf('  Applied edge contrast boost (%.1f%%)\n', edgeBoost * 100);
    end
    
    % Stage 4: Global contrast adjustment for overall pop
    % Subtle contrast stretch to make details stand out
    contrastAmount = 0.15;  % 15% contrast increase
    mid = 0.5;
    denoised = mid + (1 + contrastAmount) * (denoised - mid);
    denoised = max(0, min(1, denoised));
    fprintf('  Applied global contrast enhancement (%.1f%%)\n', contrastAmount * 100);
end


% ========================================================================
% Helper Functions for Noise Estimation
% ========================================================================

function sigma = estimate_noise_std_rgb(img)
    % Robust noise estimate for color images:
    % Use luminance high-frequency MAD from Laplacian across channels
    if size(img,3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    sigma = estimate_noise_std(gray);
end

function noise_std = estimate_noise_std(img)
    % Estimate noise standard deviation using median absolute deviation
    % This is robust to image content and works well for additive noise
    
    % Use high-frequency components via Laplacian
    h = fspecial('laplacian', 0.2);
    imgLaplacian = imfilter(img, h, 'replicate');
    
    % Robust MAD (Median Absolute Deviation) estimator
    sigma = median(abs(imgLaplacian(:))) / 0.6745;
    
    noise_std = sigma;
end


function impulse_density = estimate_impulse_density(img)
    % Estimate impulse noise density (salt & pepper)
    % by detecting extreme outliers
    
    % Compute median-filtered version
    img_median = medfilt2(img, [3 3], 'symmetric');
    
    % Find pixels that differ significantly from their neighborhood
    diff = abs(img - img_median);
    
    % Threshold for impulse detection
    threshold = 0.3;  % Pixels differing by >30% likely impulses
    impulses = diff > threshold;
    
    impulse_density = sum(impulses(:)) / numel(img);
end


function artifact_strength = detect_blocking_artifacts(img)
    % Detect JPEG blocking artifacts by analyzing 8x8 block boundaries
    % JPEG compression creates visible discontinuities at block edges
    
    [rows, cols] = size(img);
    
    % Compute horizontal gradients at 8-pixel intervals
    horiz_grads = [];
    for col = 8:8:cols-1
        if col < cols
            grad = abs(img(:, col+1) - img(:, col));
            horiz_grads = [horiz_grads; grad];
        end
    end
    
    % Compute vertical gradients at 8-pixel intervals
    vert_grads = [];
    for row = 8:8:rows-1
        if row < rows
            grad = abs(img(row+1, :) - img(row, :));
            vert_grads = [vert_grads; grad'];
        end
    end
    
    % Average gradient at block boundaries
    boundary_grad = mean([horiz_grads; vert_grads]);
    
    % Compare to overall image gradient
    [gx, gy] = gradient(img);
    overall_grad = mean(sqrt(gx(:).^2 + gy(:).^2));
    
    % Artifact strength: ratio of boundary to overall gradient
    if overall_grad > 0
        artifact_strength = boundary_grad / overall_grad;
    else
        artifact_strength = 0;
    end
end
