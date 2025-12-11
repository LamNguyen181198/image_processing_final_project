function denoised = denoise_filters(imgPath, noiseType)
    % DENOISE_FILTERS Apply optimal denoising filter based on noise type
    %
    % Inputs:
    %   imgPath - Path to noisy image
    %   noiseType - Detected noise type: 'gaussian', 'salt_pepper',
    %               'speckle', 'uniform', 'jpeg_artifact', or 'clean'
    %
    % Output:
    %   denoised - Denoised image (grayscale, double precision [0,1])
    %
    % Optimal filters for each noise type:
    %   - gaussian: Gaussian filter (sigma=1.5)
    %   - salt_pepper: Median filter (5x5)
    %   - speckle: Lee filter (5x5)
    %   - uniform: Wiener filter
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
                % Speckle noise: Adaptive Bilateral Filter
                denoised = apply_adaptive_bilateral_filter(img);
                
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
    
    % Use gentle filtering to avoid oil painting effect
    if exist('imnlmfilt', 'file')
        % Non-Local Means: best balance of noise reduction and detail
        DegreeOfSmoothing = min(noiseStd * 1.5, 0.08);
        fprintf('  Using Non-Local Means (smoothing=%.4f)\n', DegreeOfSmoothing);
        denoised = imnlmfilt(img, 'DegreeOfSmoothing', DegreeOfSmoothing);
    elseif exist('imbilatfilt', 'file')
        % Bilateral filter: preserves edges better
        fprintf('  Using bilateral filter\n');
        intensitySigma = min(noiseStd * 1.2, 0.10);
        spatialSigma = 2.5;
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
    else
        % Gentle Gaussian - single pass to preserve details
        fprintf('  Using gentle Gaussian filtering\n');
        sigma = max(0.9, min(noiseStd * 18, 2.0));
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


function denoised = apply_adaptive_bilateral_filter(img)
    % Adaptive bilateral filter for speckle noise with luminance processing
    fprintf('Applying adaptive bilateral filter for speckle...\n');
    
    img = im2double(img);
    [rows, cols, channels] = size(img);
    
    if channels == 3
        % Convert to LAB color space for separate luminance processing
        lab = rgb2lab(img);
        L = lab(:,:,1);  % Luminance
        A = lab(:,:,2);  % Color channel a
        B = lab(:,:,3);  % Color channel b
        
        fprintf('  Processing luminance and color channels separately\n');
        
        % Estimate noise level from luminance
        noiseStd = estimate_noise_std(L / 100);  % Normalize L to [0,1]
        
        % Adaptive bilateral parameters based on noise level
        if noiseStd > 0.05
            spatialSigma = 2.0;
            intensitySigma = 0.12;
        elseif noiseStd > 0.03
            spatialSigma = 1.5;
            intensitySigma = 0.10;
        else
            spatialSigma = 1.2;
            intensitySigma = 0.08;
        end
        
        fprintf('  Noise level: %.4f, Spatial: %.2f, Intensity: %.3f\n', ...
                noiseStd, spatialSigma, intensitySigma);
        
        % Denoise luminance channel with adaptive bilateral
        L_normalized = L / 100;  % Normalize to [0,1]
        L_denoised = imbilatfilt(L_normalized, intensitySigma, spatialSigma);
        
        % Gentle denoising on color channels (preserve color information)
        A_denoised = imbilatfilt(A, intensitySigma * 0.5, spatialSigma);
        B_denoised = imbilatfilt(B, intensitySigma * 0.5, spatialSigma);
        
        % Apply stronger unsharp mask to luminance for detail enhancement
        sharpenAmount = 0.35;  % 35% sharpening for better detail
        blurred = imgaussfilt(L_denoised, 0.8);
        L_sharpened = L_denoised + sharpenAmount * (L_denoised - blurred);
        L_sharpened = max(0, min(1, L_sharpened));  % Clamp to valid range
        
        fprintf('  Applied unsharp mask (%.1f%% sharpening)\n', sharpenAmount * 100);
        
        % Reconstruct LAB image
        lab_denoised = cat(3, L_sharpened * 100, A_denoised, B_denoised);
        
        % Convert back to RGB
        denoised = lab2rgb(lab_denoised);
        
    else
        % Grayscale processing
        noiseStd = estimate_noise_std(img);
        
        if noiseStd > 0.05
            spatialSigma = 2.0;
            intensitySigma = 0.12;
        elseif noiseStd > 0.03
            spatialSigma = 1.5;
            intensitySigma = 0.10;
        else
            spatialSigma = 1.2;
            intensitySigma = 0.08;
        end
        
        fprintf('  Noise level: %.4f, Spatial: %.2f, Intensity: %.3f\n', ...
                noiseStd, spatialSigma, intensitySigma);
        
        % Apply adaptive bilateral filter
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
        
        % Apply stronger unsharp mask for detail enhancement
        sharpenAmount = 0.35;  % 35% sharpening
        blurred = imgaussfilt(denoised, 0.8);
        denoised = denoised + sharpenAmount * (denoised - blurred);
        
        fprintf('  Applied unsharp mask (%.1f%% sharpening)\n', sharpenAmount * 100);
    end
    
    % Ensure output is in valid range
    denoised = max(0, min(1, denoised));
    
    fprintf('  Speckle denoising complete\n');
end


function denoised = apply_wiener_filter(img)
    % Adaptive Wiener filter for uniform noise
    fprintf('Applying adaptive Wiener filter...\n');
    
    % Estimate noise variance
    noise_std = estimate_noise_std(img);
    noise_var = noise_std^2;
    
    fprintf('  Estimated noise std: %.4f\n', noise_std);
    
    % Adaptive window size based on noise level
    if noise_var > 0.003
        windowSize = [9 9];  % Larger window for heavy noise
    elseif noise_var > 0.001
        windowSize = [7 7];  % Medium-large window
    elseif noise_var > 0.0003
        windowSize = [5 5];  % Medium window
    else
        windowSize = [3 3];  % Small window for light noise
    end
    
    fprintf('  Using window size: %dx%d\n', windowSize(1), windowSize(2));
    
    % Apply Wiener filter with estimated noise
    denoised = wiener2(img, windowSize, noise_var);
end


function denoised = apply_bilateral_for_uniform(img)
    % Conservative bilateral filter optimized for uniform noise
    % Minimal smoothing to preserve sharpness
    fprintf('Applying conservative bilateral filter for uniform noise...\n');
    
    % Estimate noise level
    noise_std = estimate_noise_std(img);
    fprintf('  Noise level: %.4f\n', noise_std);
    
    % Very conservative bilateral parameters - prioritize sharpness
    if noise_std > 0.05
        % Heavy noise - still keep it gentle
        spatialSigma = 1.8;
        intensitySigma = 0.06;
        sharpenAmount = 0.3;
        fprintf('  Heavy uniform noise: bilateral(%.1f, %.3f)\n', spatialSigma, intensitySigma);
    elseif noise_std > 0.03
        % Moderate noise
        spatialSigma = 1.3;
        intensitySigma = 0.04;
        sharpenAmount = 0.4;
        fprintf('  Moderate uniform noise: bilateral(%.1f, %.3f)\n', spatialSigma, intensitySigma);
    else
        % Light noise - minimal filtering
        spatialSigma = 1.0;
        intensitySigma = 0.03;
        sharpenAmount = 0.5;
        fprintf('  Light uniform noise: bilateral(%.1f, %.3f)\n', spatialSigma, intensitySigma);
    end
    
    % Apply single-pass bilateral filter (no second pass to avoid blur)
    if exist('imbilatfilt', 'file')
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
    else
        % Fallback: Very small Wiener filter
        fprintf('  Warning: imbilatfilt not available, using Wiener\n');
        windowSize = [3 3];
        denoised = wiener2(img, windowSize);
    end
    
    % Apply unsharp masking to enhance details
    if exist('imsharpen', 'file')
        denoised = imsharpen(denoised, 'Radius', 1.5, 'Amount', sharpenAmount);
        fprintf('  Applied detail enhancement (amount=%.1f)\n', sharpenAmount);
    else
        % Manual unsharp mask
        blurred = imgaussfilt(denoised, 1.0);
        highPass = denoised - blurred;
        denoised = denoised + sharpenAmount * highPass;
        denoised = max(0, min(1, denoised));
        fprintf('  Applied manual sharpening (amount=%.1f)\n', sharpenAmount);
    end
end


function denoised = apply_bilateral_filter(img)
    % Improved bilateral filter for JPEG blocking artifacts
    fprintf('Applying adaptive Bilateral filter...\n');
    
    % Detect JPEG artifact severity by analyzing block boundaries
    artifact_strength = detect_blocking_artifacts(img);
    
    % Adaptive parameters based on artifact strength
    if artifact_strength > 0.015
        % Heavy JPEG artifacts
        spatialSigma = 2.5;
        intensitySigma = 0.12;
        fprintf('  Heavy artifacts detected: strong filtering\n');
    elseif artifact_strength > 0.008
        % Moderate artifacts
        spatialSigma = 2.0;
        intensitySigma = 0.08;
        fprintf('  Moderate artifacts detected: medium filtering\n');
    else
        % Light artifacts
        spatialSigma = 1.5;
        intensitySigma = 0.05;
        fprintf('  Light artifacts detected: gentle filtering\n');
    end
    
    % Check if Image Processing Toolbox has imbilatfilt
    if exist('imbilatfilt', 'file')
        denoised = imbilatfilt(img, intensitySigma, spatialSigma);
    else
        % Fallback: Use custom bilateral filter implementation
        denoised = bilateral_filter_custom(img, spatialSigma, intensitySigma);
    end
end


function denoised = bilateral_filter_custom(img, spatialSigma, intensitySigma)
    % Custom bilateral filter implementation
    
    % Window size based on spatial sigma
    windowSize = ceil(3 * spatialSigma) * 2 + 1;
    halfWindow = floor(windowSize / 2);
    
    [rows, cols] = size(img);
    denoised = zeros(rows, cols);
    
    % Pad image
    imgPadded = padarray(img, [halfWindow halfWindow], 'replicate');
    
    % Create spatial Gaussian kernel
    [X, Y] = meshgrid(-halfWindow:halfWindow, -halfWindow:halfWindow);
    spatialKernel = exp(-(X.^2 + Y.^2) / (2 * spatialSigma^2));
    
    for i = 1:rows
        for j = 1:cols
            % Extract local window
            window = imgPadded(i:i+windowSize-1, j:j+windowSize-1);
            
            % Center pixel
            centerPixel = imgPadded(i+halfWindow, j+halfWindow);
            
            % Intensity/range weights
            intensityDiff = window - centerPixel;
            intensityKernel = exp(-(intensityDiff.^2) / (2 * intensitySigma^2));
            
            % Combine spatial and intensity kernels
            bilateralKernel = spatialKernel .* intensityKernel;
            
            % Normalize and apply
            denoised(i, j) = sum(window(:) .* bilateralKernel(:)) / sum(bilateralKernel(:));
        end
    end
end


% ========================================================================
% Helper Functions for Noise Estimation
% ========================================================================

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
