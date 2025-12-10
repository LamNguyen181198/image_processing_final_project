function denoised = denoise_filters(imgPath, noiseType)
    % DENOISE_FILTERS Apply optimal denoising filter based on noise type
    %
    % Inputs:
    %   imgPath - Path to noisy image
    %   noiseType - Detected noise type: 'gaussian', 'salt_pepper', 'poisson',
    %               'speckle', 'uniform', 'jpeg_artifact', or 'clean'
    %
    % Output:
    %   denoised - Denoised image (grayscale, double precision [0,1])
    %
    % Optimal filters for each noise type:
    %   - gaussian: Gaussian filter (sigma=1.5)
    %   - salt_pepper: Median filter (5x5)
    %   - poisson: Anscombe + Wiener filter
    %   - speckle: Lee filter (5x5)
    %   - uniform: Wiener filter
    %   - jpeg_artifact: Bilateral filter
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
                
            case 'poisson'
                % Poisson noise: Anscombe transform + Wiener filter
                denoised = apply_poisson_filter(img);
                
            case 'speckle'
                % Speckle noise: Lee filter
                denoised = apply_lee_filter(img);
                
            case 'uniform'
                % Uniform noise: Wiener filter
                denoised = apply_wiener_filter(img);
                
            case 'jpeg_artifact'
                % JPEG artifacts: Bilateral filter
                denoised = apply_bilateral_filter(img);
                
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
    % Gaussian filter with sigma=1.5
    fprintf('Applying Gaussian filter (sigma=1.5)...\n');
    h = fspecial('gaussian', [7 7], 1.5);
    denoised = imfilter(img, h, 'replicate');
end


function denoised = apply_median_filter(img)
    % Median filter 5x5 for impulse noise
    fprintf('Applying Median filter (5x5)...\n');
    denoised = medfilt2(img, [5 5], 'symmetric');
end


function denoised = apply_poisson_filter(img)
    % Improved Poisson filter with brightness preservation
    fprintf('Applying Poisson filter (Anscombe + Adaptive Wiener)...\n');
    
    try
        % Ensure image is in valid range
        img = max(0, min(1, img));
        
        % Check if image is too dark (mean < 0.05)
        if mean(img(:)) < 0.05
            fprintf('WARNING: Image is very dark (low photon count)\n');
            fprintf('Poisson noise at low light is difficult to remove\n');
            % For very dark images, use gentle median filter
            denoised = medfilt2(img, [3 3], 'symmetric');
            return;
        end
        
        % Anscombe variance-stabilizing transform
        % For Poisson noise: variance = mean
        img_anscombe = 2 * sqrt(max(img, 0) + 3/8);
        
        % Use adaptive Wiener with smaller window
        % This preserves more detail than larger windows
        denoised_anscombe = wiener2(img_anscombe, [3 3]);
        
        % Inverse Anscombe transform
        denoised_anscombe = max(denoised_anscombe, 0);
        denoised = (denoised_anscombe / 2).^2 - 3/8;
        
        % Brightness preservation: match mean luminance to original
        original_mean = mean(img(:));
        denoised_mean = mean(denoised(:));
        
        if denoised_mean > 0
            brightness_correction = original_mean / denoised_mean;
            % Apply gentle correction (not full correction to avoid boosting noise)
            brightness_correction = min(brightness_correction, 1.2);  % Cap at 20% boost
            denoised = denoised * brightness_correction;
        end
        
        % Final clipping to valid range
        denoised = max(0, min(1, denoised));
        
    catch ME
        fprintf('WARNING: Poisson filter failed, using Wiener filter\n');
        fprintf('Error: %s\n', ME.message);
        % Fallback to simple Wiener with small window
        denoised = wiener2(img, [3 3]);
        denoised = max(0, min(1, denoised));
    end
end


function denoised = apply_bilateral_for_dark_poisson(img)
    % Special handling for very dark images with Poisson noise
    fprintf('Using bilateral filter for low-light Poisson noise...\n');
    
    try
        if exist('imbilatfilt', 'file')
            % Aggressive bilateral filtering
            denoised = imbilatfilt(img, 0.1, 2.5);
        else
            % Custom bilateral for very noisy dark images
            denoised = bilateral_filter_custom(img, 3.0, 0.15);
        end
        denoised = max(0, min(1, denoised));
    catch
        % Last resort: median filter
        fprintf('Falling back to median filter\n');
        denoised = medfilt2(img, [3 3], 'symmetric');
    end
end


function denoised = apply_lee_filter(img)
    % Lee filter for speckle noise
    fprintf('Applying Lee filter (5x5)...\n');
    
    windowSize = 5;
    halfWindow = floor(windowSize / 2);
    [rows, cols] = size(img);
    denoised = zeros(rows, cols);
    
    % Pad image
    imgPadded = padarray(img, [halfWindow halfWindow], 'replicate');
    
    % Estimate noise variance using median absolute deviation
    noiseVar = (median(abs(img(:) - median(img(:)))) / 0.6745)^2;
    
    for i = 1:rows
        for j = 1:cols
            % Extract local window
            window = imgPadded(i:i+windowSize-1, j:j+windowSize-1);
            
            % Local statistics
            localMean = mean(window(:));
            localVar = var(window(:));
            
            % Lee filter formula
            if localVar > 0
                k = max(0, (localVar - noiseVar) / localVar);
            else
                k = 0;
            end
            
            centerPixel = imgPadded(i+halfWindow, j+halfWindow);
            denoised(i, j) = localMean + k * (centerPixel - localMean);
        end
    end
end


function denoised = apply_wiener_filter(img)
    % Wiener filter for uniform noise
    fprintf('Applying Wiener filter (5x5)...\n');
    denoised = wiener2(img, [5 5]);
end


function denoised = apply_bilateral_filter(img)
    % Bilateral filter for JPEG artifacts
    fprintf('Applying Bilateral filter...\n');
    
    % Parameters for bilateral filter
    spatialSigma = 2.0;  % Spatial domain sigma
    intensitySigma = 0.1; % Intensity/range domain sigma
    
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
