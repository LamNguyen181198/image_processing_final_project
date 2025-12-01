function noise_type = detect_noise_type(image_path)
    % DETECT_NOISE_TYPE Detects noise using decision tree approach
    %   Uses mutually exclusive tests instead of scoring
    
    % Read image
    img = imread(image_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img);
    
    fprintf('\n=== Noise Detection (Decision Tree) ===\n');
    fprintf('Image: %s\n\n', image_path);
    
    %% DECISION 1: Is this Salt & Pepper? (Most distinctive pattern)
    fprintf('TEST 1: Salt & Pepper Detection\n');
    
    % S&P creates ONLY values at extremes (0 or 255)
    % Count pixels at exact extremes
    exact_black = sum(img(:) == 0);
    exact_white = sum(img(:) == 255);
    exact_extreme_ratio = (exact_black + exact_white) / numel(img);
    
    fprintf('  Exact extreme pixels (0 or 255): %.2f%%\n', exact_extreme_ratio * 100);
    
    % S&P key feature: median filter removes it almost completely
    median_filtered = medfilt2(uint8(img), [3 3]);
    diff_before_after = mean(abs(img(:) - double(median_filtered(:))));
    
    fprintf('  Median filter effect: %.2f (high = S&P)\n', diff_before_after);
    
    % Decision: S&P if has many exact extremes AND median filter is very effective
    if exact_extreme_ratio > 0.02 && diff_before_after > 20
        noise_type = 'salt_pepper';
        fprintf('  => YES: Salt & Pepper detected\n\n');
        display_results(img, noise_type);
        return;
    else
        fprintf('  => NO: Not Salt & Pepper\n');
        fprintf('     (exact extremes too low or median filter not effective)\n\n');
    end
    
    %% DECISION 2: Is this Gaussian? (Additive, constant variance)
    fprintf('TEST 2: Gaussian Noise Detection\n');
    
    % Extract noise residual
    noise = img - imgaussfilt(img, 2);
    
    % Gaussian Test 1: Is noise distribution symmetric and normal?
    skew = abs(skewness(noise(:)));
    fprintf('  Skewness: %.3f (< 0.3 = symmetric)\n', skew);
    
    % Gaussian Test 2: Does noise have constant variance?
    [h, w] = size(img);
    n_blocks = 16;
    block_h = floor(h / 4);
    block_w = floor(w / 4);
    noise_stds = [];
    
    for i = 1:4
        for j = 1:4
            block = noise((i-1)*block_h+1:min(i*block_h,h), ...
                         (j-1)*block_w+1:min(j*block_w,w));
            noise_stds(end+1) = std(block(:));
        end
    end
    
    std_cv = std(noise_stds) / mean(noise_stds);
    fprintf('  Variance consistency: %.3f (< 0.2 = constant)\n', std_cv);
    
    % Gaussian Test 3: Is noise independent of image intensity?
    img_smooth = imgaussfilt(img, 3);
    noise_abs = abs(noise(:));
    intensity = img_smooth(:);
    
    % Divide into intensity bins and check noise variance
    [~, bin_idx] = histc(intensity, linspace(min(intensity), max(intensity), 5));
    bin_stds = [];
    for b = 1:4
        bin_noise = noise_abs(bin_idx == b);
        if length(bin_noise) > 100
            bin_stds(end+1) = std(bin_noise);
        end
    end
    
    if ~isempty(bin_stds)
        intensity_independence = std(bin_stds) / mean(bin_stds);
    else
        intensity_independence = 1;
    end
    
    fprintf('  Intensity independence: %.3f (< 0.3 = independent)\n', intensity_independence);
    
    % Decision: Gaussian if symmetric, constant variance, intensity-independent
    if skew < 0.4 && std_cv < 0.25 && intensity_independence < 0.4
        noise_type = 'gaussian';
        fprintf('  => YES: Gaussian noise detected\n\n');
        display_results(img, noise_type);
        return;
    else
        fprintf('  => NO: Not Gaussian\n');
        fprintf('     (not symmetric or variance not constant)\n\n');
    end
    
    %% No clear detection
    noise_type = 'unknown';
    fprintf('=> RESULT: Unknown noise type\n\n');
    display_results(img, noise_type);
end

function display_results(img, noise_type)
    % Visualization
    figure('Position', [100 100 1200 400]);
    
    subplot(1,3,1);
    imshow(uint8(img));
    title('Original Image');
    
    subplot(1,3,2);
    if strcmp(noise_type, 'salt_pepper')
        median_filtered = medfilt2(uint8(img), [3 3]);
        imshow(median_filtered);
        title('After Median Filter');
    elseif strcmp(noise_type, 'gaussian')
        noise = img - imgaussfilt(img, 2);
        histogram(noise(:), 50, 'Normalization', 'probability');
        title('Noise Distribution');
        xlabel('Value');
        ylabel('Probability');
        grid on;
    else
        imshow(uint8(img));
        title('Unknown Noise');
    end
    
    subplot(1,3,3);
    text(0.5, 0.5, upper(strrep(noise_type, '_', ' ')), ...
         'FontSize', 20, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
    axis off;
    title('Detection Result');
end