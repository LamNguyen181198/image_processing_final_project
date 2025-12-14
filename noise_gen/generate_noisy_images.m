function generate_noisy_images(image_path, output_dir, num_images, noise_type)
% GENERATE_NOISY_IMAGES Generate multiple noisy images with varying parameters
%
% Usage:
%   generate_noisy_images('image.jpg', 'output', 10, 'gaussian')
%   generate_noisy_images('image.jpg', 'output', 5, 'all')
%
% Arguments:
%   image_path  - Path to the clean input image
%   output_dir  - Directory to save noisy images
%   num_images  - Number of noisy images to generate per noise type
%   noise_type  - 'all', 'gaussian', 'salt_pepper', 'poisson', 'speckle', 'uniform'

    % Read image
    clean_img = imread(image_path);
    
    % Create output directory
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Save clean image
    imwrite(clean_img, fullfile(output_dir, 'clean_original.png'));
    
    % Generate noise based on type
    if strcmp(noise_type, 'all')
        types = {'gaussian', 'salt_pepper', 'speckle', 'uniform'};
        for i = 1:length(types)
            generate_noise_type(clean_img, output_dir, num_images, types{i});
        end
    else
        generate_noise_type(clean_img, output_dir, num_images, noise_type);
    end
    
    fprintf('\nGenerated %d images for %s noise type(s)\n', num_images, noise_type);
end

function generate_noise_type(img, output_dir, num_images, noise_type)
    % Generate multiple images with varying noise parameters
    % Updated with more realistic noise levels for better training
    
    for i = 1:num_images
        switch noise_type
            case 'gaussian'
                % Realistic Gaussian noise levels: sigma 3-15 (avoid extreme noise)
                sigma_values = linspace(3, 15, num_images);
                sigma = sigma_values(i);
                noisy = add_gaussian_noise(img, 0, sigma);
                filename = sprintf('gaussian_%02d_sigma%.1f.png', i, sigma);
                
            case 'salt_pepper'
                % Moderate impulse noise: density 0.01-0.15 (typical real-world range)
                % Avoid extreme densities that make images unrecognizable
                density_values = linspace(0.01, 0.15, num_images);
                density = density_values(i);
                noisy = imnoise(img, 'salt & pepper', density);
                filename = sprintf('salt_pepper_%02d_density%.4f.png', i, density);
                
            case 'speckle'
                % Realistic speckle variance: 0.05-0.25 (multiplicative noise)
                variance_values = linspace(0.05, 0.25, num_images);
                variance = variance_values(i);
                noisy = imnoise(img, 'speckle', variance);
                filename = sprintf('speckle_%02d_var%.3f.png', i, variance);
                
            case 'uniform'
                % Moderate uniform noise range: 10-40 (avoid overwhelming noise)
                range_values = linspace(10, 40, num_images);
                range = range_values(i);
                noisy = add_uniform_noise(img, -range, range);
                filename = sprintf('uniform_%02d_range%.1f.png', i, range);
        end
        
        imwrite(noisy, fullfile(output_dir, filename));
        fprintf('Saved: %s\n', filename);
    end
end

function noisy = add_gaussian_noise(img, mean_val, sigma)
    img_double = im2double(img);
    gaussian_noise = mean_val/255 + (sigma/255) * randn(size(img));
    noisy = img_double + gaussian_noise;
    noisy = im2uint8(noisy);
end

function noisy = add_uniform_noise(img, low, high)
    img_double = im2double(img);
    uniform_noise = (low + (high - low) * rand(size(img))) / 255;
    noisy = img_double + uniform_noise;
    noisy = max(0, min(1, noisy));
    noisy = im2uint8(noisy);
end