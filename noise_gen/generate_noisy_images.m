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
        types = {'gaussian', 'salt_pepper', 'poisson', 'speckle', 'uniform', 'jpeg_artifacts'};
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
    
    for i = 1:num_images
        switch noise_type
            case 'gaussian'
                sigma = randi([5, 20]);   % moderate Gaussian noise
                noisy = add_gaussian_noise(img, 0, sigma);
                filename = sprintf('gaussian_%02d_sigma%.1f.png', i, sigma);
                
            case 'salt_pepper'
                density = rand() * 0.5;   
                noisy = imnoise(img, 'salt & pepper', density);
                filename = sprintf('salt_pepper_%02d_density%.4f.png', i, density);
                
            case 'poisson'
                % Poisson noise: variance = mean (in photon counts)
                % Lower peak = image scaled to fewer photon counts = more relative noise
                % Peak values chosen to show clear progression from subtle to heavy noise
                peak_values = [50, 20, 10, 5, 2];  % Descending = increasing noise
                peak = peak_values(mod(i-1, length(peak_values)) + 1);
                
                % Convert to double [0,1]
                img_norm = im2double(img);
                
                % Scale up to peak photon count
                img_scaled = img_norm * peak;
                
                % Add Poisson noise (models photon shot noise)
                noisy_scaled = imnoise(img_scaled, 'poisson');
                
                % Scale back to [0,1]
                noisy = noisy_scaled / peak;
                
                % Clip to valid range
                noisy = max(0, min(1, noisy));
                
                filename = sprintf('poisson_%02d_peak%d.png', i, peak);
                fprintf('Generated %s (SNR = %.1f dB, lower peak = more noise)\n', ...
                        filename, 10*log10(peak));
                
                % Save as PNG (lossless)
                imwrite(im2uint8(noisy), fullfile(output_dir, filename));
                continue;  % Skip the imwrite at the end

            case 'speckle'
                variance = 0.1 + (i-1) * (0.5-0.1) / max(num_images-1, 1);  
                noisy = imnoise(img, 'speckle', variance);
                filename = sprintf('speckle_%02d_var%.3f.png', i, variance);
                
            case 'uniform'
                range = 20 + (i-1) * (50-20) / max(num_images-1, 1);   % moderate uniform noise
                noisy = add_uniform_noise(img, -range, range);
                filename = sprintf('uniform_%02d_range%.1f.png', i, range);
            
            case 'jpeg_artifacts'
                quality = randi([30, 70]);   % reasonable JPEG compression
                noisy = add_jpeg_artifacts(img, quality);
                filename = sprintf('jpeg_artifact_%02d_q%d.png', i, quality);
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

function noisy = add_jpeg_artifacts(img, quality)
    temp_file = [tempname '.jpg'];
    imwrite(img, temp_file, 'Quality', quality);
    noisy = imread(temp_file);
    delete(temp_file);
end