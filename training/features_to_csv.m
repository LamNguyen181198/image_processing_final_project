function features_to_csv(images_folder, output_csv)
    % FEATURES_TO_CSV - Batch extract features from images and save to CSV
    %
    % Usage:
    %   features_to_csv('noisy_images/', 'features.csv')
    %
    % Arguments:
    %   images_folder - Directory containing images
    %   output_csv    - Output CSV file path

    % Add the directory containing extract_features.m to path
    % (extract_features.m is now in the same directory as features_to_csv.m)
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    
    % Get all image files
    img_files = dir(fullfile(images_folder, '*.png'));
    img_files = [img_files; dir(fullfile(images_folder, '*.jpg'))];
    img_files = [img_files; dir(fullfile(images_folder, '*.jpeg'))];
    img_files = [img_files; dir(fullfile(images_folder, '*.bmp'))];
    
    if isempty(img_files)
        error('No images found in %s', images_folder);
    end
    
    fprintf('Found %d images in %s\n\n', length(img_files), images_folder);
    
    % Extract features from first image to get field names
    first_img = fullfile(img_files(1).folder, img_files(1).name);
    sample_features = extract_features(first_img);
    field_names = fieldnames(sample_features);
    
    % Initialize data storage
    num_features = length(field_names);
    num_images = length(img_files);
    
    feature_matrix = zeros(num_images, num_features);
    filenames = cell(num_images, 1);
    labels = cell(num_images, 1);
    
    % Process each image
    for i = 1:num_images
        img_path = fullfile(img_files(i).folder, img_files(i).name);
        filename = img_files(i).name;
        
        fprintf('[%d/%d] Processing %s... ', i, num_images, filename);
        
        try
            features = extract_features(img_path);
            
            % Extract feature values in order
            for j = 1:num_features
                feature_matrix(i, j) = features.(field_names{j});
            end
            
            filenames{i} = filename;
            labels{i} = extract_label_from_filename(filename);
            
            fprintf('✓ (label: %s)\n', labels{i});
        catch ME
            fprintf('✗ ERROR: %s\n', ME.message);
            feature_matrix(i, :) = NaN;
            filenames{i} = filename;
            labels{i} = 'error';
        end
    end
    
    % Create table
    data_table = table();
    data_table.filename = filenames;
    data_table.label = labels;
    
    % Add feature columns
    for j = 1:num_features
        data_table.(field_names{j}) = feature_matrix(:, j);
    end
    
    % Write to CSV
    writetable(data_table, output_csv);
    
    fprintf('\n');
    fprintf('========================================\n');
    fprintf('✓ Features extracted successfully!\n');
    fprintf('  Total images: %d\n', num_images);
    fprintf('  Features per image: %d\n', num_features);
    fprintf('  Output: %s\n', output_csv);
    fprintf('========================================\n');
    
    % Display label distribution
    fprintf('\nLabel distribution:\n');
    unique_labels = unique(labels);
    for i = 1:length(unique_labels)
        count = sum(strcmp(labels, unique_labels{i}));
        fprintf('  %s: %d\n', unique_labels{i}, count);
    end
end


function label = extract_label_from_filename(filename)
    % Extract noise type label from filename
    filename_lower = lower(filename);
    
    if contains(filename_lower, 'clean') || contains(filename_lower, 'original')
        label = 'clean';
    elseif contains(filename_lower, 'gaussian')
        label = 'gaussian';
    elseif contains(filename_lower, 'salt_pepper')
        label = 'salt_pepper';
    elseif contains(filename_lower, 'poisson')
        label = 'poisson';
    elseif contains(filename_lower, 'speckle')
        label = 'speckle';
    elseif contains(filename_lower, 'uniform')
        label = 'uniform';
    elseif contains(filename_lower, 'jpeg')
        label = 'jpeg_artifact';
    else
        label = 'unknown';
    end
end
