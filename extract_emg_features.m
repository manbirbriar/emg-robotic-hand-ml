% extract_emg_features.m
% This script loads raw OpenBCI Ganglion EMG data, filters it, extracts 
% mathematical features (MAV, RMS, FFT Peak), and exports a clean dataset
% ready for the MATLAB Classification Learner App.

%% 1. Configuration Setup
input_filename = 'master_emg_data.csv';
output_filename = 'emg_features.csv';

Fs = 200;            % Ganglion Sampling frequency in Hz
window_size = 100;   % 500ms window (100 samples)
overlap = 50;        % 250ms overlap (50 samples) for smoother transitions

%% 2. Load and Clean the Data
disp('Loading raw data...');
data = readtable(input_filename);

% Drop the old 'thumb' data so it doesn't conflict with 'new thumb'
data(strcmp(data.Label, 'thumb'), :) = []; 

labels = data.Label;
emg_signals = data{:, 1:4}; % Grab columns 1 through 4 (Ch1 to Ch4)

%% 3. Design the Digital Filters
disp('Applying digital filters...');
% A) 60 Hz Notch Filter (To remove Canadian power line noise)
d_notch = designfilt('bandstopiir', 'FilterOrder', 2, ...
    'HalfPowerFrequency1', 59, 'HalfPowerFrequency2', 61, ...
    'DesignMethod', 'butter', 'SampleRate', Fs);

% B) 20-90 Hz Butterworth Bandpass Filter 
% (Note: Because your board samples at 200Hz, the absolute highest frequency 
% it can physically see is 100Hz due to the Nyquist limit. Therefore, we 
% cap the bandpass at 90Hz instead of 500Hz like the paper did).
d_bandpass = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', 20, 'HalfPowerFrequency2', 90, ...
    'DesignMethod', 'butter', 'SampleRate', Fs);

% Apply filters to all 4 channels
filtered_emg = zeros(size(emg_signals));
for ch = 1:4
    % Pass through notch filter first, then bandpass
    temp = filtfilt(d_notch, emg_signals(:, ch));
    filtered_emg(:, ch) = filtfilt(d_bandpass, temp);
end

%% 4. Windowing and Feature Extraction
disp('Extracting mathematical features...');
num_samples = size(filtered_emg, 1);
step_size = window_size - overlap;
num_windows = floor((num_samples - window_size) / step_size) + 1;

% Pre-allocate an empty matrix for speed (4 channels * 3 features = 12 columns)
feature_matrix = zeros(num_windows, 12); 
window_labels = strings(num_windows, 1);

for w = 1:num_windows
    % Define the start and end of our 500ms chunk
    start_idx = (w-1)*step_size + 1;
    end_idx = start_idx + window_size - 1;
    
    window_data = filtered_emg(start_idx:end_idx, :);
    
    % Grab the label assigned to this window
    window_labels(w) = string(labels{end_idx}); 
    
    row_features = [];
    for ch = 1:4
        ch_data = window_data(:, ch);
        
        % Feature 1: Mean Absolute Value (MAV) - standard EMG amplitude measurement
        mav = mean(abs(ch_data));
        
        % Feature 2: Root Mean Square (RMS) - standard muscle power measurement
        rms_val = rms(ch_data);
        
        % Feature 3: FFT Maximum Peak (Inspired by the Alvarado-Díaz paper)
        Y = fft(ch_data);
        P2 = abs(Y/window_size);
        P1 = P2(1:window_size/2+1);
        fft_max = max(P1(2:end)); % Find max frequency power (ignoring the 0Hz DC component)
        
        % Append the 3 features for this specific channel
        row_features = [row_features, mav, rms_val, fft_max];
    end
    % Save all 12 features for this 500ms window as a new row
    feature_matrix(w, :) = row_features;
end

%% 5. Export to Classification Learner Format
disp('Packaging data and saving to CSV...');
var_names = {'Ch1_MAV', 'Ch1_RMS', 'Ch1_FFTMax', ...
             'Ch2_MAV', 'Ch2_RMS', 'Ch2_FFTMax', ...
             'Ch3_MAV', 'Ch3_RMS', 'Ch3_FFTMax', ...
             'Ch4_MAV', 'Ch4_RMS', 'Ch4_FFTMax', 'Label'};
         
% Convert the matrix into a formal MATLAB Table
features_table = array2table(feature_matrix);
features_table.Label = categorical(window_labels);
features_table.Properties.VariableNames = var_names;

% Save to disk
writetable(features_table, output_filename);
disp('Success! You can now import emg_features.csv into the Classification Learner App.');