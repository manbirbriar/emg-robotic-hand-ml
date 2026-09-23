import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

FS = 200

def apply_filters(data):
    """Applies a 60Hz Notch and 15-95Hz Bandpass filter."""
    # 1. 60Hz Notch Filter (Removes wall power hum)
    bn, an = iirnotch(60 / (FS / 2), 30)
    data_notched = filtfilt(bn, an, data, axis=0)
    
    # 2. 15-95Hz Butterworth Bandpass (Isolates muscle frequencies)
    b, a = butter(4, [15 / (FS / 2), 95 / (FS / 2)], btype='bandpass')
    data_filtered = filtfilt(b, a, data_notched, axis=0)
    
    return data_filtered

def extract_features(window_data):
    """Extracts Mean, Max, and Standard Deviation for all 4 channels."""
    features = []
    # Loop through each of the 4 columns (channels)
    for ch in range(4):
        ch_data = window_data[:, ch]
        # Taking the absolute value for Mean and Max is standard for EMG
        abs_data = np.abs(ch_data)
        features.extend([np.mean(abs_data), np.max(abs_data), np.std(ch_data)])
    
    return np.array(features)