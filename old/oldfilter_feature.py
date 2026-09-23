import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import butter, filtfilt

# 1. Setup the Butterworth Bandpass Filter
# Ganglion Nyquist frequency is 100Hz (half of 200Hz). We filter between 10Hz and 90Hz.
def butter_bandpass_filter(data, lowcut=10.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# 2. The Feature Extraction Math (FFT -> Avg & Max)
def extract_features_from_window(window_data):
    features = []
    # Loop through Channel_1 to Channel_4
    for col in window_data.columns:
        signal = window_data[col].values
        
        # Apply the Butterworth filter
        filtered_signal = butter_bandpass_filter(signal)
        
        # Apply the Fast Fourier Transform (FFT)
        fft_vals = np.abs(np.fft.fft(filtered_signal))
        
        # Calculate Average and Max of the FFT output
        features.append(np.mean(fft_vals))
        features.append(np.max(fft_vals))
        
    return features

def main():
    all_features = []
    
    # Find all CSV files in the current folder (excluding our final output file)
    csv_files = [f for f in glob.glob("*.csv") if f != "training_features.csv"]
    
    print(f"Found {len(csv_files)} data files. Extracting features...")
    
    for file in csv_files:
        # Extract the finger name from the filename (e.g., 'index_sample_1.csv' -> 'index')
        finger_label = file.split('_')[0].lower()
        
        df = pd.read_csv(file)
        
        # Make sure the file has enough data (2400 rows = 12 seconds at 200Hz)
        if len(df) < 1800:
            print(f"Skipping {file}: Not enough rows.")
            continue
            
        # --- EXTRACT "REST" DATA (First 3 seconds) ---
        for i in range(3):
            start_row = i * 200
            end_row = start_row + 200
            window = df.iloc[start_row:end_row]
            
            window_features = extract_features_from_window(window)
            window_features.append("rest") # Add the label
            all_features.append(window_features)
            
        # --- EXTRACT "FLEX" DATA (Seconds 3 to 9) ---
        for i in range(6):
            start_row = 600 + (i * 200)
            end_row = start_row + 200
            window = df.iloc[start_row:end_row]
            
            window_features = extract_features_from_window(window)
            window_features.append(finger_label) # Add the specific finger label
            all_features.append(window_features)

    # 3. Save everything to a Master Dataset
    columns = [
        "Ch1_Avg", "Ch1_Max", "Ch2_Avg", "Ch2_Max", 
        "Ch3_Avg", "Ch3_Max", "Ch4_Avg", "Ch4_Max", 
        "Label"
    ]
    
    final_df = pd.DataFrame(all_features, columns=columns)
    final_df.to_csv("training_features.csv", index=False)
    print("\nExtraction complete! Saved to training_features.csv")
    print(final_df['Label'].value_counts()) # Show how many samples we have per finger

if __name__ == "__main__":
    main()