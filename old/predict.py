import time
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# 1. Recreate the exact same filter used in training
def butter_bandpass_filter(data, lowcut=10.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# 2. Recreate the exact same feature extraction
def extract_features(emg_data_matrix):
    features = []
    # emg_data_matrix has shape (4 channels, 200 samples)
    for i in range(4):
        signal = emg_data_matrix[i]
        
        # Filter and FFT
        filtered_signal = butter_bandpass_filter(signal)
        fft_vals = np.abs(np.fft.fft(filtered_signal))
        
        # Average and Max
        features.append(np.mean(fft_vals))
        features.append(np.max(fft_vals))
        
    return features

def main():
    print("Loading ML model Model and Scaler...")
    try:
        knn = joblib.load("emg_knn_model.pkl")
        scaler = joblib.load("emg_scaler.pkl")
    except FileNotFoundError:
        print("Error: Could not find the .pkl files. Make sure they are in this folder!")
        return

    # 3. Connect to the Ganglion
    SERIAL_PORT = "/dev/cu.usbmodem11"
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    try:
        print("Connecting to Ganglion...")
        board.prepare_session()
        board.start_stream()
        print("\n--- LIVE STREAM STARTED ---")
        print("Perform your Table Presses! Press CTRL+C to stop.\n")
        
        # Let the buffer fill up with at least 1 second of data first
        time.sleep(1)

        while True:
            # 4. The Sliding Window
            # get_current_board_data(200) grabs the LATEST 200 samples (1 second) 
            # without deleting them from the board's internal buffer.
            data = board.get_current_board_data(200)
            
            # Make sure we actually got a full second of data
            if data.shape[1] >= 200:
                emg_channels = BoardShim.get_emg_channels(board_id)
                emg_data = data[emg_channels]
                
                # 5. Extract, Scale, and Predict
                live_features = extract_features(emg_data)
                
                # Machine learning models expect a 2D array, so we wrap our list in another list
                features_2d = [live_features] 
                
                scaled_features = scaler.transform(features_2d)
                prediction = knn.predict(scaled_features)[0]
                
                # Print the live prediction to the terminal
                print(f"Live Prediction: >>> {prediction.upper()} <<<")
            
            # Wait 0.5 seconds before grabbing the next 1-second window
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Session closed safely.")

if __name__ == "__main__":
    main()