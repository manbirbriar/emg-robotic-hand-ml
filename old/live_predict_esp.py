import time
import numpy as np
import pandas as pd
import joblib
import serial
from scipy.signal import butter, filtfilt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ==========================================
# 1. SIGNAL PROCESSING FUNCTIONS
# ==========================================
def butter_bandpass_filter(data, lowcut=10.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_features(emg_data_matrix):
    features = []
    # Loop through all 4 channels in the 1-second window
    for i in range(4):
        signal = emg_data_matrix[i]
        
        # Filter and transform
        filtered_signal = butter_bandpass_filter(signal)
        fft_vals = np.abs(np.fft.fft(filtered_signal))
        
        # Calculate Average and Max
        features.append(np.mean(fft_vals))
        features.append(np.max(fft_vals))
        
    return features

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    print("Loading ML model Model and Scaler...")
    try:
        knn = joblib.load("emg_knn_model.pkl")
        scaler = joblib.load("emg_scaler.pkl")
    except FileNotFoundError:
        print("Error: Could not find the .pkl files. Run the training script first!")
        return

    # --- ESP32 SERIAL SETUP ---
    # Update this to your exact ESP32 port (check Arduino IDE > Tools > Port)
    esp32_port = '/dev/tty.usbserial-1130' 
    baud_rate = 115200
    
    try:
        esp32 = serial.Serial(esp32_port, baud_rate, timeout=0.1)
        print(f"Successfully connected to ESP32 on {esp32_port}")
        time.sleep(2) # Give the ESP32 a moment to reset after connecting
    except serial.SerialException:
        print(f"ERROR: Could not connect to ESP32 on {esp32_port}. Check your USB cable!")
        return

    # Command translation map for the ESP32
    command_map = {
        "rest": b'0',
        "thumb": b'T',
        "index": b'I',
        "middle": b'M',
        "ring": b'R',
        "pinky": b'P'
    }

    # --- GANGLION SETUP ---
    # Update this to your Ganglion Bluetooth dongle port
    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbmodem11' 
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    try:
        print(f"Connecting to Ganglion on {params.serial_port}...")
        board.prepare_session()
        board.start_stream()
        print("\n--- LIVE STREAM STARTED ---")
        print("Perform your Table Presses! Press CTRL+C to stop.\n")
        
        # Wait 1 second to let the buffer fill with initial data
        time.sleep(1)

        while True:
            # Grab the latest 200 samples (1 second of data at 200Hz)
            data = board.get_current_board_data(200)
            
            # Ensure we have a full window before predicting
            if data.shape[1] >= 200:
                emg_channels = BoardShim.get_emg_channels(board_id)
                emg_data = data[emg_channels]
                
                # Extract features from the live data
                live_features = extract_features(emg_data)
                
                # Scale features and predict
                scaled_features = scaler.transform([live_features])
                prediction = knn.predict(scaled_features)[0]
                
                print(f"Live Prediction: >>> {prediction.upper()} <<<")
                
                # Send the corresponding byte to the ESP32
                esp32_command = command_map.get(prediction, b'0')
                esp32.write(esp32_command)
            
            # Wait 0.5 seconds before evaluating the next window
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping stream and closing ports...")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        if esp32.is_open:
            # Send a final 'rest' command to open the hand before disconnecting
            esp32.write(b'0')
            esp32.close()
        print("System shut down safely.")

if __name__ == "__main__":
    main()