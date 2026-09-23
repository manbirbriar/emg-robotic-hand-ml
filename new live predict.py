import time
import numpy as np
import joblib
from collections import deque, Counter 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from filter_feature import apply_filters, extract_features

def main():
    SERIAL_PORT = "/dev/cu.usbmodem11" # Update if needed
    
    try:
        model = joblib.load('emg_knn_model.pkl')
        print("Loaded ML Model successfully.")
    except FileNotFoundError:
        print("Model not found! Run model_creation.py first.")
        return

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    prediction_buffer = deque(maxlen=4)

    try:
        board.prepare_session()
        board.start_stream()
        print("\n=== LIVE PREDICTION STARTED ===")
        print("Flex a finger and hold for 2 seconds...")
        
        # Give the board plenty of time to fill the initial 2-second buffer
        time.sleep(2.5) 

        while True:
            # PEEK at the latest 400 samples (2 seconds) without emptying the buffer
            data = board.get_current_board_data(400) 
            
            if data.shape[1] < 400:
                time.sleep(0.1)
                continue
                
            emg_channels = BoardShim.get_emg_channels(board_id)
            raw_emg = data[emg_channels].T 

            # Filter the 2-second block
            filtered_emg = apply_filters(raw_emg)
            
            # Extract features from ONLY the most recent 500ms (100 samples)
            recent_window = filtered_emg[-100:, :]
            features = extract_features(recent_window)

            # Make a raw prediction
            raw_pred = model.predict([features])[0]
            prediction_buffer.append(raw_pred)

            # --- DEBUG PRINT---
            print(f"Prediction: {list(prediction_buffer)}")

            # 2. THE NEW RELAXED MAJORITY VOTE: 3 out of 4 match
            if len(prediction_buffer) == 4:
                # Tally up the predictions to find the most common one
                tally = Counter(prediction_buffer)
                most_common_prediction, count = tally.most_common(1)[0]
                
                # If that prediction shows up 3 or more times, trigger the servo
                if count >= 3:
                    confirmed_movement = most_common_prediction
                    
                    # Command is printed universally, treating REST as an active command
                    print(f"\n>>> SERVO COMMAND: {confirmed_movement.upper()} <<<\n")
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping live prediction...")
    finally:
        if board.is_prepared():
            board.release_session()

if __name__ == "__main__":
    main()