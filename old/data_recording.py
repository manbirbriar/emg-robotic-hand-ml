import time
import argparse
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def main():
    # 1. Set up the connection parameters for the Ganglion
    # NOTE: You MUST change 'COM3' to the actual port of your Ganglion Bluetooth Dongle!
    # On Mac/Linux, this looks like '/dev/ttyACM0' or '/dev/cu.usbmodem...'
    SERIAL_PORT = "/dev/cu.usbmodem11"
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    
    # Ganglion Board ID is 1
    board_id = BoardIds.GANGLION_BOARD.value

    # 2. Initialize the board
    board = BoardShim(board_id, params)
    
    # 3. Ask the user what finger they are recording for the file name
    finger_name = input("Which finger are you recording? (e.g., index, thumb, rest): ")
    sample_number = input("What sample number is this? (e.g., 1, 2, 3): ")
    filename = f"{finger_name}_sample_{sample_number}.csv"

    try:
        print("\nConnecting to Ganglion...")
        board.prepare_session()
        
        print("\nREADY! The 12-second recording will start in 3 seconds...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        
        # 4. Start streaming data into the background buffer
        board.start_stream()
        print("\n--- RECORDING STARTED ---")
        print("0-3s: REST")
        time.sleep(3)
        print("3-9s: FLEX AND HOLD!")
        time.sleep(6)
        print("9-12s: REST")
        time.sleep(3)
        print("--- RECORDING FINISHED ---")
        
        # 5. Stop the stream and pull the data from the buffer
        board.stop_stream()
        data = board.get_board_data() 
        
        # 6. Get the specific rows that contain the 4 EMG channels
        emg_channels = BoardShim.get_emg_channels(board_id)
        
        # Keep only the rows containing our 4 EMG channels (discarding timestamps/accelerometer data for now)
        emg_data = data[emg_channels]
        
        # 7. Save to CSV
        # We transpose the data (.T) so each column is a channel and each row is a point in time
        df = pd.DataFrame(emg_data.T, columns=["Channel_1", "Channel_2", "Channel_3", "Channel_4"])
        df.to_csv(filename, index=False)
        print(f"\nSuccess! Data saved to {filename}")

    finally:
        # Always release the session so the Bluetooth port isn't locked up
        if board.is_prepared():
            board.release_session()

if __name__ == "__main__":
    main()