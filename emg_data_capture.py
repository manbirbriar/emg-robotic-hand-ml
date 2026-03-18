import time
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def get_trial_counts(filename, fs=200, duration=5):
    """Counts how many 5-second segments exist for every label in the file."""
    if not os.path.isfile(filename):
        return {}
    try:
        df = pd.read_csv(filename, usecols=["Label"])
        counts = (df["Label"].value_counts() // (fs * duration)).to_dict()
        return counts
    except Exception:
        return {}

def main():
    SERIAL_PORT = "/dev/cu.usbmodem11" 
    MASTER_FILE = "master_emg_data.csv"
    
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    movements = {
        "1": "rest",
        "2": "thumb",
        "3": "index",
        "4": "middle",
        "5": "ring",
        "6": "pinky",
        "7": "hand_close"
    }

    try:
        board.prepare_session()
        
        while True:
            current_counts = get_trial_counts(MASTER_FILE)
            
            print("\n" + "="*35)
            print(f"{'MOVEMENT':<15} | {'TRIALS':<10}")
            print("-" * 35)
            
            for key, val in movements.items():
                count = current_counts.get(val, 0)
                print(f"[{key}] {val.upper():<10} | {count}")
            
            print("-" * 35)
            print("[Q] Quit")
            
            choice = input("\nSelect: ").lower()
            if choice == 'q': break
            if choice not in movements: continue

            label = movements[choice]
            print(f"\n>>> RECORDING: {label.upper()}")
            for i in range(3, 0, -1):
                print(f"{i}..."); time.sleep(1)

            board.start_stream()
            print("\n[PHASE 1] RESTING...")
            time.sleep(5)
            
            if label == "rest":
                print("[PHASE 2] CONTINUING REST...")
            else:
                print("[PHASE 2] FLEX AND HOLD! <<<")
            time.sleep(5)
            
            board.stop_stream()
            data = board.get_board_data(2000) #grab last 2000 from buffer. 
            #Also resets buffer on computer. Note not actaully 2000 records sometimes so need to fix.
            
            emg_channels = BoardShim.get_emg_channels(board_id)
            full_emg = data[emg_channels].T 
            
            df_phase1 = pd.DataFrame(full_emg[:1000, :], columns=["Ch1", "Ch2", "Ch3", "Ch4"])
            df_phase1["Label"] = "rest"
            
            df_phase2 = pd.DataFrame(full_emg[1000:2000, :], columns=["Ch1", "Ch2", "Ch3", "Ch4"])
            df_phase2["Label"] = label
            
            file_exists = os.path.isfile(MASTER_FILE)
            pd.concat([df_phase1, df_phase2]).to_csv(MASTER_FILE, mode='a', index=False, header=not file_exists)
            
            print(f"SUCCESS: Data added to {MASTER_FILE}")

    finally:
        if board.is_prepared(): board.release_session()

if __name__ == "__main__":
    main()
