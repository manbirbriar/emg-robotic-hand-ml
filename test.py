import time
import numpy as np
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

SERIAL_PORT = "/dev/cu.usbmodem11"
RECORD_SECONDS = 20
OUT_CSV = "ganglion_20s.csv"

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    # Optional but recommended if you know it:
    # params.mac_address = "XX:XX:XX:XX:XX:XX"

    board_id = BoardIds.GANGLION_BOARD
    board = BoardShim(board_id, params)

    try:
        print("Preparing session...")
        board.prepare_session()

        print("Starting stream...")
        board.start_stream(45000)

        print(f"Recording for {RECORD_SECONDS} seconds...")
        time.sleep(RECORD_SECONDS)

        print("Stopping stream...")
        board.stop_stream()

        # Get all data collected
        data = board.get_board_data()
        print("Data shape (rows, samples):", data.shape)

        # Save raw data
        DataFilter.write_file(data, OUT_CSV, "w")
        print(f"Saved raw data to {OUT_CSV}")

        # Figure out rows for plotting
        eeg_rows = BoardShim.get_eeg_channels(board_id)  # Ganglion has 4
        fs = BoardShim.get_sampling_rate(board_id)
        n_samples = data.shape[1]
        t = np.arange(n_samples) / fs

        # Plot each channel
        plt.figure()
        for i, row in enumerate(eeg_rows, start=1):
            plt.plot(t, data[row, :], label=f"Ch{i}")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.title("Ganglion: 20s recording (4 channels)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    finally:
        # Always release session even if something errors
        try:
            board.release_session()
        except Exception:
            pass

if __name__ == "__main__":
    main()
