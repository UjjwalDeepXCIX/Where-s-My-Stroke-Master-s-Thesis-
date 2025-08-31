import mne
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
import time

# Set up logging
log_dir = r"C:/UoM Sem-2/Research_work/src/logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"load_data_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output folder for plots
plot_dir = r"C:/UoM Sem-2/Research_work/src/plots/base_plots"
os.makedirs(plot_dir, exist_ok=True)

# Define standard EEG channels (base names, covering stroke-relevant regions)
standard_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
]

# Define base data directory (from log)
base_data_dir = r"C:/UoM Sem-2/Research_work/Datasets/Dataset"

# Define directory structure
data_dirs = {
    'Stroke': {
        'Train': os.path.join(base_data_dir, 'FIgshare', 'train', 'stroke'),
        'Eval': os.path.join(base_data_dir, 'FIgshare', 'eval', 'stroke')
    },
    'Normal': {
        'Train': os.path.join(base_data_dir, 'TUHEEG', 'train', 'normal'),
        'Eval': os.path.join(base_data_dir, 'TUHEEG', 'eval', 'normal')
    }
}

def normalize_channel_name(channel):
    """Normalize channel name by removing prefixes/suffixes and converting to uppercase."""
    if not channel or channel == '':
        return ''
    channel = channel.upper()
    for prefix in ['EEG', 'EEG ']:
        if channel.startswith(prefix):
            channel = channel[len(prefix):]
    for suffix in ['-REF', '-LE', '-AVG']:
        if channel.endswith(suffix):
            channel = channel[:-len(suffix)]
    return channel.strip()

def get_edf_files(directory, group, subset):
    """Get all .edf files from a directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    edf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.edf')]
    logger.info(f"Found {len(edf_files)} .edf files in {group} {subset}: {directory}")
    return edf_files

# Build file_groups dynamically
file_groups = {
    group: {
        subset: get_edf_files(path, group, subset)
        for subset, path in subsets.items()
    }
    for group, subsets in data_dirs.items()
}

def load_eeg_file(edf_file, group, subset):
    """Load and verify an EEG file, generate a plot."""
    start_time = time.time()
    logger.info(f"Loading {group} {subset} file: {edf_file}")
    
    if not os.path.exists(edf_file):
        logger.error(f"File not found: {edf_file}")
        return False
    
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False, verbose='error')
        logger.info(f"Successfully loaded: {edf_file}")
        logger.info(f"Channels: {raw.ch_names}")
        logger.info(f"Sampling rate: {raw.info['sfreq']} Hz")
        logger.info(f"Duration: {raw.times[-1]} seconds")
        
        # Check for invalid sampling rate
        if raw.info['sfreq'] <= 0:
            logger.error(f"Invalid sampling rate {raw.info['sfreq']} Hz in {edf_file}")
            return False
        
        # Normalize channel names for comparison
        available_channels = []
        for ch in raw.ch_names:
            norm_ch = normalize_channel_name(ch)
            if norm_ch in [normalize_channel_name(sc) for sc in standard_channels]:
                available_channels.append(ch)  # Keep original name
        logger.info(f"Found {len(available_channels)}/{len(standard_channels)} standard channels: {available_channels}")
        
        # Minimum 10 channels for stroke-relevant coverage
        if len(available_channels) < 10:
            logger.error(f"Too few standard EEG channels in {edf_file}: {available_channels}")
            logger.info(f"All channels in file: {raw.ch_names}")
            return False
        
        # Drop empty channel if present
        if '' in raw.ch_names:
            logger.info(f"Dropping empty channel name in {edf_file}")
            raw.drop_channels([''])
        
        # Get data for first 10 seconds, up to 10 channels (prioritize standard ones)
        channel_indices = [raw.ch_names.index(ch) for ch in available_channels[:10] if ch in raw.ch_names]
        data, times = raw[channel_indices, :int(10 * raw.info['sfreq'])]
        channels = [raw.ch_names[i] for i in channel_indices]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        for i, (channel_data, channel_name) in enumerate(zip(data, channels)):
            plt.plot(times, channel_data + i * np.std(channel_data) * 2, label=channel_name)
        plt.title(f"EEG Signal: {os.path.basename(edf_file)} ({group} {subset})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (offset)")
        plt.legend(loc='upper right')
        
        plot_file = os.path.join(plot_dir, f"{os.path.basename(edf_file).replace('.edf', '')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file}")
        plt.close()
        
        logger.info(f"Processed {edf_file} in {time.time() - start_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error loading {edf_file}: {str(e)}")
        return False

def main():
    logger.info("Starting EEG data loading")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Debug: List base directory contents
    if os.path.exists(base_data_dir):
        logger.info(f"Contents of {base_data_dir}: {os.listdir(base_data_dir)}")
    else:
        logger.error(f"Base data directory not found: {base_data_dir}")
    
    total_files = 0
    processed_files = 0
    
    for group_name, subsets in file_groups.items():
        logger.info(f"Processing {group_name} EEG Files")
        for subset_name, files in subsets.items():
            logger.info(f"Processing {subset_name} {group_name} Files ({len(files)} files)")
            for edf_file in files:
                total_files += 1
                if load_eeg_file(edf_file, group_name, subset_name):
                    processed_files += 1
                    logger.info(f"[THESIS] Label map {group_name}/{subset_name} -> {get_label(group_name, subset_name, edf_file)}")

    
    logger.info(f"Loading complete. Processed {processed_files}/{total_files} files")
    if processed_files < total_files:
        logger.warning("Some files failed to load. Check logs for details.")
    if total_files == 0:
        logger.error("No EDF files found. Verify dataset directories and file names.")

    
# ====================== THESIS PATCH (append-only) ======================
# Enforce dataset→label mapping to match thesis w/out deleting existing logic.
# If your file already defines `get_label`, we shadow it safely.

try:
    _ORIG_GET_LABEL = get_label  # type: ignore[name-defined]
except Exception:
    _ORIG_GET_LABEL = None

def _thesis_dataset_label_mapper(group_name: str, subset_name: str = "", path: str = "") -> int:
    g = (group_name or "") + " " + (subset_name or "") + " " + (path or "")
    g_lower = g.lower()
    # Thesis: TUH = stroke(1), Figshare motor-imagery = normal/control(0)
    if "tuh" in g_lower:
        return 1  # stroke
    if "figshare" in g_lower or "motor-imagery" in g_lower or "bci" in g_lower:
        return 0  # control
    # Fallback to original mapper if present
    if _ORIG_GET_LABEL is not None:
        try:
            return _ORIG_GET_LABEL(group_name, subset_name)  # type: ignore[misc]
        except Exception:
            pass
    # Default conservative fallback
    return 0

# Public alias used by the rest of the script:
get_label = _thesis_dataset_label_mapper  # type: ignore[assignment]
# ==================== /THESIS PATCH (append-only) =======================

if __name__ == "__main__":
    main()
