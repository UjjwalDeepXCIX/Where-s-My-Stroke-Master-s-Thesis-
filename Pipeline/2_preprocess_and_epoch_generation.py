import mne
import os
import numpy as np
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from psutil import Process
from scipy.signal import welch

# Set up logging
log_dir = r"C:/UoM Sem-2/Research_work/src/logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"preprocess_{timestamp}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output folders
epoch_dir = r"C:/UoM Sem-2/Research_work/src/epochs"
plot_dir = r"C:/UoM Sem-2/Research_work/src/plots/processed_plots"
os.makedirs(epoch_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load participants.tsv
participants_path = "C:/UoM Sem-2/Research_work/Datasets/Dataset/FIgshare/metadata/participants.tsv"
try:
    participants = pd.read_csv(participants_path, sep='\t')
    participant_dict = participants.set_index('Participant_ID').to_dict('index')
    logger.info("Loaded participants.tsv with stroke metadata")
except Exception as e:
    logger.error(f"Failed to load participants.tsv: {str(e)}. Using default channels.")
    participant_dict = {}

# Define standard channels
standard_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
]

def normalize_channel_name(channel):
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

def get_relevant_channels(participant_id):
    if participant_id not in participant_dict:
        return standard_channels
    stroke_location = participant_dict[participant_id]['StrokeLocation'].lower()
    channels = standard_channels.copy()
    if 'pons' in stroke_location or 'basal ganglia' in stroke_location:
        channels = ['C3', 'C4', 'CZ'] + [ch for ch in channels if ch not in ['C3', 'C4', 'CZ']]
    if 'frontal' in stroke_location:
        channels = ['FP1', 'FP2', 'F3', 'F4'] + [ch for ch in channels if ch not in ['FP1', 'FP2', 'F3', 'F4']]
    return channels

def get_memory_usage():
    process = Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def plot_sample_epoch(epochs, filename, group, subset, is_synthetic=False):
    try:
        data = epochs.get_data()[0]
        times = np.linspace(0, 10, data.shape[1])
        plt.figure(figsize=(10, 8))
        for i, (channel_data, ch_name) in enumerate(zip(data, epochs.ch_names)):
            plt.plot(times, channel_data + i * np.std(channel_data) * 2, label=ch_name)
        plt.title(f"Sample Epoch: {os.path.basename(filename)} ({group} {subset}{' Synthetic' if is_synthetic else ''})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (offset)")
        plt.legend(loc='upper right')
        plot_file = os.path.join(plot_dir, f"{os.path.basename(filename).replace('.edf', '').replace('.npy', '')}_epoch.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Generated plot: {plot_file}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot for {filename}: {str(e)}")

def remove_eye_artifacts_regression(raw, frontal_channels=['FP1', 'FP2']):
    try:
        raw.filter(l_freq=1.0, h_freq=30.0)
        data = raw.get_data()
        frontal_idx = [i for i, ch in enumerate(raw.ch_names) if normalize_channel_name(ch) in frontal_channels]
        if frontal_idx:
            ref_signal = np.mean(data[frontal_idx], axis=0)
            for i in range(data.shape[0]):
                beta = np.corrcoef(data[i], ref_signal)[0, 1] if np.std(data[i]) > 0 else 0
                data[i] -= beta * ref_signal * 0.2
        raw._data = data
        logger.info(f"Enhanced regression-based artifact correction using {frontal_channels}")
    except Exception as e:
        logger.warning(f"Regression correction failed: {str(e)}. Data unchanged.")

def preprocess_eeg(edf_file, group, subset, is_synthetic=False):
    try:
        logger.info(f"Processing {group} {subset} file: {edf_file} {'(synthetic)' if is_synthetic else ''}")
        logger.info(f"Memory usage before loading: {get_memory_usage():.2f} MB")
        participant_id = os.path.basename(edf_file).split('_')[0] if group == 'Stroke' and not is_synthetic else None
        if group == 'Normal' and not is_synthetic:
            logger.info(f"Processing non-stroke file {edf_file} with default metadata")
        
        if is_synthetic:
            epochs_data = np.load(edf_file)
            raw = mne.io.RawArray(epochs_data[0].T, mne.create_info(standard_channels[:epochs_data.shape[1]], sfreq=256.0, ch_types='eeg'))
            epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=2.0, preload=True)
            if len(epochs) == 0:
                logger.error(f"No epochs created for {edf_file}")
                return False
            logger.info(f"Loaded {len(epochs)} synthetic epochs for {edf_file}")
        else:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='error')
            logger.info(f"Memory usage after loading: {get_memory_usage():.2f} MB")
            
            if '' in raw.ch_names:
                raw.drop_channels([''])
            
            selected_channels = get_relevant_channels(participant_id) if group == 'Stroke' else standard_channels
            available_channels = [ch for ch in raw.ch_names if normalize_channel_name(ch) in selected_channels]
            if len(available_channels) < 10:
                logger.error(f"Too few standard channels in {edf_file}: {available_channels}")
                return False
            raw.pick_channels(available_channels)
            logger.info(f"Selected channels: {raw.ch_names}")
            
            channels_to_drop = [ch for ch in raw.ch_names if ch in ['HEOL', 'HEOR']]
            if channels_to_drop:
                raw.drop_channels(channels_to_drop)
                logger.info(f"Dropped EOG channels: {channels_to_drop}")
            
            raw.resample(256)
            logger.info(f"Memory usage after resampling: {get_memory_usage():.2f} MB")
            
            raw.filter(l_freq=0.5, h_freq=40.0, method='fir')
            if participant_id and participant_id in participant_dict:
                nihss = participant_dict[participant_id]['NIHSS']
                logger.info(f"NIHSS for {participant_id}: {nihss}")
            
            raw.notch_filter(freqs=60, verbose='error')
            
            if group == 'Stroke' and 'HEOL' not in raw.ch_names and 'HEOR' not in raw.ch_names:
                try:
                    ica = mne.preprocessing.ICA(n_components=min(len(raw.ch_names)-1, 15), random_state=42)
                    ica.fit(raw)
                    ica.exclude = []
                    ica.apply(raw)
                    logger.info(f"Applied ICA for {edf_file}")
                except Exception as e:
                    logger.warning(f"ICA failed for {edf_file}: {str(e)}. Applying regression-based artifact correction.")
                    remove_eye_artifacts_regression(raw)
            else:
                logger.warning(f"No EOG channels in {edf_file}. Applying regression-based artifact correction.")
                remove_eye_artifacts_regression(raw)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=2.0, preload=True)
            # enforce 250 Hz + CAR
            raw = thesis_resample_and_car(raw)

            # quality control on epochs (SNR & variance-z)
            raw, epochs = apply_thesis_preproc_and_qc(raw, epochs)

            if len(epochs) == 0:
                logger.error(f"No epochs created for {edf_file}")
                return False
            logger.info(f"Created {len(epochs)} epochs for {edf_file}")
        
        plot_sample_epoch(epochs, edf_file, group, subset, is_synthetic)
        
        output_dir = os.path.join(epoch_dir, subset.lower(), group.lower())
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(edf_file).replace('.edf', '').replace('.npy', '')
        epoch_file = os.path.join(output_dir, f"{group.lower()}_{subset.lower()}_{base_name}_epochs.npy")
        np.save(epoch_file, epochs.get_data())
        logger.info(f"Saved epochs: {epoch_file}")
        
        risk_score = 0.9 if group == 'Stroke' else 0.1
        risk_file = os.path.join(output_dir, f"{group.lower()}_{subset.lower()}_{base_name}_risk.npy")
        np.save(risk_file, np.array([risk_score] * len(epochs)))
        logger.info(f"Saved risk scores: {risk_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {edf_file}: {str(e)}")
        return False

def main():
    logger.info("Starting EEG preprocessing")
    data_dirs = {
        'Stroke': {
            'Train': "C:/UoM Sem-2/Research_work/Datasets/Dataset/FIgshare/train/stroke",
            'Eval': "C:/UoM Sem-2/Research_work/Datasets/Dataset/FIgshare/eval/stroke",
            'Synthetic': os.path.join(epoch_dir, "synthetic", "stroke")
        },
        'Normal': {
            'Train': "C:/UoM Sem-2/Research_work/Datasets/Dataset/TUHEEG/train/normal",
            'Eval': "C:/UoM Sem-2/Research_work/Datasets/Dataset/TUHEEG/eval/normal",
            'Synthetic': os.path.join(epoch_dir, "synthetic", "normal")
        }
    }
    total_files = 0
    processed_files = 0
    for group, subsets in data_dirs.items():
        for subset, dir_path in subsets.items():
            is_synthetic = subset == 'Synthetic'
            file_ext = '.npy' if is_synthetic else '.edf'
            if os.path.exists(dir_path):
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]
                logger.info(f"Processing {subset} {group} Files ({len(files)} files)")
                for file in files:
                    total_files += 1
                    if preprocess_eeg(file, group, subset.lower(), is_synthetic):
                        processed_files += 1
            else:
                logger.warning(f"Directory not found: {dir_path}")
    logger.info(f"Preprocessing complete. Processed {processed_files}/{total_files} files")
    if processed_files < total_files:
        logger.warning("Some files failed to process. Check logs for details.")

# ====================== THESIS PATCH (append-only) ======================
import numpy as _np
from scipy.signal import welch as _welch

# Config toggles (no deletions to your code)
_THESIS_RESAMPLE_HZ = 250
_THESIS_APPLY_CAR = True
_THESIS_SNR_DB_MIN = 0.0      # keep epochs with SNR >= 0 dB
_THESIS_VAR_Z_MAX = 3.0       # reject epochs whose channel variance z-score exceeds ±3

def thesis_resample_and_car(raw):
    """Force 250 Hz + common-average reference, without removing existing steps."""
    try:
        if abs(raw.info.get("sfreq", _THESIS_RESAMPLE_HZ) - _THESIS_RESAMPLE_HZ) > 0.1:
            raw.resample(_THESIS_RESAMPLE_HZ, npad="auto")
    except Exception as e:
        try: logger.warning(f"[THESIS] Resample to 250 Hz failed: {e}")  # noqa: F821
        except: pass
    if _THESIS_APPLY_CAR:
        try:
            raw.set_eeg_reference(ref_channels="average", projection=False)
        except Exception as e:
            try: logger.warning(f"[THESIS] CAR failed: {e}")  # noqa: F821
            except: pass
    return raw

def _bandpower_ratio_snr(x, fs):
    """Estimate SNR as bandpower(1–40 Hz) / bandpower(45–100 Hz) in dB."""
    f, pxx = _welch(x, fs=fs, window="hamming", nperseg=int(2*fs), noverlap=int(fs), axis=-1)
    sig = _np.mean(pxx[..., (f >= 1) & (f < 40)], axis=-1)
    noise = _np.mean(pxx[..., (f >= 45) & (f <= 100)], axis=-1)
    snr = 10.0 * _np.log10((_np.maximum(sig, 1e-12)) / (_np.maximum(noise, 1e-12)))
    # Average across channels
    return _np.mean(snr)

def thesis_reject_epochs_by_qc(epochs, fs=_THESIS_RESAMPLE_HZ, snr_db_min=_THESIS_SNR_DB_MIN, var_z_max=_THESIS_VAR_Z_MAX):
    """Keep only epochs meeting SNR and variance-z thresholds. No deletions to existing epochs logic."""
    try:
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    except Exception:
        # If not an MNE Epochs object, assume ndarray already
        data = _np.asarray(epochs)
    n_ep = data.shape[0]
    keep_idx = []
    for i in range(n_ep):
        ep = data[i]
        snr_db = _bandpower_ratio_snr(ep, fs)
        ch_var = ep.var(axis=-1)
        z = (ch_var - ch_var.mean()) / (ch_var.std() + 1e-12)
        if (snr_db >= snr_db_min) and (_np.all(_np.abs(z) <= var_z_max)):
            keep_idx.append(i)
    try:
        return epochs[keep_idx] if hasattr(epochs, "__getitem__") else data[keep_idx]
    except Exception:
        return data[keep_idx]

# OPTIONAL: helper to apply right after your existing preprocessing
def apply_thesis_preproc_and_qc(raw, epochs):
    raw = thesis_resample_and_car(raw)
    try:
        fs = raw.info["sfreq"]
    except Exception:
        fs = _THESIS_RESAMPLE_HZ
    epochs_kept = thesis_reject_epochs_by_qc(epochs, fs=fs)
    try:
        logger.info(f"[THESIS] QC kept {len(epochs_kept)} / {len(epochs)} epochs")  # noqa: F821
    except Exception:
        pass
    return raw, epochs_kept
# ==================== /THESIS PATCH (append-only) =======================

if __name__ == "__main__":
    main()

