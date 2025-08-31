import numpy as np
import pandas as pd
import os
import tensorflow as tf
import logging
from datetime import datetime
from scipy.signal import welch
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Reshape, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sys
import io

# Set up logging with UTF-8 encoding
log_dir = r"C:/UoM Sem-2/Research_work/src/logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"generate_synthetic_{timestamp}.log")

# Create a custom StreamHandler with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)
        self.stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories
epoch_dir = r"C:/UoM Sem-2/Research_work/src/epochs"
synthetic_dir = os.path.join(epoch_dir, "synthetic")
os.makedirs(synthetic_dir, exist_ok=True)
plot_dir = r"C:/UoM Sem-2/Research_work/src/plots/synthetic"
os.makedirs(plot_dir, exist_ok=True)

# Load participants.tsv
participants_path = "C:/UoM Sem-2/Research_work/Datasets/Dataset/FIgshare/metadata/participants.tsv"
try:
    participants = pd.read_csv(participants_path, sep='\t')
    participant_dict = participants.set_index('Participant_ID').to_dict('index')
    logger.info("Loaded participants.tsv with stroke metadata")
except Exception as e:
    logger.error(f"Failed to load participants.tsv: {str(e)}")
    participant_dict = {}

# Load real epochs
def load_real_epochs():
    epoch_files = []
    for subset in ['train']:
        for group in ['stroke', 'normal']:
            subdir = os.path.join(epoch_dir, subset, group)
            if os.path.exists(subdir):
                files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('_epochs.npy')]
                epoch_files.extend(files)
                logger.info(f"Found {len(files)} epoch files in {subdir}")
            else:
                logger.warning(f"Directory not found: {subdir}")
    
    if not epoch_files:
        logger.error("No epoch files found in epoch_dir. Please run preprocess_eeg.py first to generate epoch files.")
        raise FileNotFoundError("No epoch files found in epoch_dir")
    
    real_epochs = []
    real_labels = []
    real_metadata = []
    for file in sorted(epoch_files):
        try:
            logger.info(f"Loading file: {file}")
            epochs = np.load(file)
            risk_file = file.replace('_epochs.npy', '_risk.npy')
            risks = np.load(risk_file)
            participant_id = os.path.basename(file).split('_')[2] if 'stroke' in file.lower() else 'normal'
            if participant_id == 'normal':
                metadata = {'NIHSS': 0, 'StrokeLocation': 'none'}
                logger.info(f"Assigned default metadata for non-stroke file: {file}")
            else:
                metadata = participant_dict.get(participant_id, {'NIHSS': 0, 'StrokeLocation': 'none'})
            nihss = metadata['NIHSS'] / 11.0
            stroke_loc = 1.0 if 'pons' in metadata['StrokeLocation'].lower() or 'basal ganglia' in metadata['StrokeLocation'].lower() else 0.0
            real_epochs.append(epochs)
            real_labels.extend(risks)
            # Include risk in metadata for condition tensor
            real_metadata.extend([[risk, nihss, stroke_loc] for risk in risks])
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not real_epochs:
        logger.error("No valid epoch data loaded. Check epoch files and logs.")
        raise ValueError("No valid epoch data loaded")
    
    real_epochs = np.concatenate(real_epochs, axis=0)
    real_labels = np.array(real_labels)
    real_metadata = np.array(real_metadata)
    logger.info(f"Loaded {len(real_epochs)} epochs, labels shape: {real_labels.shape}, metadata shape: {real_metadata.shape}")
    return real_epochs, real_labels, real_metadata

# Define GAN
n_channels = 19
n_samples = 2560
latent_dim = 100
condition_dim = 3  # [label, nihss, stroke_loc]

def build_generator():
    noise_input = Input(shape=(latent_dim,), name='noise_input')
    condition_input = Input(shape=(condition_dim,), name='condition_input')
    x = Concatenate(name='concatenate')([noise_input, condition_input])
    logger.info(f"After Concatenate: shape = {(latent_dim + condition_dim,)}")
    x = Dense(256, activation='relu', name='dense_1')(x)
    logger.info(f"After Dense_1: shape = {(256,)}")
    x = Reshape((16, 16), name='reshape_1')(x)
    logger.info(f"After Reshape_1: shape = {(16, 16)}")
    x = LSTM(128, return_sequences=True, name='lstm_1')(x)
    logger.info(f"After LSTM_1: shape = {(16, 128)}")
    x = LSTM(64, return_sequences=False, name='lstm_2')(x)
    logger.info(f"After LSTM_2: shape = {(64,)}")
    x = Dense(n_channels * n_samples, name='dense_2')(x)
    logger.info(f"After Dense_2: shape = {(n_channels * n_samples,)}")
    output = Reshape((n_channels, n_samples), name='reshape_output')(x)
    logger.info(f"After Reshape_output: shape = {(n_channels, n_samples)}")
    model = Model([noise_input, condition_input], output, name='generator')
    model.summary(print_fn=lambda x: logger.info(x))
    return model

def build_discriminator():
    eeg_input = Input(shape=(n_channels, n_samples), name='eeg_input')
    condition_input = Input(shape=(condition_dim,), name='condition_input')
    x = LSTM(64, return_sequences=True, name='lstm_1')(eeg_input)
    x = LSTM(32, name='lstm_2')(x)
    x = Concatenate(name='concatenate')([x, condition_input])
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model([eeg_input, condition_input], output, name='discriminator')
    model.summary(print_fn=lambda x: logger.info(x))
    return model

# Compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
discriminator.trainable = False
gan_input_noise = Input(shape=(latent_dim,), name='gan_noise_input')
gan_input_condition = Input(shape=(condition_dim,), name='gan_condition_input')
gan_output = generator([gan_input_noise, gan_input_condition])
gan_output_disc = discriminator([gan_output, gan_input_condition])
gan = Model([gan_input_noise, gan_input_condition], gan_output_disc, name='gan')
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
gan.summary(print_fn=lambda x: logger.info(x))

# Train GAN
def train_gan(epochs_data, labels, metadata, epochs=50, batch_size=32):
    logger.info("Training GAN...")
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    if metadata.shape[1] != condition_dim:
        logger.error(f"Metadata shape {metadata.shape} does not match condition_dim {condition_dim}")
        raise ValueError(f"Expected metadata with {condition_dim} features, got {metadata.shape[1]}")
    for epoch in range(epochs):
        idx = np.random.randint(0, epochs_data.shape[0], batch_size)
        real_epochs = epochs_data[idx]
        real_conditions = metadata[idx]
        real_conditions = thesis_condition_hook(real_conditions)   # ← ADD
        logger.debug(f"Batch {epoch}: real_conditions shape = {real_conditions.shape}")
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_conditions = real_conditions
        fake_conditions = thesis_condition_hook(fake_conditions) 
        fake_epochs = generator.predict([noise, fake_conditions], verbose=0)
        d_loss_real = discriminator.train_on_batch([real_epochs, real_conditions], real)
        d_loss_fake = discriminator.train_on_batch([fake_epochs, fake_conditions], fake)
        g_loss = gan.train_on_batch([noise, fake_conditions], real)
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: D Loss Real: {d_loss_real:.4f}, D Loss Fake: {d_loss_fake:.4f}, G Loss: {g_loss:.4f}")
    logger.info("GAN training complete")

# Generate synthetic data
def generate_synthetic_data(n_samples_per_class=1000):
    logger.info("Generating synthetic data...")
    synthetic_epochs = []
    synthetic_labels = []
    synthetic_weights = []
    for label, group in [(0.9, 'stroke'), (0.1, 'normal')]:
        n_samples = n_samples_per_class
        noise = np.random.normal(0, 1, (n_samples, latent_dim))
        if group == 'stroke':
            participant_ids = list(participant_dict.keys())
            sampled_ids = np.random.choice(participant_ids, n_samples, replace=True)
            conditions = np.array([[label, participant_dict[pid]['NIHSS'] / 11.0,
                                   1.0 if 'pons' in participant_dict[pid]['StrokeLocation'].lower() or 'basal ganglia' in participant_dict[pid]['StrokeLocation'].lower() else 0.0]
                                  for pid in sampled_ids])
            weights = [1.0 + (participant_dict[pid]['NIHSS'] / 11.0) * (1.2 if 'pons' in participant_dict[pid]['StrokeLocation'].lower() or 'basal ganglia' in participant_dict[pid]['StrokeLocation'].lower() else 1.0)
                       for pid in sampled_ids]
        else:
            conditions = np.array([[label, 0.0, 0.0]] * n_samples)
            weights = [1.0] * n_samples
            logger.info("Generating synthetic non-stroke data with default metadata: NIHSS=0, StrokeLocation=none")
        logger.info(f"Generating {n_samples} {group} epochs with conditions shape: {conditions.shape}")
        conditions = thesis_condition_hook(conditions)    
        generated_epochs = generator.predict([noise, conditions], verbose=0)
        synthetic_epochs.append(generated_epochs)
        synthetic_labels.extend([label] * n_samples)
        synthetic_weights.extend(weights)
        output_dir = os.path.join(synthetic_dir, group)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(n_samples):
            epoch_file = os.path.join(output_dir, f"synthetic_{group}_train_s{i:04d}_epochs.npy")
            risk_file = os.path.join(output_dir, f"synthetic_{group}_train_s{i:04d}_risk.npy")
            np.save(epoch_file, generated_epochs[i:i+1])
            np.save(risk_file, np.array([label]))
            logger.info(f"Saved synthetic {group} epoch: {epoch_file}")
    return np.concatenate(synthetic_epochs, axis=0), np.array(synthetic_labels), np.array(synthetic_weights)

# Validate synthetic data
def validate_synthetic_data(real_epochs, real_labels, synthetic_epochs, synthetic_labels):
    logger.info("Validating synthetic data...")
    real_psd_stroke = []
    real_psd_normal = []
    synthetic_psd_stroke = []
    synthetic_psd_normal = []
    for epoch, label in zip(real_epochs[:100], real_labels[:100]):
        for ch in range(n_channels):
            freqs, psd = welch(epoch[ch], fs=256.0, nperseg=256, noverlap=128)
            if label > 0.5:
                real_psd_stroke.append(np.mean(psd))
            else:
                real_psd_normal.append(np.mean(psd))
    for epoch, label in zip(synthetic_epochs[:100], synthetic_labels[:100]):
        for ch in range(n_channels):
            freqs, psd = welch(epoch[ch], fs=256.0, nperseg=256, noverlap=128)
            if label > 0.5:
                synthetic_psd_stroke.append(np.mean(psd))
            else:
                synthetic_psd_normal.append(np.mean(psd))
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.hist(real_psd_stroke, bins=50, alpha=0.5, label='Real Stroke PSD')
    plt.hist(synthetic_psd_stroke, bins=50, alpha=0.5, label='Synthetic Stroke PSD')
    plt.title("Stroke PSD Distribution Comparison")
    plt.xlabel("Mean PSD")
    plt.ylabel("Count")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(real_psd_normal, bins=50, alpha=0.5, label='Real Normal PSD')
    plt.hist(synthetic_psd_normal, bins=50, alpha=0.5, label='Synthetic Normal PSD')
    plt.title("Normal PSD Distribution Comparison")
    plt.xlabel("Mean PSD")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "synthetic_psd_comparison.png"))
    plt.close()
    logger.info("Saved PSD comparison plot")

def main():
    try:
        real_epochs, real_labels, real_metadata = load_real_epochs()
        train_gan(real_epochs, real_labels, real_metadata, epochs=50, batch_size=32)
        synthetic_epochs, synthetic_labels, synthetic_weights = generate_synthetic_data(n_samples_per_class=1000)
        validate_synthetic_data(real_epochs, real_labels, synthetic_epochs, synthetic_labels)
        logger.info("Synthetic data generation complete")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


# ====================== THESIS PATCH (append-only) ======================
# Keep your richer conditioning (NIHSS, stroke location) but allow class-only per thesis.
_THESIS_CONDITION_CLASS_ONLY = True

def thesis_condition_hook(cond_vector, class_index: int = 0):
    """
    cond_vector: original conditioning vector you build for the generator.
    If _THESIS_CONDITION_CLASS_ONLY is True, zero out all but the class bit(s).
    This preserves your code path while matching thesis wording.
    """
    import numpy as _np
    cv = _np.asarray(cond_vector).copy()
    if _THESIS_CONDITION_CLASS_ONLY:
        # Keep only the first dimension (assumed 'class'), zero the rest safely.
        if cv.ndim == 1 and cv.size > 1:
            cv[1:] = 0.0
        elif cv.ndim == 2 and cv.shape[1] > 1:
            cv[:, 1:] = 0.0
    return cv

# Example usage inside your training/inference loop (no deletions to your code):
# new_cond = thesis_condition_hook(original_cond_vector)
# feed `new_cond` to your generator instead of the raw one.
# ==================== /THESIS PATCH (append-only) =======================

if __name__ == "__main__":
    main()
