import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate
import os

def detect_beep_start(recorded_file, beep_freq=1000, beep_duration=0.5, target_sr=16000):
    """
    Detects the start of the beep in the recorded audio using cross-correlation.
    
    Args:
        recorded_file (str): Path to recorded WAV
        beep_freq (float): Frequency of the beep in Hz
        beep_duration (float): Duration of the beep in seconds
        target_sr (int): Sampling rate of audio
        
    Returns:
        int: Sample index where beep starts
        np.ndarray: Recorded audio array
    """
    audio, sr = sf.read(recorded_file)
    if sr != target_sr:
        raise ValueError(f"Sample rate mismatch: {sr} != {target_sr}")
    
    # Generate reference beep
    t = np.linspace(0, beep_duration, int(target_sr * beep_duration), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * beep_freq * t).astype(np.float32)
    
    # Cross-correlation
    corr = correlate(audio, beep, mode='valid')
    start_idx = np.argmax(corr)
    
    return start_idx, audio

def split_recorded_audio(
    audio,
    start_idx,
    metadata_csv,
    output_folder="split_recordings",
    target_sr=16000,
    device="unknown",
    distance="unknown",
    beep_duration=0.5
):
    """
    Splits recorded audio into individual clips using metadata durations,
    renames files to include device, and creates a new metadata CSV with device/distance fields.
    
    Args:
        audio (np.ndarray): Full recorded audio
        start_idx (int): Sample index where beep starts
        metadata_csv (str): Path to original metadata CSV
        output_folder (str): Folder to save split audio
        target_sr (int): Sampling rate
        device (str): Device name to add to filename
        distance (str or float): Distance variable to add to metadata
    """
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(metadata_csv)
    new_metadata = []
    
    current_idx = start_idx + int(beep_duration * target_sr)  # skip the beep
    
    for i, row in df.iterrows():
        duration_samples = int(row['duration_sec'] * target_sr)
        clip = audio[current_idx:current_idx + duration_samples]
        
        
        # Build new filename
        old_filename = os.path.basename(row['filename'])
        new_filename = old_filename.replace("librispeech", device)
        save_path = os.path.join(output_folder, new_filename)
        
        # Save audio clip
        sf.write(save_path, clip, target_sr)
        
        # Update metadata
        meta = row.to_dict()
        meta['filename'] = save_path
        meta['device'] = device
        meta['distance'] = distance
        new_metadata.append(meta)
        
        current_idx += duration_samples
        current_idx += target_sr #for the second of silence
    
    # Save new metadata CSV
    new_metadata_df = pd.DataFrame(new_metadata)
    new_csv_file = os.path.join(output_folder, f"{device}_metadata.csv")
    new_metadata_df.to_csv(new_csv_file, index=False)
    print(f"Saved split audio and metadata to {output_folder}")
    print(f"New metadata CSV: {new_csv_file}")

# Example usage
start_idx, recorded_audio = detect_beep_start("raw_recording_w_beep.wav")
split_recorded_audio(recorded_audio, start_idx, "raw_recordings_metadata.csv", output_folder="data")