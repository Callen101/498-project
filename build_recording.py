import os
import random
import librosa
from datasets import load_dataset
import soundfile as sf
import pandas as pd
import numpy as np

root = "./data"
N = 100
target_sr = 16000
csv_file = "raw_recordings_metadata.csv"

def download_and_sample_audio(n, split, target_sr=16000):
    """
    Downloads n*10 audio samples from LibriSpeech (train.clean.100) 
    and returns a random sample of n audio arrays with their sampling rates.
    
    Returns:
        List of tuples: [(audio_array, sampling_rate), ...] of length n
    """
    dataset = load_dataset(
        "librispeech_asr",
        split=split,
        streaming=True
    )
    
    # Download n*10 samples
    audio_list = []
    for i, sample in enumerate(dataset):
        if i >= n * 10:
            break
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        audio_list.append((audio, sr, sample, split))
    
    # Randomly select n samples
    sampled_audio = random.sample(audio_list, n)
    
    return sampled_audio

def collect_files():
    test_clean_pct = 0.2
    test_other_pct = 0.1
    train_clean_pct = 0.5
    train_other_pct = 0.2
    sampled_audio = []
    
    sampled_audio = sampled_audio + download_and_sample_audio(int(test_clean_pct*N), "test.clean", target_sr=target_sr)
    sampled_audio = sampled_audio + download_and_sample_audio(int(test_other_pct*N), "test.other", target_sr=target_sr)
    sampled_audio = sampled_audio + download_and_sample_audio(int(train_clean_pct*N), "train.clean.100", target_sr=target_sr)
    sampled_audio = sampled_audio + download_and_sample_audio(int(train_other_pct*N), "train.other.500", target_sr=target_sr)

    return sampled_audio

def download_files(audio):
    metadata = []
    print(len(audio))
    print(audio[0])
    print(len(audio[0][0]))
    for idx, (wav, sr, sample, split) in enumerate(audio):
        id = sample['id'].replace("-", "_").replace("'","").replace("(","").replace(")","").replace(",","")
        speaker_id = sample['speaker_id']
        chapter_id = sample['chapter_id']
        text = sample['text']
        filename = os.path.join(root, "raw_recordings", f"librispeech_{id}.wav")
        print(sr)
        print(split)
        sf.write(filename, wav, sr)

        # Collect metadata
        meta = {
            "filename": filename,
            "shape": wav.shape,
            "sampling_rate": sr,
            "duration_sec": len(wav) / sr,
            "split": split,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "id": id,
            "text": text
        }
        metadata.append(meta)

    # Save metadata to CSV
    df = pd.DataFrame(metadata)
    df.to_csv(csv_file, index=False)
    print(f"Saved metadata to {csv_file}")

def concatenate_sampled_audio(sampled_audio, output_file="concatenated.wav", target_sr=16000, silence_sec=1.0):
    """
    Concatenates audio arrays from sampled_audio with silence in between.

    Args:
        sampled_audio (list): List of tuples (audio_array, sr, sample_dict)
        output_file (str): Path to save the concatenated WAV
        target_sr (int): Sampling rate for silence padding
        silence_sec (float): Seconds of silence between each audio
    """
    silence = np.zeros(int(target_sr * silence_sec), dtype=np.float32)
    concatenated_audio = []

    for idx, (audio, sr, sample, split) in enumerate(sampled_audio):
        # Ensure the audio is float32 for writing
        audio = audio.astype(np.float32)
        
        # Add audio
        concatenated_audio.append(audio)
        
        # Add silence if not the last audio
        if idx < len(sampled_audio) - 1:
            concatenated_audio.append(silence)

    final_audio = np.concatenate(concatenated_audio)
    sf.write(output_file, final_audio, target_sr)
    print(f"Saved concatenated WAV to {output_file} (duration: {len(final_audio)/target_sr:.2f} sec)")
    return final_audio

def add_beep_to_audio(audio_array, output_file="beep.wav", target_sr=16000, beep_freq=1000, beep_duration=0.5):
    """
    Prepends a beep to an audio array.
    
    Args:
        audio_array (np.ndarray): Original audio
        target_sr (int): Sampling rate
        beep_freq (float): Frequency of the beep in Hz
        beep_duration (float): Duration of the beep in seconds
        
    Returns:
        np.ndarray: Audio with beep prepended
    """
    silence = np.zeros(int(target_sr * 5), dtype=np.float32)
    t = np.linspace(0, beep_duration, int(target_sr * beep_duration), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * beep_freq * t).astype(np.float32)
    final_audio = np.concatenate([silence, beep, audio_array])
    sf.write(output_file, final_audio, target_sr)
    print(f"Saved concatenated WAV to {output_file} (duration: {len(final_audio)/target_sr:.2f} sec)")

audio_files = collect_files()
download_files(audio_files)
cat_audio = concatenate_sampled_audio(audio_files, output_file="raw_recording.wav")
add_beep_to_audio(cat_audio, output_file="raw_recording_w_beep.wav")

