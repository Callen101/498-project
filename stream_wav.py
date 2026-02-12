import torch
torch._dynamo.disable()
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from audioseal import AudioSeal
import os
import soundfile as sf
import time
# Where LibriSpeech will be stored (or is already stored)
root = "./data"

# Pick a split
dataset = LIBRISPEECH(
    root=root,
    url="test-clean",   # or: train-clean-100, train-clean-360, dev-clean, etc.
    download=True
)

# Use dataset metadata safely
_, _, _, speaker_id, chapter_id, utterance_id = dataset.get_metadata(0)

wav_path = os.path.join(
    root,
    "LibriSpeech",
    "test-clean",
    f"{speaker_id}",
    f"{chapter_id}",
    f"{speaker_id}-{chapter_id}-{utterance_id:04}.flac"
)

# Load wav manually
wav, sr = sf.read(wav_path, dtype="float32")

sf.write("speech_sample.wav", wav, sr)
wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
print(wav.shape)     # (batch, channels, samples)
print(sr)        # usually 16000

model = AudioSeal.load_generator("audioseal_wm_streaming")
model.eval()

streaming_watermarked_audio = []
times = []
secret_mesage = torch.randint(0, 2, (1, 16), dtype=torch.int32)

with model.streaming(batch_size=1):
    
    # Simulate the streaming input by splitting the audio into chunks
    chunk_size = int(sr * 2)  # 2-second chunks
    for start in range(0, wav.size(2), chunk_size):
        end = min(start + chunk_size, wav.size(2))
        chunk = wav[:, :, start:end]
        start_time = time.perf_counter()
        watermarked_chunk = model(chunk, sample_rate=sr, message=secret_mesage, alpha=1)
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        streaming_watermarked_audio.append(watermarked_chunk)
    
streaming_watermarked_audio = torch.cat(streaming_watermarked_audio, dim=2)
sf.write("encoded_audio.wav", streaming_watermarked_audio.detach().squeeze(0).squeeze(0).numpy(), sr)


# # You can detect a chunk of watermarked output, or the whole audio:

detector = AudioSeal.load_detector("audioseal_detector_streaming")
detector.eval()

full_result, message = detector.detect_watermark(streaming_watermarked_audio)
print(full_result)
print(secret_mesage)
print(message)  
print(times)


