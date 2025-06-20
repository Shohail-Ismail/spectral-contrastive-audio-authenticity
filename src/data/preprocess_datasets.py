import os, numpy as np, librosa
from ffmpeg import input as ffmpeg_inp # Avoid namespace ambiguitiy

# Converts raw audio file to 16khz mono wav and mel-spect array
def convert_and_melspectrogram(src_path, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    
    # Extract base filename without extension
    stem = os.path.splitext(os.path.basename(src_path))[0]
    wav = os.path.join(out_dir, stem + ".wav")
    npy = os.path.join(out_dir, stem + ".npy")
    
    # Convert to 16 kHz mono wav
    ffmpeg_inp(src_path).output(wav, ar = 16000, ac = 1).run(quiet = True)
    
    # Loads audio waveform from converted wav
    y, samp_rate = librosa.load(wav,samp_rate = 16000)
    
    # Convert to 64-band mel-spectrogram
    mels = librosa.feature.melspectrogram(y,samp_rate = samp_rate, n_mels = 64)
    
    # Saves mel-spectrogram as binary
    np.save(npy, mels)

if __name__ == "__main__":
    # Processes files
    for subset in ["fakeav", "asvspoof"]:
        raw = f"data/raw/{subset}"
        prep = f"data/preprocessed/{subset}"
        for file in os.listdir(raw):
            if subset == "fakeav" and file.lower().endswith(".mp4"):
                convert_and_melspectrogram(os.path.join(raw, file), prep)
            if subset == "asvspoof" and file.lower().endswith(".wav"):
                convert_and_melspectrogram(os.path.join(raw, file), prep)
    print("----Preprocessing complete----")
