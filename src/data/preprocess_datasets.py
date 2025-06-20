import os, numpy as np, librosa
from ffmpeg import input as ffmpeg_inp

# Number of files to process for quick testing
SUBSET = 10

# Converts raw audio file to 16khz mono wav and mel-spectrogram array
def convert_and_melspectrogram(src_path, out_dir):
    os.makedirs(out_dir, exist_ok  = True)
    
    # Extract base filename without extension
    stem = os.path.splitext(os.path.basename(src_path))[0]
    wav = os.path.join(out_dir, stem + ".wav")
    npy = os.path.join(out_dir, stem + ".npy")

    # Convert to 16 kHz mono wav
    ffmpeg_inp(src_path).output(wav, ar = 16000, ac = 1).run(quiet = True)

    # Load and compute 64-band mel-spectrogram, and save as binary array
    y, sr = librosa.load(wav, sr = 16000)
    mels = librosa.feature.melspectrogram(y, sr = sr, n_mels = 64)
    np.save(npy, mels)

    print(f" -- Processed {stem} -- ")

if __name__ == "__main__":
    
    # FakeAVCeleb
    raw = "data/raw/fakeav"
    prep = "data/preprocessed/fakeav"
    files = [
        fn for fn in os.listdir(raw)
        if fn.lower().endswith((".mp4", ".wav"))
    ]
    files = files[:SUBSET]
    for fn in files:
        convert_and_melspectrogram(os.path.join(raw, fn), prep)
    
    #ASVspoof2019 LA dev-set
    raw_asv = "data/raw/asvspoof/LA/ASVspoof2019_LA_dev/flac"
    prep_asv = "data/preprocessed/asvspoof"
    files_asv = [
        fn for fn in os.listdir(raw_asv)
        if fn.lower().endswith(".flac")
    ]
    files_asv = files_asv[:SUBSET]
    for fn in files_asv:
        convert_and_melspectrogram(os.path.join(raw_asv, fn), prep_asv)
    
    print("--- Preprocessing complete ---")
