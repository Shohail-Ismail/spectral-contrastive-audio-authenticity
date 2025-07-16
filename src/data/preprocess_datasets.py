import os, subprocess, numpy as np, librosa, multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Secs before ffmpeg conversion gives up
TIMEOUT = 30

# Paths to ASVspoof
RAW = "data/raw/asvspoof/LA/ASVspoof2019_LA_dev/flac"
PREP = "data/preprocessed/asvspoof"

# Keeps count of already done files
done_asv = {os.path.splitext(f)[0] for f in os.listdir(PREP) if f.endswith(".npy")}
print(f"Skipping {len(done_asv)} as already preprocessed")


# Converts raw audio file to 16 kHz mono WAV and mel-spectrogram array
def convert_and_melspectrogram(src_path):
    stem = os.path.splitext(os.path.basename(src_path))[0]
    
    # Avoid redoing prep'd files
    if stem in done_asv:
        return stem, True
    
    wav_path = os.path.join(PREP, stem + ".wav")
    npy_path = os.path.join(PREP, stem + ".npy")

    # Extract audio only (-vn), resample to 16 kHz mono
    cmd = [
        "ffmpeg", "-i", src_path,
        "-vn",               # drop video track
        "-ar", "16000",      # sample rate
        "-ac", "1",          # 1 channel
        wav_path,
        "-y"                 # overwrite if exists
    ]
    
    try:
        subprocess.run(
            cmd,
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMED OUT] -- {stem}")
        return stem, False
    except subprocess.CalledProcessError:
        print(f"[FAILED] -- {stem}")
        return stem, False

    # Load wav and compute mel-spectrogram
    try:
        y, sr = librosa.load(wav_path, sr = 16000)
        mels = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 64)
        np.save(npy_path, mels)
    except Exception as e:
        return stem, False

    return stem, True


if __name__ == "__main__":
    
    # List of ASVspoof2019 LA dev-set files to process
    candidates_asv = []
    for dp, _, files in os.walk(RAW):
        for fn in files:
            if fn.lower().endswith(".flac"):
                candidates_asv.append(os.path.join(dp, fn))

    # Parallelisation
    # Use max cores
    NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
    print(f"-- Using {NUM_WORKERS} cores --")
    
    # Run conversions in parallel
    with ProcessPoolExecutor(max_workers = NUM_WORKERS) as exe:
        for idx, (stem, success) in enumerate(exe.map(convert_and_melspectrogram, candidates_asv), start = 1):
            result = "PASS" if success else "FAIL"
            print(f"{result}: {stem} ({idx}/{len(candidates_asv)})")