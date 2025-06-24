import os, subprocess, numpy as np, librosa

# Number of files to process for quick testing per run
SUBSET = 10

# Secs before ffmpeg conversion gives up
TIMEOUT = 30

# Converts raw audio file to 16 kHz mono WAV and mel-spectrogram array
def convert_and_melspectrogram(src_path, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    wav_path = os.path.join(out_dir, stem + ".wav")
    npy_path = os.path.join(out_dir, stem + ".npy")

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
        print(f"[TIMED OUT] ---- {stem}")
        return False
    except subprocess.CalledProcessError:
        print(f"[FAILED] ---- {stem}")
        return False

    # Load wav and compute mel-spectrogram
    try:
        y, sr = librosa.load(wav_path, sr = 16000)
        mels = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 64)
        np.save(npy_path, mels)
    except Exception as e:
        print(f"[ERROR] ---- {stem} → {e}")
        return False

    print(f"[PASS] ---- {stem}")
    return True

if __name__ == "__main__":
    # # FakeAVCeleb
    # raw_root = "data/raw/fakeav"
    # prep_dir = "data/preprocessed/fakeav"
    
    # # Determine which stems have already been processed
    # done = {os.path.splitext(f)[0] for f in os.listdir(prep_dir) if f.endswith(".npy")}

    # # Gather all .mp4/.wav files under raw_root
    # candidates = []
    # for dp, _, files in os.walk(raw_root):
    #     for fn in files:
    #         if fn.lower().endswith((".mp4", ".wav")):
    #             stem = os.path.splitext(fn)[0]
    #             if stem not in done:
    #                 candidates.append(os.path.join(dp, fn))

    # print(f"\n### {len(candidates)} FAKEAV. Processing {SUBSET}")
    
    # i = 0
    # for src in candidates:
    #     if i >= SUBSET:
    #         break
    #     if convert_and_melspectrogram(src, prep_dir):
    #         i += 1

    # ASVspoof2019 LA dev-set
    raw_asv  = "data/raw/asvspoof/LA/ASVspoof2019_LA_dev/flac"
    prep_asv = "data/preprocessed/asvspoof"
    done_asv = {os.path.splitext(f)[0] for f in os.listdir(prep_asv) if f.endswith(".npy")}

    candidates_asv = []
    for dp, _, files in os.walk(raw_asv):
        for fn in files:
            if fn.lower().endswith(".flac"):
                stem = os.path.splitext(fn)[0]
                if stem not in done_asv:
                    candidates_asv.append(os.path.join(dp, fn))

    print(f"\n### {len(candidates_asv)} LADEVSET. Processing {SUBSET}")
    
    i = 0
    for src in candidates_asv:
        if i >= SUBSET:
            break
        if convert_and_melspectrogram(src, prep_asv):
            i += 1

    print("\n--- Preprocessing complete ---")
