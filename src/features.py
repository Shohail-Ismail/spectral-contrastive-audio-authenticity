import os, csv, numpy as np, librosa, torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Initialise wav2vec2 processor and model for embedding extraction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Puts model into evaluation mode
model.eval()

# Load asvspoof protocol for bonafide and spoofed audio labels
protocol_labels = {}
with open("data/raw/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "r") as protocol_file:
    for line in protocol_file:
        parts = line.strip().split()
        file_id, bonafide_label = parts[1], parts[-1]
        label = 0 if bonafide_label == "bonafide" else 1
        protocol_labels[file_id] = label

# Writes features.csv header
with open("data/features.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    
    # Header row - columns explained in README.md
    header = (["file", "label", "centroid", "entropy"] + [f"mfcc_{i}" for i in range(13)]
        + [f"emb_{i}" for i in range(model.config.hidden_size)])
    writer.writerow(header)

    # Loop over preprocessed mel-spectrograms
    prep_dir = "data/preprocessed/asvspoof"
    all_fns = [fn for fn in os.listdir(prep_dir) if fn.endswith(".npy")]
    total = len(all_fns)
    
    for idx, fn in enumerate(all_fns, start = 1):
        base_name = fn[:-4]
        label = protocol_labels.get(base_name)
        if label is None:
            print(f"SKIPPED -- no label for {base_name}")
            continue

        ## Feature extraction
        npy_path = os.path.join(prep_dir, fn)
        wav_path = os.path.join(prep_dir, base_name + ".wav")

        try:
            # Loads mel-spectrogram
            mels = np.load(npy_path).astype(np.float32)
            
            # Loads waveform
            y, sr = librosa.load(wav_path, sr = 16000)

            # Spectral centroid
            centroid = np.mean(librosa.feature.spectral_centroid(y = y, sr = sr))
            
            # Entropy over mel-bins
            entropy = -np.sum(mels * np.log(mels + 1e-9))
            
            # MFCC (13-dim)
            mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13).mean(axis = 1)

            # Pool frame-wise vectors to get one embedding per file
            inputs = processor(y, sampling_rate = sr, return_tensors = "pt", padding = True)
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
                emb = hidden.mean(dim = 1).cpu().numpy().ravel().astype(np.float32)

            # Write features to csv
            writer.writerow([base_name, label, centroid, entropy] + mfcc.tolist() + emb.tolist())
            print(f"SUCCESS -- {base_name} : [{idx}/{total}] ")

        except Exception as e:
            print(f"FAILED -- {base_name}: {e}")