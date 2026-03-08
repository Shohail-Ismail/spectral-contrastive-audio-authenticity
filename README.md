# Spectral Contrastive Audio Authenticity
Audio deepfake detection with novel spectral-contrastive loss.

WIP - paused from ~07/2025, resuming ~ 04/26 | 05/26

## Data Acquisition
### FakeAVCeleb
https://github.com/DASH-Lab/FakeAVCeleb -> data/raw/fakeav

### ASVspoof2019 LA dev-set
https://huggingface.co/datasets/LanceaKing/asvspoof2019 -> LA.zip -> data/rawasvspoof -> data/raw/fakeav

---

# Rough notes

## Implementation

## Header:
**file** = ID from asvspoof2019 LA dev set (e.g. LA_0001)

**label** = 0 = real human speech , 1 = spoofed audio (as specified in `ASVspoof2019.LA.cm.dev.trl.txt`) 

**centroid** = Mean spectral centroid (timbral sharpness - mean audio brightness)

**entropy** = entropy over mel-spectrogram bins (signal complexity)

**mfcc_0...mfcc_12** = average mfcc (13 values)

**emb_0...emb_767** = wav2vec2 embedding (768-dim vector)

## Parallelism

24844 files so parallelism used to speed things up - done before creating training script

For `preprocess_datasets.py`: reduced time from estimated 8 hours to 40 mins on 11 cores (estimate = time taken for 100 * 2500 (25000 files))

For `features.py`: 1 core takes estimated 8 hours so further optimisations needed -> trialling further optimisations only slowed process down due to inference dominating runtime, some preprocessing still running serially, and padding to the longest clip negating batching/quantisation gains -> decided to run feature extraction overnight as efficiency in this step is not critical to core research.

Final iteration could be optimised by refactoring preprocess_datasets.py to absorb file-loading and numpy work from features.py -> decreases time drastically

## Contrastive training

Frozen wav2vec to skip backprop over 300M params (takes estimated training time from ~40 hours to ~20). May gain statistically significant AUC points so shelf that for later when GPU => good way might be to:
    - run every file through, fit a logreg classifier and report roc-auc
    - in second run, unfreeze last 1 or 2 transformer layers and fine tune those + the 128-dim projection head -> compare roc auc to see if >=1% change at least from baseline 0.9744
