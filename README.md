# Spectral Contrastive Audio Authenticity
Audio deepfake detection with a spectral + contrastive representation-learning pipeline.

WIP - paused from ~07/2025, resumed 03/26

## Datasets
### FakeAVCeleb
https://github.com/DASH-Lab/FakeAVCeleb -> data/raw/fakeav

### ASVspoof2019 LA dev-set
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset -> LA.zip -> data/rawasvspoof -> data/raw/fakeav

## Summary
This repo documents my independent, research-oriented project on detecting audio deepfakes, building on my work in AI mis/disinformation detection (such as AtomicDeFake), for both learning and to evaluate the usefulness of this _________

Implemented so far:
- End-to-end preprocessing and feature generation for ASVspoof2019 (25K+ files) 
- Classical spectral descriptors per file (spectral centroid, mel-entropy proxy, MFCC means)
- 768-dim wav2vec2 pooled embeddings
- Baseline classifier training with feature scaling, 5-fold CV hyperparameter search, and held-out evaluation
- Contrastive representation training with a frozen wav2vec2 backbone +trainable projection head

Current status:
- Baseline metrics are recorded and reproducible from the current scripts.
- Contrastive training loop is implemented and checkpoints/metrics are written.

In-progress:
- Final comparative evaluation suite (full ablations + downstream contrastive ROC AUC comparison)


# Rough notes
## Implementation

## Header:
**file** = ID from asvspoof2019 LA dev set (e.g. LA_0001)

**label** = 0 = real human speech , 1 = spoofed audio (as specified in `ASVspoof2019.LA.cm.dev.trl.txt`) 

**centroid** = Mean spectral centroid (timbral sharpness - mean audio brightness)

**entropy** = entropy over mel-spectrogram bins (signal complexity)

**mfcc_0...mfcc_12** = average mfcc (13 values)


## Parallelism

24K+ files so parallelism used to speed things up - done before creating training script

For `preprocess_datasets.py`: reduced time from estimated 8 hours to 40 mins on 11 cores (estimate = time taken for 100 * 2500 (25000 files))

For `features.py`: 1 core takes estimated 8 hours so further optimisations needed -> trialling further optimisations only slowed process down due to inference dominating runtime, some preprocessing still running serially, and padding to the longest clip negating batching/quantisation gains -> decided to run feature extraction overnight as efficiency in this step is not critical to core research.

Final iteration could be optimised by refactoring preprocess_datasets.py to absorb file-loading and numpy work from features.py -> decreases time drastically

## Contrastive training

Frozen wav2vec to skip backprop over 300M params (takes estimated training time from ~40 hours to ~20). May gain statistically significant AUC points so shelf that for later when GPU => good way might be to:
    - run every file through, fit a logreg classifier and report roc-auc
    - in second run, unfreeze last 1 or 2 transformer layers and fine tune those + the 128-dim projection head -> compare roc auc to see if >=1% change at least from baseline 0.9744
