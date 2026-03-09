# Spectral Contrastive Audio Authenticity
This repo documents my independent, research-oriented project on detecting audio deepfakes, building on my work in AI mis/disinformation detection (such as AtomicDeFake). 

The project is a CPU-only research prototype for audio deepfake detection using frozen Wav2Vec2 embeddings, spectral features, and an in-progress contrastive training branch. The current repo includes a validated logreg baseline, with downstream contrastive evaluation/testing remains in progress.

 --  WIP --

## Datasets
### FakeAVCeleb
https://github.com/DASH-Lab/FakeAVCeleb -> data/raw/fakeav

### ASVspoof2019 LA dev-set
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset -> LA.zip -> data/rawasvspoof -> data/raw/fakeav


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
