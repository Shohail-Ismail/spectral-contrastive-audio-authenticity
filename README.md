# spectral-contrastive-audio-authenticity
Audio deepfake detection with novel spectral-contrastive loss


# data acquisition
## FakeAVCeleb
https://www.kaggle.com/datasets/sardertanvirahmed/fakeavcaleb -> data/raw/fakeav

## ASVspoof2019 LA dev-set
https://datashare.ed.ac.uk/handle/10283/3336 -> LA.zip -> data/rawasvspoof


### Header:
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

## Baseline training

### train_test_split
Used to ensure random shuffling before split to avoid bias (reproducibility included, seed = 10)

### Feature scaling
Applied to training and test to bring centroid, mfccs and embeddings onto a common scale for better logreg convergence

### 5 fold CV grid search
C vals = 0.01, 0.1, 1.0 using GridSearchCV so mean ROC AUC across 5 folds to give best C value. This optimal C is then used to refit model on full training set.