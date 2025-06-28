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

For `features.py`: 1 core takes estimated 8 hours so further optimisations made:

Quantisation = halves memory use and doubles inference speed on CPU -> deepcopies HF model so needs ++ weird code -> MAYBE

Fixed-length = stablisises losses but need sliding window or lose prob important data -> MAYBE

Batch = 8 clips per forward pass so roughh -8x time -> YES (in theory) -> in practice, NO because I/O and Librosa + NumPy stuff before the batch is what takes up time -> not letting it run for 8 hours bruh just refactor preproc in morning cus you can parallelise this -> should have planned your files better. im going back to audiomentations man.